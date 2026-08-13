from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class PipelineInspection:
    """Model output passed from the inspection node to the decision node."""

    product_id: str
    image_path: str
    defect_type: str
    is_defect: bool
    anomaly_score: float
    raw_score: float
    threshold: float
    latency_ms: float


@dataclass(frozen=True)
class PipelineEvent:
    step: int
    offset_ms: int
    node: str
    interface: str
    channel: str
    event: str
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PipelineRun:
    inspection: PipelineInspection
    decision: str
    final_state: str
    events: tuple[PipelineEvent, ...]

    @property
    def actuator_progress(self) -> tuple[int, ...]:
        values: list[int] = []
        for event in self.events:
            if event.interface == "ACTION_FEEDBACK":
                values.append(int(event.detail.split("%", 1)[0]))
        return tuple(values)

    def event_rows(self, through_step: int | None = None) -> list[dict[str, object]]:
        events: Iterable[PipelineEvent] = self.events
        if through_step is not None:
            events = (event for event in events if event.step <= through_step)
        return [event.to_dict() for event in events]


def simulate_sort_pipeline(inspection: PipelineInspection) -> PipelineRun:
    """Create the same observable topic/action flow used by the ROS2 nodes."""

    events: list[PipelineEvent] = []

    def emit(
        offset_ms: int,
        node: str,
        interface: str,
        channel: str,
        event: str,
        detail: str,
    ) -> None:
        events.append(
            PipelineEvent(
                step=len(events) + 1,
                offset_ms=offset_ms,
                node=node,
                interface=interface,
                channel=channel,
                event=event,
                detail=detail,
            )
        )

    emit(0, "camera_node", "TOPIC", "/qcell/camera/product", "PUBLISH", inspection.product_id)
    emit(15, "inspection_node", "TOPIC", "/qcell/camera/product", "RECEIVE", inspection.image_path)
    emit(
        15 + int(inspection.latency_ms),
        "inspection_node",
        "TOPIC",
        "/qcell/inspection/result",
        "PUBLISH",
        f"score={inspection.raw_score:.4f}, threshold={inspection.threshold:.4f}",
    )
    emit(
        25 + int(inspection.latency_ms),
        "decision_node",
        "TOPIC",
        "/qcell/inspection/result",
        "RECEIVE",
        "REJECT" if inspection.is_defect else "PASS",
    )

    if inspection.is_defect:
        action_start = 35 + int(inspection.latency_ms)
        emit(
            action_start,
            "decision_node",
            "ACTION_GOAL",
            "/qcell/reject_product",
            "SEND",
            f"{inspection.product_id}: anomaly score exceeded threshold",
        )
        for index, (progress, state) in enumerate(
            ((25, "gate preparing"), (50, "gate extending"), (75, "product diverting"), (100, "gate retracted")),
            start=1,
        ):
            emit(
                action_start + index * 120,
                "reject_action_server",
                "ACTION_FEEDBACK",
                "/qcell/reject_product",
                "FEEDBACK",
                f"{progress}% · {state}",
            )
        emit(
            action_start + 600,
            "reject_action_server",
            "ACTION_RESULT",
            "/qcell/reject_product",
            "SUCCEED",
            "REJECT_BIN",
        )
        final_state = "REJECT_BIN"
        decision = "REJECT"
    else:
        emit(
            35 + int(inspection.latency_ms),
            "decision_node",
            "TOPIC",
            "/qcell/sort/pass",
            "PUBLISH",
            inspection.product_id,
        )
        final_state = "PASS_LANE"
        decision = "PASS"

    emit(
        events[-1].offset_ms + 10,
        "dashboard_bridge",
        "TOPIC",
        "/qcell/dashboard/event",
        "UPDATE",
        final_state,
    )
    return PipelineRun(inspection, decision, final_state, tuple(events))
