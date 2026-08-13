from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TwinFrame:
    progress: float
    product_x: float
    product_y: float
    gate_angle: float
    actuator_progress: float
    state: str


def twin_frame(decision: str, progress: float) -> TwinFrame:
    progress = min(1.0, max(0.0, float(progress)))
    decision = decision.upper()
    if decision not in {"PASS", "REJECT"}:
        raise ValueError("decision must be PASS or REJECT")

    product_x = 4.0 + progress * 88.0
    product_y = 50.0
    gate_angle = 0.0
    actuator = 0.0
    state = "CONVEYING"

    if decision == "PASS":
        if progress >= 0.7:
            state = "PASS_LANE"
        if progress >= 1.0:
            state = "COMPLETE"
    else:
        if 0.35 <= progress < 0.55:
            gate_angle = -55.0 * ((progress - 0.35) / 0.20)
            actuator = (progress - 0.35) / 0.20 * 50.0
            state = "GATE_EXTENDING"
        elif 0.55 <= progress < 0.78:
            gate_angle = -55.0
            actuator = 50.0 + (progress - 0.55) / 0.23 * 25.0
            product_x = 52.4 + (progress - 0.55) / 0.23 * 12.0
            product_y = 50.0 + (progress - 0.55) / 0.23 * 35.0
            state = "DIVERTING"
        elif progress >= 0.78:
            gate_angle = -55.0 * max(0.0, 1.0 - (progress - 0.78) / 0.22)
            actuator = 75.0 + (progress - 0.78) / 0.22 * 25.0
            product_x = 64.4
            product_y = 85.0
            state = "REJECT_BIN" if progress >= 0.95 else "GATE_RETRACTING"
    return TwinFrame(
        progress=progress,
        product_x=round(product_x, 2),
        product_y=round(product_y, 2),
        gate_angle=round(gate_angle, 2),
        actuator_progress=round(min(100.0, actuator), 1),
        state=state,
    )


def twin_timeline(decision: str, frame_count: int = 61) -> tuple[TwinFrame, ...]:
    if frame_count < 2:
        raise ValueError("frame_count must be at least 2")
    return tuple(twin_frame(decision, index / (frame_count - 1)) for index in range(frame_count))
