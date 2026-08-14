from datetime import datetime, timedelta, timezone
import json

import pytest

from qcell.quality_analytics import (
    QualityReportStore,
    build_control_chart,
    build_defect_pareto,
    build_quality_report,
    generate_demo_shift,
    quality_events_to_csv,
    quality_report_to_json,
    summarize_quality,
)


def _event(
    index: int,
    *,
    result: str = "PASS",
    defect_type: str = "none",
    latency_ms: float = 20.0,
    confidence: float = 0.96,
) -> dict[str, object]:
    timestamp = datetime(2026, 8, 14, 8, 0, tzinfo=timezone.utc) + timedelta(
        seconds=index * 10
    )
    return {
        "timestamp": timestamp.isoformat(timespec="seconds"),
        "product_id": f"Q-{index:04d}",
        "result": result,
        "defect_type": defect_type,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "action": "REJECT" if result == "DEFECT" else "PASS_THROUGH",
        "lot_id": "LOT-A",
    }


def test_summary_and_pareto_are_calculated_from_inspection_events() -> None:
    events = [_event(index) for index in range(40)]
    events.extend(
        _event(
            40 + index,
            result="DEFECT",
            defect_type="scratch" if index < 6 else "contamination",
            latency_ms=60.0,
        )
        for index in range(10)
    )

    summary = summarize_quality(events)
    pareto = build_defect_pareto(events)

    assert summary.inspected == 50
    assert summary.passed == 40
    assert summary.defects == 10
    assert summary.first_pass_yield_percent == 80.0
    assert summary.defect_rate_percent == 20.0
    assert summary.defect_ppm == 200_000
    assert summary.p95_latency_ms == 60.0
    assert pareto == [
        {
            "defect_type": "scratch",
            "count": 6,
            "share_percent": 60.0,
            "cumulative_percent": 60.0,
        },
        {
            "defect_type": "contamination",
            "count": 4,
            "share_percent": 40.0,
            "cumulative_percent": 100.0,
        },
    ]


def test_control_chart_and_rule_engine_detect_process_drift() -> None:
    events = [_event(index) for index in range(40)]
    events.extend(
        _event(40 + index, result="DEFECT", defect_type="scratch", latency_ms=65.0)
        for index in range(10)
    )

    points = build_control_chart(events, subgroup_size=10)
    report = build_quality_report(
        events,
        subgroup_size=10,
        target_defect_rate_percent=5.0,
        latency_sla_ms=40.0,
        generated_at="2026-08-14T09:00:00+00:00",
    )
    alert_codes = {alert["code"] for alert in report["alerts"]}

    assert len(points) == 5
    assert points[-1].defect_rate_percent == 100.0
    assert points[-1].out_of_control is True
    assert report["generated_at"] == "2026-08-14T09:00:00+00:00"
    assert alert_codes == {
        "SPC_OUT_OF_CONTROL",
        "DEFECT_RATE_HIGH",
        "LATENCY_SLA_MISS",
    }


def test_demo_shift_is_reproducible_and_contains_drift_context() -> None:
    started = datetime(2026, 8, 14, 8, 0, tzinfo=timezone.utc)
    first = generate_demo_shift(30, seed=91, start_at=started)
    second = generate_demo_shift(30, seed=91, start_at=started)

    assert first == second
    assert len(first) == 30
    assert {event["process_phase"] for event in first} == {"baseline", "drift"}
    assert first[0]["lot_id"] == "LOT-A"
    assert first[-1]["product_id"] == "SHIFT-00030"


def test_csv_json_exports_and_report_store_round_trip(tmp_path) -> None:
    events = generate_demo_shift(24, seed=5)
    report = build_quality_report(
        events, generated_at="2026-08-14T10:00:00+00:00"
    )
    csv_text = quality_events_to_csv(events)
    report_text = quality_report_to_json(report)
    store = QualityReportStore(tmp_path / "quality-reports")

    older = store.save(
        report,
        name="A조",
        saved_at="2026-08-14T10:01:00+00:00",
    )
    newer = store.save(
        report,
        name="B조",
        saved_at="2026-08-14T11:01:00+00:00",
    )
    snapshots = store.list()

    assert "product_id" in csv_text
    assert "SHIFT-00001" in csv_text
    assert json.loads(report_text)["summary"]["inspected"] == 24
    assert snapshots[0]["snapshot_id"] == newer["snapshot_id"]
    assert snapshots[1]["snapshot_id"] == older["snapshot_id"]
    assert json.loads(store.manifest_path.read_text(encoding="utf-8"))["schema_version"] == 1


@pytest.mark.parametrize(
    ("event_update", "message"),
    [
        ({"result": "UNKNOWN"}, "result must be PASS or DEFECT"),
        ({"confidence": 1.2}, "confidence must be between 0 and 1"),
        ({"latency_ms": -1}, "latency_ms must not be negative"),
    ],
)
def test_invalid_events_are_rejected(event_update, message) -> None:
    event = {**_event(1), **event_update}

    with pytest.raises(ValueError, match=message):
        summarize_quality([event])
