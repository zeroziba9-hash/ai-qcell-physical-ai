from qcell.ros2_pipeline import PipelineInspection, simulate_sort_pipeline


def make_inspection(is_defect: bool) -> PipelineInspection:
    return PipelineInspection(
        product_id="QCELL-0001",
        image_path="sample.png",
        defect_type="contamination" if is_defect else "good",
        is_defect=is_defect,
        anomaly_score=70.0 if is_defect else 30.0,
        raw_score=0.7 if is_defect else 0.3,
        threshold=0.4,
        latency_ms=42.0,
    )


def test_reject_pipeline_emits_action_feedback_and_result():
    run = simulate_sort_pipeline(make_inspection(True))

    assert run.decision == "REJECT"
    assert run.final_state == "REJECT_BIN"
    assert run.actuator_progress == (25, 50, 75, 100)
    assert any(event.interface == "ACTION_GOAL" for event in run.events)
    assert run.events[-1].node == "dashboard_bridge"


def test_pass_pipeline_skips_reject_action():
    run = simulate_sort_pipeline(make_inspection(False))

    assert run.decision == "PASS"
    assert run.final_state == "PASS_LANE"
    assert run.actuator_progress == ()
    assert all(not event.interface.startswith("ACTION") for event in run.events)
    assert any(event.channel == "/qcell/sort/pass" for event in run.events)


def test_event_rows_can_replay_a_prefix():
    run = simulate_sort_pipeline(make_inspection(True))

    rows = run.event_rows(through_step=3)
    assert [row["step"] for row in rows] == [1, 2, 3]
