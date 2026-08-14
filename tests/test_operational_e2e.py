from qcell.auth import MANAGE_CAPA, CredentialStore
from qcell.quality_analytics import build_quality_report, generate_demo_shift
from qcell.traceability import TraceabilityStore


def test_quality_alert_to_capa_closure_and_product_audit(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QCELL_AUTH_MODE", "demo")
    credentials = CredentialStore.from_environment()
    quality_manager = credentials.authenticate("quality", "qcell-quality")
    assert quality_manager is not None and quality_manager.can(MANAGE_CAPA)

    events = generate_demo_shift(240, seed=23)
    report = build_quality_report(
        events,
        subgroup_size=20,
        target_defect_rate_percent=5.0,
        latency_sla_ms=50.0,
        source="e2e-shift",
        generated_at="2026-08-14T12:00:00+00:00",
    )
    assert {alert["code"] for alert in report["alerts"]} == {
        "SPC_OUT_OF_CONTROL",
        "DEFECT_RATE_HIGH",
    }

    trace_store = TraceabilityStore(tmp_path / "traceability.db")
    for event in events[:8]:
        trace_store.record_inspection(
            event,
            model_version="deep-patchcore-v3",
            actor="operator",
        )

    created_cases = []
    for alert in report["alerts"]:
        case, created = trace_store.create_capa(
            alert_code=str(alert["code"]),
            severity=str(alert["severity"]),
            title=str(alert["title"]),
            description=str(alert["detail"]),
            actor=quality_manager.username,
            lot_id="LOT-A,LOT-B,LOT-C",
            owner=quality_manager.username,
            dedupe_key=f"e2e:{alert['code']}",
        )
        assert created is True
        created_cases.append(case)

    selected = created_cases[0]
    trace_store.transition_capa(
        selected.case_id,
        "ACKNOWLEDGED",
        actor=quality_manager.username,
    )
    trace_store.transition_capa(
        selected.case_id,
        "IN_PROGRESS",
        actor=quality_manager.username,
    )
    trace_store.transition_capa(
        selected.case_id,
        "VERIFIED",
        actor=quality_manager.username,
        root_cause="Sealer temperature drift",
        corrective_action="Recalibrated sealer and verified 30 samples",
    )
    closed = trace_store.transition_capa(
        selected.case_id,
        "CLOSED",
        actor=quality_manager.username,
    )

    assert closed.status == "CLOSED"
    assert trace_store.metrics()["products"] == 8
    assert trace_store.metrics()["open_capa"] == 1
    assert len(trace_store.timeline(str(events[0]["product_id"]))) == 2
    audit_actions = {entry["action"] for entry in trace_store.audit_entries()}
    assert audit_actions == {
        "TRACE_EVENT_RECORDED",
        "CAPA_CREATED",
        "CAPA_TRANSITIONED",
    }
