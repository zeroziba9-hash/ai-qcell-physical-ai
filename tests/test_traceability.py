import json

import pytest

from qcell.traceability import TraceabilityStore, trace_events_to_csv


def _inspection(product_id: str = "Q-0001", *, defect: bool = True):
    return {
        "timestamp": "2026-08-14T08:00:00+00:00",
        "product_id": product_id,
        "lot_id": "LOT-X",
        "result": "DEFECT" if defect else "PASS",
        "defect_type": "scratch" if defect else "none",
        "confidence": 0.94,
        "latency_ms": 27.4,
        "action": "REJECT" if defect else "PASS_THROUGH",
    }


def test_inspection_builds_product_genealogy_and_audit_trail(tmp_path) -> None:
    store = TraceabilityStore(tmp_path / "trace.db")

    inspection, sorting = store.record_inspection(
        _inspection(), model_version="model-v3", actor="operator"
    )
    products = store.products()
    timeline = store.timeline("Q-0001")

    assert inspection.stage == "VISION"
    assert sorting.stage == "SORTING"
    assert products[0].product_id == "Q-0001"
    assert products[0].latest_status == "REJECT"
    assert [event.status for event in timeline] == ["DEFECT", "REJECT"]
    assert store.metrics() == {
        "products": 1,
        "rejected": 1,
        "open_capa": 0,
        "audit_entries": 2,
    }
    assert len(store.audit_entries()) == 2
    assert "AI_INSPECTION" in trace_events_to_csv(timeline)


def test_capa_follows_controlled_transitions_and_deduplicates(tmp_path) -> None:
    store = TraceabilityStore(tmp_path / "trace.db")
    store.record_inspection(_inspection(), actor="operator")

    case, created = store.create_capa(
        alert_code="SPC_OUT_OF_CONTROL",
        severity="critical",
        title="공정 이탈",
        description="관리한계 초과",
        actor="quality",
        product_id="Q-0001",
        lot_id="LOT-X",
        dedupe_key="LOT-X:SPC",
    )
    duplicate, duplicate_created = store.create_capa(
        alert_code="SPC_OUT_OF_CONTROL",
        severity="critical",
        title="중복",
        description="중복",
        actor="quality",
        lot_id="LOT-X",
        dedupe_key="LOT-X:SPC",
    )

    assert created is True
    assert duplicate_created is False
    assert duplicate.case_id == case.case_id
    acknowledged = store.transition_capa(
        case.case_id, "ACKNOWLEDGED", actor="quality", owner="operator"
    )
    in_progress = store.transition_capa(
        case.case_id, "IN_PROGRESS", actor="operator"
    )
    verified = store.transition_capa(
        case.case_id,
        "VERIFIED",
        actor="quality",
        root_cause="조명 광량 저하",
        corrective_action="조명 교체 및 재교정",
    )
    closed = store.transition_capa(case.case_id, "CLOSED", actor="quality")

    assert acknowledged.status == "ACKNOWLEDGED"
    assert in_progress.status == "IN_PROGRESS"
    assert verified.status == "VERIFIED"
    assert closed.status == "CLOSED"
    assert closed.closed_at
    assert store.metrics()["open_capa"] == 0


def test_capa_rejects_skipped_or_unsubstantiated_transitions(tmp_path) -> None:
    store = TraceabilityStore(tmp_path / "trace.db")
    case, _ = store.create_capa(
        alert_code="QUALITY",
        severity="warning",
        title="품질 기준 초과",
        description="확인 필요",
        actor="quality",
    )

    with pytest.raises(ValueError, match="invalid CAPA transition"):
        store.transition_capa(case.case_id, "IN_PROGRESS", actor="quality")

    store.transition_capa(case.case_id, "ACKNOWLEDGED", actor="quality")
    store.transition_capa(case.case_id, "IN_PROGRESS", actor="quality")
    with pytest.raises(ValueError, match="root cause and corrective action"):
        store.transition_capa(case.case_id, "VERIFIED", actor="quality")


def test_demo_seed_is_idempotent_and_searchable(tmp_path) -> None:
    store = TraceabilityStore(tmp_path / "trace.db")

    assert store.seed_demo() == 48
    assert store.seed_demo() == 0
    assert len(store.products(lot_id="LOT-A")) == 24
    assert len(store.products(query="TRACE-0003")) == 10
    assert store.metrics()["products"] == 48
    assert store.metrics()["open_capa"] == 2
    assert {case.status for case in store.capa_cases()} == {"OPEN", "IN_PROGRESS"}


def test_capa_validates_required_fields_and_severity(tmp_path) -> None:
    store = TraceabilityStore(tmp_path / "trace.db")

    with pytest.raises(ValueError, match="required"):
        store.create_capa(
            alert_code="QUALITY",
            severity="warning",
            title="",
            description="조치 필요",
            actor="quality",
        )
    with pytest.raises(ValueError, match="severity"):
        store.create_capa(
            alert_code="QUALITY",
            severity="urgent",
            title="품질 기준 초과",
            description="조치 필요",
            actor="quality",
        )


def test_status_filter_searches_product_event_history(tmp_path) -> None:
    store = TraceabilityStore(tmp_path / "trace.db")
    store.record_inspection(_inspection(product_id="Q-HISTORY"), actor="operator")

    assert store.products()[0].latest_status == "REJECT"
    assert store.products(status="DEFECT")[0].product_id == "Q-HISTORY"
    assert store.products(status="PASS_THROUGH") == []

def test_database_schema_version_is_recorded(tmp_path) -> None:
    store = TraceabilityStore(tmp_path / "trace.db")
    with store._connect() as connection:
        value = connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()[0]

    assert json.loads(value) == 1
