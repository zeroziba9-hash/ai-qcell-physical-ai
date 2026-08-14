"""SQLite-backed product genealogy, CAPA workflow, and audit trail."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import csv
import io
import json
from pathlib import Path
import sqlite3
from typing import Iterable, Mapping
from uuid import uuid4


CAPA_TRANSITIONS: dict[str, frozenset[str]] = {
    "OPEN": frozenset({"ACKNOWLEDGED"}),
    "ACKNOWLEDGED": frozenset({"IN_PROGRESS"}),
    "IN_PROGRESS": frozenset({"VERIFIED"}),
    "VERIFIED": frozenset({"CLOSED"}),
    "CLOSED": frozenset(),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ProductTrace:
    product_id: str
    lot_id: str
    first_seen_at: str
    updated_at: str
    latest_stage: str
    latest_status: str
    model_version: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class TraceEvent:
    event_id: str
    product_id: str
    occurred_at: str
    stage: str
    event_type: str
    status: str
    model_version: str
    actor: str
    payload: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CapaCase:
    case_id: str
    product_id: str
    lot_id: str
    alert_code: str
    severity: str
    status: str
    title: str
    description: str
    owner: str
    root_cause: str
    corrective_action: str
    due_at: str
    created_at: str
    updated_at: str
    closed_at: str
    dedupe_key: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class TraceabilityStore:
    schema_version = 1

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS products (
                    product_id TEXT PRIMARY KEY,
                    lot_id TEXT NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    latest_stage TEXT NOT NULL,
                    latest_status TEXT NOT NULL,
                    model_version TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS trace_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    product_id TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY(product_id) REFERENCES products(product_id)
                );
                CREATE INDEX IF NOT EXISTS idx_trace_product_time
                    ON trace_events(product_id, occurred_at, sequence);
                CREATE INDEX IF NOT EXISTS idx_products_lot_status
                    ON products(lot_id, latest_status);
                CREATE TABLE IF NOT EXISTS capa_cases (
                    case_id TEXT PRIMARY KEY,
                    product_id TEXT NOT NULL,
                    lot_id TEXT NOT NULL,
                    alert_code TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    root_cause TEXT NOT NULL,
                    corrective_action TEXT NOT NULL,
                    due_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    closed_at TEXT NOT NULL,
                    dedupe_key TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_capa_dedupe
                    ON capa_cases(dedupe_key);
                CREATE INDEX IF NOT EXISTS idx_capa_status_due
                    ON capa_cases(status, due_at);
                CREATE TABLE IF NOT EXISTS audit_log (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    audit_id TEXT NOT NULL UNIQUE,
                    occurred_at TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );
                """
            )
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
                (str(self.schema_version),),
            )

    @staticmethod
    def _audit(
        connection: sqlite3.Connection,
        *,
        actor: str,
        action: str,
        entity_type: str,
        entity_id: str,
        details: Mapping[str, object] | None = None,
        occurred_at: str | None = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO audit_log(
                audit_id, occurred_at, actor, action, entity_type, entity_id, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uuid4().hex[:16],
                occurred_at or _utc_now(),
                actor,
                action,
                entity_type,
                entity_id,
                json.dumps(dict(details or {}), ensure_ascii=False, sort_keys=True),
            ),
        )

    @staticmethod
    def _upsert_product(
        connection: sqlite3.Connection,
        *,
        product_id: str,
        lot_id: str,
        occurred_at: str,
        stage: str,
        status: str,
        model_version: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO products(
                product_id, lot_id, first_seen_at, updated_at,
                latest_stage, latest_status, model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(product_id) DO UPDATE SET
                lot_id = excluded.lot_id,
                updated_at = excluded.updated_at,
                latest_stage = excluded.latest_stage,
                latest_status = excluded.latest_status,
                model_version = CASE
                    WHEN excluded.model_version = '' THEN products.model_version
                    ELSE excluded.model_version
                END
            """,
            (
                product_id,
                lot_id,
                occurred_at,
                occurred_at,
                stage,
                status,
                model_version,
            ),
        )

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection,
        *,
        product_id: str,
        lot_id: str,
        stage: str,
        event_type: str,
        status: str,
        model_version: str,
        actor: str,
        payload: Mapping[str, object] | None,
        occurred_at: str,
    ) -> TraceEvent:
        event_id = uuid4().hex[:16]
        TraceabilityStore._upsert_product(
            connection,
            product_id=product_id,
            lot_id=lot_id,
            occurred_at=occurred_at,
            stage=stage,
            status=status,
            model_version=model_version,
        )
        normalized_payload = dict(payload or {})
        connection.execute(
            """
            INSERT INTO trace_events(
                event_id, product_id, occurred_at, stage, event_type,
                status, model_version, actor, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                product_id,
                occurred_at,
                stage,
                event_type,
                status,
                model_version,
                actor,
                json.dumps(normalized_payload, ensure_ascii=False, sort_keys=True),
            ),
        )
        TraceabilityStore._audit(
            connection,
            actor=actor,
            action="TRACE_EVENT_RECORDED",
            entity_type="product",
            entity_id=product_id,
            details={"event_id": event_id, "stage": stage, "status": status},
            occurred_at=occurred_at,
        )
        return TraceEvent(
            event_id=event_id,
            product_id=product_id,
            occurred_at=occurred_at,
            stage=stage,
            event_type=event_type,
            status=status,
            model_version=model_version,
            actor=actor,
            payload=normalized_payload,
        )

    def record_event(
        self,
        *,
        product_id: str,
        lot_id: str,
        stage: str,
        event_type: str,
        status: str,
        model_version: str = "",
        actor: str = "system",
        payload: Mapping[str, object] | None = None,
        occurred_at: str | None = None,
    ) -> TraceEvent:
        if not product_id.strip() or not lot_id.strip():
            raise ValueError("product_id and lot_id are required")
        timestamp = occurred_at or _utc_now()
        with self._connect() as connection:
            return self._insert_event(
                connection,
                product_id=product_id.strip(),
                lot_id=lot_id.strip(),
                stage=stage.strip().upper(),
                event_type=event_type.strip().upper(),
                status=status.strip().upper(),
                model_version=model_version.strip(),
                actor=actor.strip() or "system",
                payload=payload,
                occurred_at=timestamp,
            )

    def record_inspection(
        self,
        event: Mapping[str, object],
        *,
        lot_id: str = "LIVE-SESSION",
        model_version: str = "deep-patchcore-production",
        actor: str = "system",
    ) -> tuple[TraceEvent, TraceEvent]:
        product_id = str(event.get("product_id", "")).strip()
        if not product_id:
            raise ValueError("inspection event must include product_id")
        timestamp = str(event.get("timestamp", "")) or _utc_now()
        result = str(event.get("result", "")).upper()
        action = str(event.get("action", "")).upper()
        if result not in {"PASS", "DEFECT"}:
            raise ValueError("inspection result must be PASS or DEFECT")
        if action not in {"PASS_THROUGH", "REJECT"}:
            raise ValueError("inspection action must be PASS_THROUGH or REJECT")
        event_lot = str(event.get("lot_id", lot_id))
        with self._connect() as connection:
            inspection = self._insert_event(
                connection,
                product_id=product_id,
                lot_id=event_lot,
                stage="VISION",
                event_type="AI_INSPECTION",
                status=result,
                model_version=model_version,
                actor=actor,
                payload={
                    "defect_type": event.get("defect_type", "none"),
                    "confidence": event.get("confidence", 0.0),
                    "latency_ms": event.get("latency_ms", 0.0),
                },
                occurred_at=timestamp,
            )
            sorting = self._insert_event(
                connection,
                product_id=product_id,
                lot_id=event_lot,
                stage="SORTING",
                event_type="ROS2_ACTION",
                status=action,
                model_version=model_version,
                actor=actor,
                payload={"source_result": result},
                occurred_at=timestamp,
            )
        return inspection, sorting

    @staticmethod
    def _product_from_row(row: sqlite3.Row) -> ProductTrace:
        return ProductTrace(**dict(row))

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> TraceEvent:
        payload = dict(row)
        payload["payload"] = json.loads(payload.pop("payload_json"))
        payload.pop("sequence", None)
        return TraceEvent(**payload)

    @staticmethod
    def _capa_from_row(row: sqlite3.Row) -> CapaCase:
        return CapaCase(**dict(row))

    def products(
        self,
        *,
        query: str = "",
        lot_id: str = "",
        status: str = "",
        limit: int = 500,
    ) -> list[ProductTrace]:
        clauses: list[str] = []
        parameters: list[object] = []
        if query.strip():
            clauses.append("product_id LIKE ?")
            parameters.append(f"%{query.strip()}%")
        if lot_id.strip():
            clauses.append("lot_id = ?")
            parameters.append(lot_id.strip())
        if status.strip():
            clauses.append(
                "EXISTS (SELECT 1 FROM trace_events event "
                "WHERE event.product_id = products.product_id AND event.status = ?)"
            )
            parameters.append(status.strip().upper())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(max(1, min(limit, 5000)))
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM products {where} ORDER BY updated_at DESC, product_id DESC LIMIT ?",
                parameters,
            ).fetchall()
        return [self._product_from_row(row) for row in rows]

    def lots(self) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT lot_id FROM products ORDER BY lot_id"
            ).fetchall()
        return [str(row["lot_id"]) for row in rows]

    def timeline(self, product_id: str) -> list[TraceEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM trace_events
                WHERE product_id = ?
                ORDER BY occurred_at, sequence
                """,
                (product_id,),
            ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def create_capa(
        self,
        *,
        alert_code: str,
        severity: str,
        title: str,
        description: str,
        actor: str,
        lot_id: str = "",
        product_id: str = "",
        owner: str = "",
        due_at: str = "",
        dedupe_key: str = "",
    ) -> tuple[CapaCase, bool]:
        if not alert_code.strip() or not title.strip() or not description.strip():
            raise ValueError("alert code, title, and description are required")
        normalized_severity = severity.strip().upper()
        if normalized_severity not in {"INFO", "WARNING", "CRITICAL"}:
            raise ValueError("severity must be INFO, WARNING, or CRITICAL")
        normalized_key = dedupe_key.strip() or ":".join(
            [alert_code.strip().upper(), lot_id.strip(), product_id.strip()]
        )
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM capa_cases WHERE dedupe_key = ?",
                (normalized_key,),
            ).fetchone()
            if existing is not None:
                return self._capa_from_row(existing), False
            now = _utc_now()
            case = CapaCase(
                case_id=f"CAPA-{uuid4().hex[:10].upper()}",
                product_id=product_id.strip(),
                lot_id=lot_id.strip(),
                alert_code=alert_code.strip().upper(),
                severity=normalized_severity,
                status="OPEN",
                title=title.strip(),
                description=description.strip(),
                owner=owner.strip(),
                root_cause="",
                corrective_action="",
                due_at=due_at.strip(),
                created_at=now,
                updated_at=now,
                closed_at="",
                dedupe_key=normalized_key,
            )
            connection.execute(
                """
                INSERT INTO capa_cases(
                    case_id, product_id, lot_id, alert_code, severity, status,
                    title, description, owner, root_cause, corrective_action,
                    due_at, created_at, updated_at, closed_at, dedupe_key
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(case.to_dict().values()),
            )
            self._audit(
                connection,
                actor=actor,
                action="CAPA_CREATED",
                entity_type="capa",
                entity_id=case.case_id,
                details={"alert_code": case.alert_code, "severity": case.severity},
                occurred_at=now,
            )
        return case, True

    def capa_cases(self, *, status: str = "") -> list[CapaCase]:
        parameters: tuple[object, ...] = ()
        where = ""
        if status.strip():
            where = "WHERE status = ?"
            parameters = (status.strip().upper(),)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM capa_cases {where} ORDER BY updated_at DESC, case_id DESC",
                parameters,
            ).fetchall()
        return [self._capa_from_row(row) for row in rows]

    def get_capa(self, case_id: str) -> CapaCase:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM capa_cases WHERE case_id = ?", (case_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"CAPA case not found: {case_id}")
        return self._capa_from_row(row)

    def transition_capa(
        self,
        case_id: str,
        new_status: str,
        *,
        actor: str,
        owner: str | None = None,
        root_cause: str | None = None,
        corrective_action: str | None = None,
        note: str = "",
    ) -> CapaCase:
        target = new_status.strip().upper()
        if target not in CAPA_TRANSITIONS:
            raise ValueError(f"unsupported CAPA status: {target}")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM capa_cases WHERE case_id = ?", (case_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"CAPA case not found: {case_id}")
            current = self._capa_from_row(row)
            if target not in CAPA_TRANSITIONS[current.status]:
                raise ValueError(f"invalid CAPA transition: {current.status} -> {target}")
            next_owner = current.owner if owner is None else owner.strip()
            next_root = current.root_cause if root_cause is None else root_cause.strip()
            next_action = (
                current.corrective_action
                if corrective_action is None
                else corrective_action.strip()
            )
            if target in {"VERIFIED", "CLOSED"} and (not next_root or not next_action):
                raise ValueError(
                    "root cause and corrective action are required before verification"
                )
            now = _utc_now()
            closed_at = now if target == "CLOSED" else current.closed_at
            connection.execute(
                """
                UPDATE capa_cases
                SET status = ?, owner = ?, root_cause = ?, corrective_action = ?,
                    updated_at = ?, closed_at = ?
                WHERE case_id = ?
                """,
                (
                    target,
                    next_owner,
                    next_root,
                    next_action,
                    now,
                    closed_at,
                    case_id,
                ),
            )
            self._audit(
                connection,
                actor=actor,
                action="CAPA_TRANSITIONED",
                entity_type="capa",
                entity_id=case_id,
                details={
                    "from": current.status,
                    "to": target,
                    "owner": next_owner,
                    "note": note.strip(),
                },
                occurred_at=now,
            )
            updated = connection.execute(
                "SELECT * FROM capa_cases WHERE case_id = ?", (case_id,)
            ).fetchone()
        return self._capa_from_row(updated)

    def audit_entries(self, *, limit: int = 300) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT audit_id, occurred_at, actor, action, entity_type,
                       entity_id, details_json
                FROM audit_log ORDER BY sequence DESC LIMIT ?
                """,
                (max(1, min(limit, 5000)),),
            ).fetchall()
        return [
            {
                **{key: row[key] for key in row.keys() if key != "details_json"},
                "details": json.loads(row["details_json"]),
            }
            for row in rows
        ]

    def metrics(self) -> dict[str, int]:
        with self._connect() as connection:
            product_count = int(
                connection.execute("SELECT COUNT(*) FROM products").fetchone()[0]
            )
            rejected = int(
                connection.execute(
                    "SELECT COUNT(DISTINCT product_id) FROM trace_events WHERE status = 'REJECT'"
                ).fetchone()[0]
            )
            open_capa = int(
                connection.execute(
                    "SELECT COUNT(*) FROM capa_cases WHERE status != 'CLOSED'"
                ).fetchone()[0]
            )
            audit_count = int(
                connection.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
            )
        return {
            "products": product_count,
            "rejected": rejected,
            "open_capa": open_capa,
            "audit_entries": audit_count,
        }

    def seed_demo(self) -> int:
        if self.metrics()["products"]:
            return 0
        started = datetime(2026, 8, 14, 8, 0, tzinfo=timezone.utc)
        rejected_ids: list[str] = []
        for index in range(1, 49):
            is_defect = index % 11 == 0 or index in {34, 35, 36}
            product_id = f"TRACE-{index:05d}"
            lot_id = "LOT-A" if index <= 24 else "LOT-B"
            event = {
                "timestamp": (started + timedelta(seconds=index * 15)).isoformat(
                    timespec="seconds"
                ),
                "product_id": product_id,
                "lot_id": lot_id,
                "result": "DEFECT" if is_defect else "PASS",
                "defect_type": "seal_damage" if is_defect else "none",
                "confidence": 0.91 if is_defect else 0.982,
                "latency_ms": 46.0 if index > 32 else 24.0,
                "action": "REJECT" if is_defect else "PASS_THROUGH",
            }
            self.record_inspection(
                event,
                model_version="deep-patchcore-v3",
                actor="demo-seeder",
            )
            if is_defect:
                rejected_ids.append(product_id)
                self.record_event(
                    product_id=product_id,
                    lot_id=lot_id,
                    stage="REVIEW",
                    event_type="HUMAN_REVIEW",
                    status="PENDING",
                    model_version="deep-patchcore-v3",
                    actor="demo-seeder",
                    payload={"queue": "quality-review"},
                    occurred_at=event["timestamp"],
                )
        self.create_capa(
            alert_code="SPC_OUT_OF_CONTROL",
            severity="CRITICAL",
            title="LOT-B 공정 관리한계 이탈",
            description="후반 교대 구간의 seal damage 불량률이 3σ 상한을 초과했습니다.",
            actor="demo-seeder",
            lot_id="LOT-B",
            owner="quality",
            dedupe_key="demo:LOT-B:SPC_OUT_OF_CONTROL",
        )
        latency_case, _ = self.create_capa(
            alert_code="LATENCY_SLA_MISS",
            severity="WARNING",
            title="Edge 추론 지연 SLA 초과",
            description="교대 후반 p95 추론 지연이 운영 기준을 초과했습니다.",
            actor="demo-seeder",
            product_id=rejected_ids[-1],
            lot_id="LOT-B",
            owner="operator",
            dedupe_key="demo:LOT-B:LATENCY_SLA_MISS",
        )
        self.transition_capa(
            latency_case.case_id,
            "ACKNOWLEDGED",
            actor="quality",
            owner="operator",
            note="Edge node runtime 확인 요청",
        )
        self.transition_capa(
            latency_case.case_id,
            "IN_PROGRESS",
            actor="operator",
            owner="operator",
            note="TensorRT 프로파일 재측정 중",
        )
        return 48


def trace_events_to_csv(events: Iterable[TraceEvent]) -> str:
    rows = [event.to_dict() for event in events]
    if not rows:
        return ""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({**row, "payload": json.dumps(row["payload"], ensure_ascii=False)})
    return buffer.getvalue()
