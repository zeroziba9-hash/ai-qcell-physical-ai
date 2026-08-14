"""Shift-level quality analytics, SPC signals, and report persistence."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import csv
import io
import json
import math
from pathlib import Path
import random
from threading import RLock
from typing import Iterable, Mapping
from uuid import uuid4


@dataclass(frozen=True)
class QualitySummary:
    inspected: int
    passed: int
    defects: int
    first_pass_yield_percent: float
    defect_rate_percent: float
    defect_ppm: int
    average_confidence_percent: float
    average_latency_ms: float
    p95_latency_ms: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class ControlChartPoint:
    subgroup: int
    start_at: str
    end_at: str
    inspected: int
    defects: int
    defect_rate_percent: float
    center_line_percent: float
    lower_control_limit_percent: float
    upper_control_limit_percent: float
    out_of_control: bool

    def to_dict(self) -> dict[str, int | float | str | bool]:
        return asdict(self)


@dataclass(frozen=True)
class QualityAlert:
    code: str
    severity: str
    title: str
    detail: str
    measured: float
    threshold: float

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _materialize(events: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    materialized: list[dict[str, object]] = []
    for index, event in enumerate(events, start=1):
        result = str(event.get("result", "")).upper()
        if result not in {"PASS", "DEFECT"}:
            raise ValueError(f"event {index} result must be PASS or DEFECT")
        confidence = float(event.get("confidence", 0.0))
        latency_ms = float(event.get("latency_ms", 0.0))
        if not 0 <= confidence <= 1:
            raise ValueError(f"event {index} confidence must be between 0 and 1")
        if latency_ms < 0:
            raise ValueError(f"event {index} latency_ms must not be negative")
        materialized.append(
            {
                **dict(event),
                "result": result,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "defect_type": str(event.get("defect_type", "none")),
                "timestamp": str(event.get("timestamp", "")),
            }
        )
    return materialized


def summarize_quality(events: Iterable[Mapping[str, object]]) -> QualitySummary:
    records = _materialize(events)
    inspected = len(records)
    defects = sum(record["result"] == "DEFECT" for record in records)
    passed = inspected - defects
    defect_rate = defects / inspected if inspected else 0.0
    confidences = [float(record["confidence"]) for record in records]
    latencies = [float(record["latency_ms"]) for record in records]
    return QualitySummary(
        inspected=inspected,
        passed=passed,
        defects=defects,
        first_pass_yield_percent=round((1.0 - defect_rate) * 100.0, 2),
        defect_rate_percent=round(defect_rate * 100.0, 2),
        defect_ppm=round(defect_rate * 1_000_000),
        average_confidence_percent=round(
            sum(confidences) / len(confidences) * 100.0 if confidences else 0.0, 2
        ),
        average_latency_ms=round(
            sum(latencies) / len(latencies) if latencies else 0.0, 2
        ),
        p95_latency_ms=round(_percentile(latencies, 0.95), 2),
    )


def build_control_chart(
    events: Iterable[Mapping[str, object]], subgroup_size: int = 20
) -> list[ControlChartPoint]:
    if subgroup_size < 2:
        raise ValueError("subgroup_size must be at least 2")
    records = _materialize(events)
    if not records:
        return []
    center_line = sum(record["result"] == "DEFECT" for record in records) / len(records)
    points: list[ControlChartPoint] = []
    for offset in range(0, len(records), subgroup_size):
        group = records[offset : offset + subgroup_size]
        inspected = len(group)
        defects = sum(record["result"] == "DEFECT" for record in group)
        rate = defects / inspected
        sigma = math.sqrt(center_line * (1.0 - center_line) / inspected)
        lower = max(0.0, center_line - 3.0 * sigma)
        upper = min(1.0, center_line + 3.0 * sigma)
        points.append(
            ControlChartPoint(
                subgroup=len(points) + 1,
                start_at=str(group[0]["timestamp"]),
                end_at=str(group[-1]["timestamp"]),
                inspected=inspected,
                defects=defects,
                defect_rate_percent=round(rate * 100.0, 2),
                center_line_percent=round(center_line * 100.0, 2),
                lower_control_limit_percent=round(lower * 100.0, 2),
                upper_control_limit_percent=round(upper * 100.0, 2),
                out_of_control=rate < lower or rate > upper,
            )
        )
    return points


def build_defect_pareto(
    events: Iterable[Mapping[str, object]],
) -> list[dict[str, int | float | str]]:
    records = _materialize(events)
    counts = Counter(
        str(record["defect_type"])
        for record in records
        if record["result"] == "DEFECT" and record["defect_type"] != "none"
    )
    total = sum(counts.values())
    cumulative = 0
    rows: list[dict[str, int | float | str]] = []
    for defect_type, count in counts.most_common():
        cumulative += count
        rows.append(
            {
                "defect_type": defect_type,
                "count": count,
                "share_percent": round(count / total * 100.0, 2) if total else 0.0,
                "cumulative_percent": round(cumulative / total * 100.0, 2)
                if total
                else 0.0,
            }
        )
    return rows


def detect_quality_alerts(
    summary: QualitySummary,
    control_chart: Iterable[ControlChartPoint],
    *,
    target_defect_rate_percent: float = 5.0,
    latency_sla_ms: float = 50.0,
    confidence_floor_percent: float = 90.0,
) -> list[QualityAlert]:
    if target_defect_rate_percent < 0 or latency_sla_ms < 0:
        raise ValueError("quality thresholds must not be negative")
    alerts: list[QualityAlert] = []
    unstable = [point for point in control_chart if point.out_of_control]
    if unstable:
        latest = unstable[-1]
        above_limit = latest.defect_rate_percent > latest.upper_control_limit_percent
        direction = "상한" if above_limit else "하한"
        limit = latest.upper_control_limit_percent if above_limit else latest.lower_control_limit_percent
        alerts.append(
            QualityAlert(
                code="SPC_OUT_OF_CONTROL",
                severity="critical",
                title="공정 관리한계 이탈",
                detail=f"소그룹 {latest.subgroup}의 불량률이 3σ {direction} 관리한계를 벗어났습니다.",
                measured=latest.defect_rate_percent,
                threshold=limit,
            )
        )
    if summary.defect_rate_percent > target_defect_rate_percent:
        alerts.append(
            QualityAlert(
                code="DEFECT_RATE_HIGH",
                severity="warning",
                title="목표 불량률 초과",
                detail="교대 누적 불량률이 운영 목표를 초과했습니다.",
                measured=summary.defect_rate_percent,
                threshold=target_defect_rate_percent,
            )
        )
    if summary.p95_latency_ms > latency_sla_ms:
        alerts.append(
            QualityAlert(
                code="LATENCY_SLA_MISS",
                severity="warning",
                title="추론 지연 SLA 초과",
                detail="p95 추론 시간이 설정된 운영 한계를 초과했습니다.",
                measured=summary.p95_latency_ms,
                threshold=latency_sla_ms,
            )
        )
    if summary.inspected and summary.average_confidence_percent < confidence_floor_percent:
        alerts.append(
            QualityAlert(
                code="CONFIDENCE_LOW",
                severity="info",
                title="평균 판정 신뢰도 저하",
                detail="조명·제품 위치 또는 모델 드리프트 점검이 필요합니다.",
                measured=summary.average_confidence_percent,
                threshold=confidence_floor_percent,
            )
        )
    return alerts


def build_quality_report(
    events: Iterable[Mapping[str, object]],
    *,
    subgroup_size: int = 20,
    target_defect_rate_percent: float = 5.0,
    latency_sla_ms: float = 50.0,
    confidence_floor_percent: float = 90.0,
    source: str = "quality-console",
    generated_at: str | None = None,
) -> dict[str, object]:
    records = _materialize(events)
    summary = summarize_quality(records)
    control_chart = build_control_chart(records, subgroup_size=subgroup_size)
    alerts = detect_quality_alerts(
        summary,
        control_chart,
        target_defect_rate_percent=target_defect_rate_percent,
        latency_sla_ms=latency_sla_ms,
        confidence_floor_percent=confidence_floor_percent,
    )
    return {
        "schema_version": 1,
        "generated_at": generated_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "parameters": {
            "subgroup_size": subgroup_size,
            "target_defect_rate_percent": target_defect_rate_percent,
            "latency_sla_ms": latency_sla_ms,
            "confidence_floor_percent": confidence_floor_percent,
        },
        "summary": summary.to_dict(),
        "control_chart": [point.to_dict() for point in control_chart],
        "defect_pareto": build_defect_pareto(records),
        "alerts": [alert.to_dict() for alert in alerts],
    }


def generate_demo_shift(
    count: int = 240,
    *,
    baseline_defect_rate: float = 0.03,
    drift_defect_rate: float = 0.24,
    drift_start_ratio: float = 0.72,
    seed: int = 23,
    start_at: datetime | None = None,
) -> list[dict[str, object]]:
    if count < 1:
        raise ValueError("count must be positive")
    for name, value in (
        ("baseline_defect_rate", baseline_defect_rate),
        ("drift_defect_rate", drift_defect_rate),
        ("drift_start_ratio", drift_start_ratio),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    rng = random.Random(seed)
    started = start_at or datetime(2026, 8, 14, 8, 0, tzinfo=timezone.utc)
    drift_index = round(count * drift_start_ratio)
    defect_types = ("scratch", "contamination", "seal_damage", "deformation")
    weights = (0.42, 0.27, 0.19, 0.12)
    records: list[dict[str, object]] = []
    for index in range(count):
        drifted = index >= drift_index
        defect_rate = drift_defect_rate if drifted else baseline_defect_rate
        is_defect = rng.random() < defect_rate
        defect_type = (
            rng.choices(defect_types, weights=weights, k=1)[0] if is_defect else "none"
        )
        confidence = rng.uniform(0.86, 0.965) if drifted else rng.uniform(0.92, 0.995)
        latency = max(5.0, rng.gauss(38.0 if drifted else 24.0, 4.2))
        captured_at = started + timedelta(seconds=index * 12)
        records.append(
            {
                "timestamp": captured_at.isoformat(timespec="seconds"),
                "product_id": f"SHIFT-{index + 1:05d}",
                "result": "DEFECT" if is_defect else "PASS",
                "defect_type": defect_type,
                "confidence": round(confidence, 3),
                "latency_ms": round(latency, 1),
                "action": "REJECT" if is_defect else "PASS_THROUGH",
                "lot_id": f"LOT-{chr(65 + min(index // 80, 25))}",
                "process_phase": "drift" if drifted else "baseline",
            }
        )
    return records


def quality_events_to_csv(events: Iterable[Mapping[str, object]]) -> str:
    records = _materialize(events)
    if not records:
        return ""
    preferred = [
        "timestamp",
        "product_id",
        "lot_id",
        "result",
        "defect_type",
        "confidence",
        "latency_ms",
        "action",
        "process_phase",
    ]
    extras = sorted({key for record in records for key in record} - set(preferred))
    fields = [field for field in preferred if any(field in record for record in records)] + extras
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue()


def quality_report_to_json(report: Mapping[str, object]) -> str:
    return json.dumps(dict(report), ensure_ascii=False, indent=2)


class QualityReportStore:
    schema_version = 1

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.manifest_path = self.root / "reports.json"
        self._lock = RLock()

    def _read(self) -> list[dict[str, object]]:
        if not self.manifest_path.exists():
            return []
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != self.schema_version:
            raise ValueError("unsupported quality report schema")
        return [dict(snapshot) for snapshot in payload.get("reports", [])]

    def _write(self, snapshots: list[dict[str, object]]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.schema_version,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "reports": snapshots,
        }
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.manifest_path)

    def list(self) -> list[dict[str, object]]:
        return sorted(
            self._read(),
            key=lambda snapshot: str(snapshot.get("saved_at", "")),
            reverse=True,
        )

    def save(
        self,
        report: Mapping[str, object],
        *,
        name: str = "",
        saved_at: str | None = None,
    ) -> dict[str, object]:
        snapshot = {
            "snapshot_id": uuid4().hex[:12],
            "name": name.strip() or "Shift quality report",
            "saved_at": saved_at
            or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "report": dict(report),
        }
        with self._lock:
            snapshots = self._read()
            snapshots.append(snapshot)
            self._write(snapshots)
        return snapshot
