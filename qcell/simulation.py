from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import random


DEFECT_TYPES = ("scratch", "crack", "missing_part")


@dataclass(frozen=True)
class InspectionEvent:
    timestamp: str
    product_id: str
    result: str
    defect_type: str
    confidence: float
    latency_ms: float
    action: str

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


class LineSimulator:
    """Generate deterministic-looking inspection events for the UI prototype."""

    def __init__(self, defect_rate: float = 0.12, seed: int | None = None) -> None:
        if not 0 <= defect_rate <= 1:
            raise ValueError("defect_rate must be between 0 and 1")
        self.defect_rate = defect_rate
        self._random = random.Random(seed)
        self._sequence = 0

    def inspect_next(self) -> InspectionEvent:
        self._sequence += 1
        is_defect = self._random.random() < self.defect_rate
        defect_type = self._random.choice(DEFECT_TYPES) if is_defect else "none"
        result = "DEFECT" if is_defect else "PASS"
        action = "REJECT" if is_defect else "PASS_THROUGH"
        confidence = self._random.uniform(0.87, 0.995)
        latency_ms = self._random.uniform(28, 74)

        return InspectionEvent(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            product_id=f"Q-{self._sequence:05d}",
            result=result,
            defect_type=defect_type,
            confidence=round(confidence, 3),
            latency_ms=round(latency_ms, 1),
            action=action,
        )

