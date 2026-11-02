from dataclasses import dataclass
from src.data import PoincareSection, PropagationDirection


@dataclass(frozen=True)
class PoincareMap:
    section: PoincareSection
    max_intersections: int
    direction: PropagationDirection | None = None

    def __post_init__(self):
        if self.max_intersections <= 0:
            raise ValueError("max_intersections must be positive")
