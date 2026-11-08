from dataclasses import dataclass, field
from typing import List
from .HandResult import HandResult

@dataclass
class AnalysisResult:
    has_hand: bool
    hands: List[HandResult] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0