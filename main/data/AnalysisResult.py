from dataclasses import dataclass, field
from typing import List, Optional
from .HandResult import HandResult
from .ArmResult import ArmResult

@dataclass
class AnalysisResult:
    has_hand: bool
    hand: Optional[HandResult]
    arms: List[ArmResult] = field(default_factory=list) 
    image_width: int = 0
    image_height: int = 0