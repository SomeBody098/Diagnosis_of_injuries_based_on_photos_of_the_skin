from typing import List, Optional
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ArmResult:
    """Информация о руке целиком (плечо–локоть–запястье)"""
    side: str                          # "Left" / "Right"
    shoulder_px: List[float]
    elbow_px: List[float]
    wrist_px: List[float]
    shoulder_visibility: float
    elbow_visibility: float
    wrist_visibility: float
    forearm_bbox: List[int]            # bbox области предплечья
    forearm_mask_path: Optional[str] = None

