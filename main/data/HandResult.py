from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class HandResult:
    handedness: str
    landmarks_px: List[List[float]]
    landmarks_normalized: List[List[float]]
    hand_bbox: List[int]
    hand_hull: List[List[float]]
    has_clothing: bool = False
    clothing_bbox: Optional[List[int]] = None
    has_tattoo: bool = False
    tattoo_bboxes: List[List[int]] = field(default_factory=list)
    tattoo_confidences: List[float] = field(default_factory=list)
    tattoo_mask_paths: List[str] = field(default_factory=list)
    exclude_mask_path: Optional[str] = None