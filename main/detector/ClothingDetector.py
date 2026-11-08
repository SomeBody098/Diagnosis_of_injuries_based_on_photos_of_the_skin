import cv2
import numpy as np
from fashn_human_parser import FashnHumanParser
from .Detector import Detector
from typing import Optional

class ClothingDetector(Detector): # FIXME бывает принимает за одежду просто ткань, например, на которой лежит рука.
    def __init__(self):
        self.parser = FashnHumanParser()

    def detect(self, image_rgb: np.ndarray, roi_mask: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Возвращает маску класса 'top' (3) для всего изображения.
        Если передан roi_mask, вырезает ROI для ускорения.
        """
        if roi_mask is not None and cv2.countNonZero(roi_mask) > 0:
            ys, xs = np.where(roi_mask > 0)
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            crop_rgb = image_rgb[y1:y2, x1:x2]
            seg_crop = self.parser.predict(crop_rgb)
            full_seg = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
            full_seg[y1:y2, x1:x2] = seg_crop
        else:
            full_seg = self.parser.predict(image_rgb)

        return np.where(full_seg == 3, 255, 0).astype(np.uint8)