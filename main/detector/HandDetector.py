from .Detector import Detector
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Tuple

class HandDetector(Detector):
    def __init__(self, model_path: str):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, image_rgb: np.ndarray, **kwargs) -> Tuple[str, list, list]:
        """
        Возвращает список: [(handedness, landmarks_norm, landmarks_px), ...]
        landmarks_norm: [[x,y,z], ...] нормализованные
        landmarks_px: [[x,y,z], ...] в пикселях
        """
        h, w = image_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self.landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return ("", [], [])

        lms = result.hand_landmarks[0]
        side = result.handedness[0][0].category_name if result.handedness else "Hand"

        lms_norm = [[lm.x, lm.y, lm.z] for lm in lms]
        lms_px = [[lm.x * w, lm.y * h, lm.z] for lm in lms]
        return (side, lms_norm, lms_px)

    def close(self):
        self.landmarker.close()