from .Detector import Detector
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Optional

class PoseDetector(Detector):
    def __init__(self, model_path: str):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(self, image_rgb: np.ndarray, **kwargs) -> Optional[list]:
        """Возвращает 33 точки позы (нормализованные) или None."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self.landmarker.detect(mp_image)
        if result.pose_landmarks:
            return result.pose_landmarks[0]
        return None

    def close(self):
        self.landmarker.close()