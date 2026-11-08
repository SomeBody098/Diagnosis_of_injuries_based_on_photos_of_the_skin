from .Detector import Detector
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Tuple

class PoseDetector(Detector): # FIXME модель вычисляет руки и другие конечности только если сначало обнаружит ТЕЛО. Нужно найти другую модель, независищую от этого требования. Holistic - под сомнением
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    LEFT_ELBOW, RIGHT_ELBOW = 13, 14
    LEFT_WRIST, RIGHT_WRIST = 15, 16

    def __init__(self, model_path: str):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(self, image_rgb: np.ndarray, **kwargs) -> Tuple[str, list, list, list]:
        """
        Возвращает список: [(side, landmarks_norm, landmarks_px), ...]
        landmarks_norm / landmarks_px — все 33 точки позы.
        Если поза не найдена — пустой список.
        """
        h, w = image_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self.landmarker.detect(mp_image)

        print("PoseDetector:", result)

        if not result.pose_landmarks:
            return ("", [], [], [])

        pose = result.pose_landmarks[0]
        lms_norm = [[lm.x, lm.y, lm.z] for lm in pose]
        lms_px = [[lm.x * w, lm.y * h, lm.z] for lm in pose]
        lms_vis = [lm.visibility for lm in pose]

        # "side" для позы не имеет смысла — это один человек.
        # Возвращаем один кортеж, а HandAnalyzer сам разберёт, где левая/правая рука.
        return ("Person", lms_norm, lms_px, lms_vis)

    def close(self):
        self.landmarker.close()