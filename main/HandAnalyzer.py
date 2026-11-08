import os
import cv2
import numpy as np
from detector.HandDetector import HandDetector
from detector.PoseDetector import PoseDetector
from detector.TattooDetector import TattooDetector
from detector.ClothingDetector import ClothingDetector
from data.AnalysisResult import AnalysisResult
from data.HandResult import HandResult


class HandAnalyzer:
    def __init__(self, hand_model: str, pose_model: str, tattoo_model: str):
        self.hand_detector = HandDetector(hand_model)
        self.pose_detector = PoseDetector(pose_model)
        self.tattoo_detector = TattooDetector(tattoo_model)
        self.clothing_detector = ClothingDetector()

    def analyze(self, image_path: str, output_dir: str = "output") -> AnalysisResult:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить: {image_path}")

        h, w = image.shape[:2]
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 1. Кисти
        hands_data = self.hand_detector.detect(image_rgb)
        if not hands_data:
            return AnalysisResult(has_hand=False, image_width=w, image_height=h)

        # 2. Поза (для предплечий)
        pose_landmarks = self.pose_detector.detect(image_rgb)

        # 3. Для каждой кисти строим ROI и ищем тату/одежду
        results = []
        for side, lms_norm, lms_px in hands_data:
            # Hull и bbox
            pts = np.array([[int(p[0]), int(p[1])] for p in lms_px], dtype=np.int32)
            hull = cv2.convexHull(pts)
            x, y, bw, bh = cv2.boundingRect(hull)
            hand_bbox = [int(x), int(y), int(x + bw), int(y + bh)]

            hand_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(hand_mask, hull, 255)

            hand_res = HandResult(
                handedness=side,
                landmarks_px=lms_px,
                landmarks_normalized=lms_norm,
                hand_bbox=hand_bbox,
                hand_hull=hull.reshape(-1, 2).tolist(),
            )

            # 4. Татуировки (только в ROI руки)
            tattoo_masks, tattoo_confs = self.tattoo_detector.detect(image, roi_mask=hand_mask)
            exclude_mask = np.zeros((h, w), dtype=np.uint8)
            for i, tm in enumerate(tattoo_masks):
                overlap = cv2.bitwise_and(hand_mask, tm)
                if cv2.countNonZero(overlap) > 50:
                    ys, xs = np.where(overlap > 0)
                    hand_res.has_tattoo = True
                    hand_res.tattoo_bboxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
                    hand_res.tattoo_confidences.append(tattoo_confs[i])
                    mask_path = os.path.join(output_dir, f"{base_name}_{side}_tattoo_{i}.png")
                    cv2.imwrite(mask_path, overlap)
                    hand_res.tattoo_mask_paths.append(mask_path)
                    exclude_mask = cv2.bitwise_or(exclude_mask, overlap)

            # 5. Одежда (только в ROI руки)
            clothing_mask = self.clothing_detector.detect(image_rgb, roi_mask=hand_mask)
            overlap_cloth = cv2.bitwise_and(hand_mask, clothing_mask)
            if cv2.countNonZero(overlap_cloth) > 100:
                ys, xs = np.where(overlap_cloth > 0)
                hand_res.has_clothing = True
                hand_res.clothing_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                exclude_mask = cv2.bitwise_or(exclude_mask, overlap_cloth)

            exclude_path = os.path.join(output_dir, f"{base_name}_{side}_exclude.png")
            cv2.imwrite(exclude_path, exclude_mask)
            hand_res.exclude_mask_path = exclude_path

            results.append(hand_res)

        return AnalysisResult(has_hand=True, hands=results, image_width=w, image_height=h)

    def close(self):
        self.hand_detector.close()
        self.pose_detector.close()