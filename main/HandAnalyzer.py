import os
import cv2
import numpy as np
from detector.HandDetector import HandDetector
from detector.PoseDetector import PoseDetector
from detector.TattooDetector import TattooDetector
from detector.ClothingDetector import ClothingDetector
from data.AnalysisResult import AnalysisResult
from data.HandResult import HandResult
from data.ArmResult import ArmResult
from typing import List

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

        # 1. Кисти и остальные части рук
        hand_res, hand_mask = self.getHandResultAndMask(image_rgb, h, w)

        if (hand_res == None):
            return AnalysisResult(has_hand=False, hand=None, image_width=w, image_height=h)

        side = hand_res.handedness
        exclude_tattoo_mask = self.createAndGetExcludeTattooMask(image, hand_mask, hand_res, h, w, side, output_dir, base_name)
        exclude_cloth_mask = self.createAndGetExcludeClothMask(image_rgb, hand_mask, hand_res, side, exclude_tattoo_mask, output_dir, base_name)

        return AnalysisResult(has_hand=True, hand=hand_res, arms=[], image_width=w, image_height=h) # HACK нужно переписать метод getArmResult


    def getHandResultAndMask(self, image_rgb, h, w) -> list:
        # 1. Кисти
        hands_data = self.hand_detector.detect(image_rgb)
        
        if hands_data == ("", [], []):
            return [None, None]
        
        # 3. Для каждой кисти строим ROI и ищем тату/одежду
        side = hands_data[0]
        lms_norm = hands_data[1]
        lms_px = hands_data[2]
        
        # Hull и bbox
        pts = np.array([[int(p[0]), int(p[1])] for p in lms_px], dtype=np.int32)
        hull = cv2.convexHull(pts)
        x, y, bw, bh = cv2.boundingRect(hull)
        hand_bbox = [int(x), int(y), int(x + bw), int(y + bh)]

        hand_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(hand_mask, hull, 255)
        
        return [HandResult(
            handedness=side,
            landmarks_px=lms_px,
            landmarks_normalized=lms_norm,
            hand_bbox=hand_bbox,
            hand_hull=hull.reshape(-1, 2).tolist(),
        ), hand_mask]


    def getArmResult(self, image_rgb, h, w, output_dir, base_name): # FIXME переписать под новую версию pose_detector
        # 1 Поза (для предплечий)
        pose_data = self.pose_detector.detect(image_rgb)
        
        if pose_data == ("", [], [], []):
            return None
        
        # Индексы точек позы
        LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
        LEFT_ELBOW, RIGHT_ELBOW = 13, 14
        LEFT_WRIST, RIGHT_WRIST = 15, 16
        
        arms: List[ArmResult] = []
        
        _, pose_norm, pose_px, vis = pose_data
        
        for side, s_idx, e_idx, w_idx in [
            ("Left",  LEFT_SHOULDER,  LEFT_ELBOW,  LEFT_WRIST),
            ("Right", RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)]:
        
            shoulder = pose_px[s_idx]   # [x, y, z]
            elbow    = pose_px[e_idx]
            wrist    = pose_px[w_idx]
        
            # Bbox предплечья: прямоугольник вокруг отрезка локоть–запястье + отступ
            pts = np.array([
                [int(elbow[0]), int(elbow[1])],
                [int(wrist[0]), int(wrist[1])],
            ], dtype=np.int32)
            x, y, bw, bh = cv2.boundingRect(pts)
            pad = 40
            forearm_bbox = [
                max(0, x - pad),
                max(0, y - pad),
                min(w, x + bw + pad),
                min(h, y + bh + pad),
            ]
        
            # Маска предплечья (толстая линия локоть–запястье)
            forearm_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.line(
                forearm_mask,
                (int(elbow[0]), int(elbow[1])),
                (int(wrist[0]), int(wrist[1])),
                255,
                thickness=80
            )
            mask_path = os.path.join(output_dir, f"{base_name}_{side}_forearm.png")
            cv2.imwrite(mask_path, forearm_mask)
        
            arms.append(ArmResult(
                side,
                shoulder,
                elbow,
                wrist,
                vis[s_idx],
                elbow_visibility=vis[e_idx],
                wrist_visibility=vis[w_idx],
                forearm_bbox=forearm_bbox,
                forearm_mask_path=mask_path,
            ))

        return arms
        

    def createAndGetExcludeTattooMask(self, image, hand_mask, hand_res, h, w, side, output_dir, base_name):
        # Татуировки (только в ROI руки)
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
        
        return exclude_mask


    def createAndGetExcludeClothMask(self, image_rgb, hand_mask, hand_res, side, exclude_tattoo_mask, output_dir, base_name):
        # Одежда (только в ROI руки)
        clothing_mask = self.clothing_detector.detect(image_rgb, roi_mask=hand_mask)
        overlap_cloth = cv2.bitwise_and(hand_mask, clothing_mask) 
        exclude_mask = None
        if cv2.countNonZero(overlap_cloth) > 100:
            ys, xs = np.where(overlap_cloth > 0)
            hand_res.has_clothing = True
            hand_res.clothing_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            exclude_mask = cv2.bitwise_or(exclude_tattoo_mask, overlap_cloth)
        
        exclude_path = os.path.join(output_dir, f"{base_name}_{side}_exclude.png")
        if (exclude_mask is not None):
            cv2.imwrite(exclude_path, exclude_mask)
            hand_res.exclude_mask_path = exclude_path

        return exclude_mask


    def close(self):
        self.hand_detector.close()
        self.pose_detector.close()