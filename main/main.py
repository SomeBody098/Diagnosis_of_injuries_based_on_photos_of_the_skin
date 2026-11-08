import cv2
from HandAnalyzer import *
from data.AnalysisResult import AnalysisResult

handAnalyzer: HandAnalyzer = HandAnalyzer("utils/hand_landmarker.task", "utils/pose_landmarker_full.task", "utils/yolov8s_tattoo_manual_v3_best.pt")

result: AnalysisResult = handAnalyzer.analyze("hand_withow_cloth.jpg")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # большой палец
    (0, 5), (5, 6), (6, 7), (7, 8),           # указательный
    (5, 9), (9, 10), (10, 11), (11, 12),      # средний
    (9, 13), (13, 14), (14, 15), (15, 16),    # безымянный
    (13, 17), (17, 18), (18, 19), (19, 20),   # мизинец
    (0, 17),                                  # основание ладони
]

img = cv2.imread("hand_withow_cloth.jpg")
# 4. Отрисовка каждой руки
for hand in result.hands:
    points = [(int(p[0]), int(p[1])) for p in hand.landmarks_px]

    # 4.1. Линии (кости)
    for start, end in HAND_CONNECTIONS:
        cv2.line(img, points[start], points[end], (255, 0, 0), 2, cv2.LINE_AA)

    # 4.2. Точки (суставы)
    for i, (x, y) in enumerate(points):
        color = (0, 255, 0) if i == 0 else (0, 200, 255)  # запястье — зелёное, остальные — жёлтые
        cv2.circle(img, (x, y), 5, color, -1, cv2.LINE_AA)

    # 4.3. Bounding box руки
    x1, y1, x2, y2 = hand.hand_bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # 4.4. Подпись (Left/Right)
    cv2.putText(
        img, hand.handedness, (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA
    )

    # 4.5. Татуировки (если есть)
    for tb in hand.tattoo_bboxes:
        tx1, ty1, tx2, ty2 = tb
        cv2.rectangle(img, (tx1, ty1), (tx2, ty2), (0, 0, 255), 2)
        cv2.putText(
            img, "tattoo", (tx1, ty1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA
        )

    # 4.6. Одежда (если есть)
    if hand.has_clothing and hand.clothing_bbox:
        cx1, cy1, cx2, cy2 = hand.clothing_bbox
        cv2.rectangle(img, (cx1, cy1), (cx2, cy2), (255, 0, 255), 2)
        cv2.putText(
            img, "clothing", (cx1, cy1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA
        )

cv2.imshow("result", img)
cv2.waitKey(0)