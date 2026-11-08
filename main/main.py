import cv2
from HandAnalyzer import *
from data.AnalysisResult import AnalysisResult

handAnalyzer: HandAnalyzer = HandAnalyzer("utils/hand_landmarker.task", "utils/pose_landmarker_full.task", "utils/yolov8s_tattoo_manual_v3_best.pt")

def paint_hand(path, path_output="output", print_result=False):
    # 1 анализ
    result: AnalysisResult = handAnalyzer.analyze(path, path_output)
    if (print_result):  
        print(result)

    # 2 Связи между точками кисти
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),           # большой палец
        (0, 5), (5, 6), (6, 7), (7, 8),           # указательный
        (5, 9), (9, 10), (10, 11), (11, 12),      # средний
        (9, 13), (13, 14), (14, 15), (15, 16),    # безымянный
        (13, 17), (17, 18), (18, 19), (19, 20),   # мизинец
        (0, 17),                                  # основание ладони
    ]

    img = cv2.imread(path)
    # 3. Загрузка изображения для отрисовки
    if img is None:
        raise ValueError("Не удалось загрузить изображение")


    # 4. Отрисовка кисти
    if result.hand:
        hand = result.hand
        points = [(int(p[0]), int(p[1])) for p in hand.landmarks_px]

        # 4.1. Кости
        for start, end in HAND_CONNECTIONS:
            cv2.line(img, points[start], points[end], (255, 0, 0), 2, cv2.LINE_AA)

        # 4.2. Суставы
        for i, (x, y) in enumerate(points):
            color = (0, 255, 0) if i == 0 else (0, 200, 255)
            cv2.circle(img, (x, y), 5, color, -1, cv2.LINE_AA)

        # 4.3. Bbox кисти
        x1, y1, x2, y2 = hand.hand_bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(img, f"{hand.handedness} hand", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # 4.4. Татуировки
        for tb in hand.tattoo_bboxes:
            tx1, ty1, tx2, ty2 = tb
            cv2.rectangle(img, (tx1, ty1), (tx2, ty2), (0, 0, 255), 2)
            cv2.putText(img, "tattoo", (tx1, ty1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        # 4.5. Одежда
        if hand.has_clothing and hand.clothing_bbox:
            cx1, cy1, cx2, cy2 = hand.clothing_bbox
            cv2.rectangle(img, (cx1, cy1), (cx2, cy2), (255, 0, 255), 2)
            cv2.putText(img, "clothing", (cx1, cy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)

# # 5. Отрисовка руки (плечо–локоть–запястье)                           TODO
# if result.arms:
#     arm = result.arms
#     shoulder = (int(arm.shoulder_px[0]), int(arm.shoulder_px[1]))
#     elbow = (int(arm.elbow_px[0]), int(arm.elbow_px[1]))
#     wrist = (int(arm.wrist_px[0]), int(arm.wrist_px[1]))

#     cv2.line(img, shoulder, elbow, (0, 255, 0), 2, cv2.LINE_AA)
#     cv2.line(img, elbow, wrist, (0, 255, 0), 2, cv2.LINE_AA)
#     cv2.circle(img, shoulder, 6, (255, 0, 0), -1)
#     cv2.circle(img, elbow, 6, (0, 255, 255), -1)
#     cv2.circle(img, wrist, 6, (0, 0, 255), -1)

#     # Bbox предплечья
#     fx1, fy1, fx2, fy2 = arm.forearm_bbox
#     cv2.rectangle(img, (fx1, fy1), (fx2, fy2), (255, 0, 255), 2)
#     cv2.putText(img, f"{arm.side} forearm", (fx1, fy1 - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

    # 6. сохранение
    path_output = path_output + "/result.jpg"
    cv2.imwrite(path_output, img)


# Тест на
paint_hand("hand_with_cloth.jpg", "output/hand_with_cloth")
paint_hand("hand_withow_cloth.jpg", "output/hand_withow_cloth")