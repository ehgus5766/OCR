import cv2
import numpy as np
from paddleocr import PaddleOCR
import time
from ultralytics import YOLO

# YOLO 모델 로드 (기본 COCO → 추후 커스텀 모델로 교체 가능)
model = YOLO("runs/detect/train2/weights/best.pt")

# OCR 설정
ocr = PaddleOCR(use_angle_cls=True, lang='korean')

cap = cv2.VideoCapture(0)
stable = 0

def enhance_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    # 조명 보정
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # 대비/밝기 보정 (약하게)
    gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)

    # Adaptive Threshold (조금 완화)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5
    )
    enhanced = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    enhanced = cv2.resize(enhanced, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
    return enhanced

def four_point_transform(image, pts):
    if len(pts) > 4:
        rect = cv2.minAreaRect(pts.astype(np.float32))
        box = cv2.boxPoints(rect)
        rect = np.array(sorted(box, key=lambda x: (x[1], x[0])), dtype="float32")
    else:
        rect = np.array(pts, dtype="float32")

    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxWidth = int(max(widthA, widthB))
    maxHeight = int(max(heightA, heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = cv2.flip(frame, 1)

    #  YOLO 감지
    results = model(display_frame, verbose=False)
    detected = None

    for r in results:
        boxes = r.boxes.xyxy  # [x1, y1, x2, y2]
        for (x1, y1, x2, y2) in boxes:
            # 감지된 영역 표시
            cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            detected = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ], dtype=np.float32)

            stable += 1
            break  # 첫 번째 감지만 처리

    if detected is None or len(detected) == 0:
        stable = 0

    if stable > 5 and detected is not None:
        print(" 명함 인식 시도 중...")
        warped = four_point_transform(display_frame, detected)
        enhanced = enhance_image(warped)
        enhanced = cv2.resize(enhanced, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)

        result = ocr.ocr(enhanced)
        text = "\n".join([line[1][0] for line in result[0]]) if result and result[0] else ""
        print("==== OCR 결과 ====\n", text)
        stable = 0
        time.sleep(2)

    cv2.imshow("YOLO + OCR", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
