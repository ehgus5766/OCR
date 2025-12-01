import cv2
import numpy as np
from paddleocr import PaddleOCR
import time
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
ocr = PaddleOCR(lang='korean')
cap = cv2.VideoCapture(0)
stable = 0

def enhance_image(img):
    # 명함 부분 대비 강화 + 밝기 자동 보정
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.3, beta=20)  # 대비 ↑, 밝기 약간 ↑
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # Adaptive Thresholding (글씨 강조)
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 9
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

def four_point_transform(image, pts):
    # pts의 점이 4개 초과할 때 자동 보정
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
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

while True:
    ret, frame = cap.read()
    display_frame = cv2.flip(frame, 1)
    if not ret:
        break

    #frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    box_w, box_h = 600, 350
    x1, y1 = (w - box_w)//2, (h - box_h)//2
    x2, y2 = x1 + box_w, y1 + box_h
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=2.0, beta=40)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 30, 100)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    cv2.imshow("Canny Edge (ROI)", edges)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = None

    for c in contours:
        area = cv2.contourArea(c)
        if area < 8000 or area > 200000:  # 너무 작거나 큰 객체는 제외
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if 4 <= len(approx) <= 8:
            detected = approx.reshape(-1, 2)
            cv2.drawContours(roi, [approx], -1, (0, 255, 0), 2)
            stable += 1
            break

    if detected is None or len(detected) == 0:
        stable = 0

    if stable > 8 and detected is not None:
        print(" 명함 인식 시도 중...")
        warped = four_point_transform(roi, detected)
        enhanced = enhance_image(warped)
        result = ocr.ocr(enhanced)
        text = "\n".join([line[1][0] for line in result[0]]) if result and result[0] else ""
        print("==== OCR 결과 ====\n", text)
        stable = 0
        time.sleep(2)

    cv2.imshow("Business Card OCR", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
