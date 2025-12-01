import cv2
import numpy as np
from paddleocr import PaddleOCR
import time
from ultralytics import YOLO

# YOLO 모델 (명함 감지용)
model = YOLO("runs/detect/train2/weights/best.pt")

# OCR 초기화
ocr = PaddleOCR(lang='korean', use_textline_orientation=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

stable_count = 0
prev_box = None
STABLE_REQUIRED = 10       # 안정 프레임 기준 (8프레임 이상 유지 시 인식)
DETECT_INTERVAL = 0.3     # 감지 주기 제한 (초)
last_detect_time = 0
FRAME_MARGIN = 80       # OCR crop padding

def enhance_image(img):
    # 밝기 자동 감지
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    # 너무 밝으면 어둡게 (감마 증가)
    if brightness > 170:
        gamma = 1.6
    elif brightness < 80:
        gamma = 0.7
    else:
        gamma = 1.0

    # 감마 보정
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    img = cv2.LUT(img, table)

    # CLAHE로 글자 대비 강화
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    y = clahe.apply(y)
    ycrcb = cv2.merge((y, cr, cb))
    enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # 경계 강조 (Unsharp mask)
    blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
    enhanced = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)

    #  명도 살짝 조정 (너무 하얀 배경 줄이기)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = np.clip(hsv[..., 2] * 0.9, 0, 255)
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # OCR 입력 크기 확보
    if enhanced.shape[1] < 1000:
        scale = 1280 / enhanced.shape[1]
        enhanced = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    return enhanced





print(" 명함 자동 인식 시스템 시작")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # 중앙 네모 프레임 정의 (참조 박스)
    box_w, box_h = 500, 300
    x1, y1 = (w - box_w)//2, (h - box_h)//2
    x2, y2 = x1 + box_w, y1 + box_h
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    # Canny Edge 항상 표시
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    cv2.imshow("Canny Edge", edges)

    now = time.time()
    if now - last_detect_time < DETECT_INTERVAL:
        cv2.imshow("YOLO + OCR", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    last_detect_time = now

    # YOLO 감지 실행
    results = model(frame, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 else []

    detected = None
    if len(boxes) > 0:
        # 가장 큰 박스 선택
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
        idx = int(np.argmax(areas))
        x1d, y1d, x2d, y2d = boxes[idx]
        detected = np.array([x1d, y1d, x2d, y2d])

        # 중심/크기 비율 계산
        cx, cy = (x1d + x2d)/2, (y1d + y2d)/2
        w_ratio = (x2d - x1d) / w
        h_ratio = (y2d - y1d) / h
        in_center = 0.4 < cx/w < 0.6 and 0.4 < cy/h < 0.6
        proper_size = 0.3 < w_ratio < 0.9 and 0.2 < h_ratio < 0.9

        # 안정도 측정 (IoU)
        if prev_box is not None:
            xA = max(prev_box[0], x1d)
            yA = max(prev_box[1], y1d)
            xB = min(prev_box[2], x2d)
            yB = min(prev_box[3], y2d)
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (prev_box[2] - prev_box[0]) * (prev_box[3] - prev_box[1])
            boxBArea = (x2d - x1d) * (y2d - y1d)
            iou = interArea / float(boxAArea + boxBArea - interArea)
        else:
            iou = 0

        if in_center and proper_size and iou > 0.85:
            stable_count += 1
        else:
            stable_count = 0
        prev_box = detected
        progress = stable_count / STABLE_REQUIRED
        progress = min(progress, 1.0)
        color = (
            int(255 * progress),
            int(255 * (1 - progress)),
            0
        )

        # 박스 표시
        color = (0,255,0) if stable_count < STABLE_REQUIRED else (0,0,255)
        cv2.rectangle(frame, (int(x1d), int(y1d)), (int(x2d), int(y2d)), color, 2)
        status_text = "STABILIZING..." if stable_count < STABLE_REQUIRED else "CAPTURING..."
        cv2.putText(frame, status_text, (int(x1d) + 10, int(y1d) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < 50:
            continue
        if stable_count >= STABLE_REQUIRED:
            print(" 안정 프레임 도달 — OCR 시도 중...")

            pad = FRAME_MARGIN
            x1p, y1p = max(0, int(x1d - pad)), max(0, int(y1d - pad))
            x2p, y2p = min(w, int(x2d + pad)), min(h, int(y2d + pad))
            roi = frame[y1p:y2p, x1p:x2p]

            enhanced = enhance_image(roi)
            if enhanced.shape[1] < 1000:
                scale = 1280 / enhanced.shape[1]
                enhanced = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            cv2.imshow("OCR Input Debug", enhanced)
            cv2.imwrite("debug_ocr_input_new.jpg", enhanced)
            result = ocr.predict(enhanced)
            if result and result[0]:
                text = "\n".join([line[1][0] for line in result[0]])
                print("==== OCR 결과 ====\n", text)
            else:
                print(" OCR 실패 (텍스트 없음)")

            stable_count = 0
            time.sleep(2)  # OCR 후 감지 잠시 중단

    else:
        stable_count = 0
        prev_box = None

    cv2.imshow("YOLO + OCR", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
