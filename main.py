import cv2
import time
import numpy as np
from detector.yolo_detect import CardDetector
from ocr.craft_crnn_reader import CardOCR
from llm_model.llm_parser import parse_card_with_llm
from config import YOLO_MODEL_PATH,STABLE_REQUIRED,DETECT_INTERVAL,FRAME_MARGIN,CAM_WIDTH,CAM_HEIGHT


def main():
    print("인식 시작")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    detector = CardDetector(YOLO_MODEL_PATH)
    ocr = CardOCR()

    stable_count = 0
    prev_box = None
    last_detect_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h,w, _= frame.shape
        box_w, box_h = 500,300
        x1, y1 = (w - box_w) // 2, (h - box_h) // 2
        x2, y2 = x1 + box_w, y1 + box_h
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        now = time.time()
        if now - last_detect_time < DETECT_INTERVAL:
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        last_detect_time = now

        box = detector.detect_card(frame)
        if box is None:
            stable_count = 0
            prev_box = None
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        x1d, y1d, x2d, y2d = box
        detected= np.array([x1d, y1d, x2d, y2d])

        cx, cy = (x1d + x2d)/2, (y1d + y2d)/2
        w_ratio = (x2d - x1d) / w
        h_ratio = (y2d - y1d) / h

        in_center = 0.25 < cx / w < 0.75 and 0.25 < cy / h < 0.75
        proper_size = 0.4 < w_ratio < 0.95 and 0.4 < h_ratio < 0.95

        if prev_box is not None:
            xA = max(prev_box[0], x1d)
            yA = max(prev_box[1], y1d)
            xB = min(prev_box[2], x2d)
            yB = min(prev_box[3], y2d)
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (prev_box[2]-prev_box[0]) * (prev_box[3] - prev_box[1])
            boxBArea = (x2d - x1d) * (y2d - y1d)
            iou = interArea / float(boxAArea + boxBArea - interArea)
        else:
            iou = 0
        if in_center and proper_size and iou > 0.85:
            stable_count += 1
        else:
            stable_count = 0
        prev_box = detected

        color = (0,225,0) if stable_count < STABLE_REQUIRED else (0,0,255)
        cv2.rectangle(frame,(int(x1d),int(y1d)),(int(x2d),int(y2d)),color,2)
        status_text = "STABILIZING..." if stable_count < STABLE_REQUIRED else "CAPTURING..."
        cv2.putText(frame,status_text,(int(x1d)+10 , int(y1d)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if stable_count >= STABLE_REQUIRED:
            print("분석중...")
            clean_frame = frame.copy()
            pad = FRAME_MARGIN
            x1p, y1p = max(0, int(x1d - pad)), max(0, int(y1d - pad))
            x2p, y2p = min(w, int(x2d + pad)), min(h, int(y2d + pad))
            roi = clean_frame[y1p:y2p, x1p:x2p]
            text, enhanced = ocr.read(roi)
            text = "\n".join([t for t in text.splitlines() if "CAPTUR" not in t])
            cv2.imshow("OCR Input Debug", enhanced)
            cv2.imwrite("debug_ocr_input_new.jpg", enhanced)
            if text.strip():
                print("==== OCR 결과 ====\n", text)
                info = parse_card_with_llm(text)
                print("분류결과 :" ,info)
            else:
                print("OCR 실패 (텍스트 없음)")

            stable_count = 0
            time.sleep(2)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()