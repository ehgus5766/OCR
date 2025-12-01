import cv2
import torch
import numpy as np
from craft_text_detector import Craft
from utils.image_utils import enhance_image
from ocr.crnn_model import CRNNModel
from ocr.train_crnn import LabelConverter
device = "cuda" if torch.cuda.is_available() else "cpu"

def preprocess_card(image):
    """
    1️ 명함 외곽 contour 제거
    2️ 중심부 ROI 확대 crop
    3️ 텍스트 대비 강화
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # 가장 큰 contour를 명함 외곽으로 판단
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cropped = image[y+10:y+h-10, x+10:x+w-10]  # 둥근 모서리 여백 제거
    else:
        cropped = image

    # contrast 조정 (CLAHE 사용)
    lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 리사이즈 (CRAFT 안정구간)
    h, w = enhanced.shape[:2]
    if w < 960:
        scale = 960 / w
        enhanced = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

    return enhanced


def rect_approx(polys, img_shape):
    rects = []
    h, w = img_shape[:2]
    for p in polys:
        try:
            p = np.array(p).reshape(-1, 2).astype(np.int32)
            epsilon = 0.02 * cv2.arcLength(p, True)
            approx = cv2.approxPolyDP(p, epsilon, True)
            if len(approx) == 4:
                rects.append(approx.reshape(4, 2))
        except Exception:
            continue
    #  폴리곤이 비었을 때 fallback (명함 전체를 OCR)
    if len(rects) == 0:
        rects = [np.array([[0,0],[w,0],[w,h],[0,h]], np.int32)]
    return rects

def resize_for_craft(img, min_width=768):
    h, w = img.shape[:2]
    if w < min_width:
        scale = min_width / w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    return img

def crop_text_region(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    cropped = img[y:y+h, x:x+w]
    return cropped
def merge_boxes(polys, y_tolerance=10):
    polys = sorted(polys, key=lambda b: np.min(np.array(b)[:,1]))
    merged = []
    current_line = []
    prev_y = None
    for p in polys:
        y_mean = np.mean(np.array(p)[:,1])
        if prev_y is None or abs(y_mean - prev_y) < y_tolerance:
            current_line.append(p)
        else:
            merged.append(current_line)
            current_line = [p]
        prev_y = y_mean
    if current_line:
        merged.append(current_line)
    return [np.concatenate(line) for line in merged]

def ctc_decode(pred_idx, converter):
    pred_idx = pred_idx.cpu().numpy()
    collapsed = []
    prev = -1
    for p in pred_idx:
        if p != prev and p != 0:  # 0은 blank
            collapsed.append(p)
        prev = p
    return converter.decode([np.array(collapsed)])[0]
class CardOCR:
    def __init__(self, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print("🔹 [CRAFT + CRNN] OCR 엔진 초기화 중...")

        # Craft
        self.craft = Craft(output_dir=None, crop_type="box", cuda=(device == "cuda"))

        # Alphabet 및 모델 설정
        checkpoint = torch.load("C:/Users/user/Desktop/CV/ocr/weights/crnn_best.pth", map_location=device)
        alphabet = checkpoint["alphabet"]
        num_classes = len(alphabet) + 1
        self.model = CRNNModel(num_classes=num_classes).to(device)
        self.converter = LabelConverter(alphabet)
        #  저장된 checkpoint 로드
        weight_path = "C:/Users/user/Desktop/CV/ocr/weights/crnn_best.pth"
        checkpoint = torch.load(weight_path, map_location=device)

        # 체크포인트 형식에 따라 처리
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(" Loaded model_state_dict from checkpoint.")
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        self.device = device
        self.alphabet = alphabet

    def read(self, image):
        #  전처리 (명암 및 해상도 보정)
        enhanced = preprocess_card(image)  # ← 교체
        try:
            prediction = self.craft.detect_text(enhanced)
        except ValueError:
            print("[CRAFT] polygon decode error → skip frame")
            return "", enhanced

        polys = prediction.get("boxes", [])
        polys = merge_boxes(polys)
        if polys is None or len(polys) == 0:
            print("[CRAFT] no boxes detected")
            return "", enhanced

        #  polygon을 (4,2) 정규화
        rect_polys = rect_approx(polys, enhanced.shape)

        if len(rect_polys) == 0:
            print("[CRAFT] invalid polygon shapes (skipped)")
            return "", enhanced

        #  OCR 입력 준비
        texts = []
        for box in rect_polys:
            x_min, y_min = np.min(box[:, 0]), np.min(box[:, 1])
            x_max, y_max = np.max(box[:, 0]), np.max(box[:, 1])
            cropped = enhanced[int(y_min):int(y_max), int(x_min):int(x_max)]

            # grayscale resize
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (100, 32))  # 반드시 학습 시와 동일
            gray = gray.astype(np.float32) / 255.0
            gray = (gray - 0.5) / 0.5  # 표준화 추가
            img_tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(self.device)

            with torch.no_grad():
                preds = self.model(img_tensor)
                preds = preds.log_softmax(2)
                _, pred_idx = preds.max(2)
                pred_idx = pred_idx.squeeze(0)
                text = ctc_decode(pred_idx, self.converter)
                texts.append(text)

        full_text = "\n".join(texts)
        return full_text, enhanced

    def unload(self):
        self.craft.unload_craftnet_model()
        self.craft.unload_refinenet_model()
