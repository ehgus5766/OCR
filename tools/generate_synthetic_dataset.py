import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ----------------------------
# 1️ 폰트 설정
# ----------------------------
FONT_KOR = [
    "C:/Windows/Fonts/malgun.ttf",   # 한글 지원
    "C:/Windows/Fonts/batang.ttc",
    "C:/Windows/Fonts/gulim.ttc"
]

FONT_ENG = [
    "C:/Windows/Fonts/arial.ttf",    # 영문 전용
    "C:/Windows/Fonts/times.ttf"
]

FONT_KOR = [f for f in FONT_KOR if os.path.exists(f)]
FONT_ENG = [f for f in FONT_ENG if os.path.exists(f)]


if not FONT_KOR:
    raise RuntimeError("한글 폰트가 존재하지 않습니다.")
if not FONT_ENG:
    raise RuntimeError("영문 폰트가 존재하지 않습니다.")

OUTPUT_DIR = "dataset"
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
# ----------------------------
# 문자 세트 정의 (최대 커버)
# ----------------------------
KOR = [chr(i) for i in range(0xAC00, 0xD7A4)]
ENG = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
NUM = list("0123456789")
SYM = list("@.-_()+/:,;&!?₩$# ")

CHARSET = KOR + ENG + NUM + SYM  # 전체 문자 풀
print(f"총 {len(CHARSET)}자 문자 세트 생성")


# ----------------------------
#  한글 포함 여부 판별
# ----------------------------
def has_korean(text):
    return any('가' <= ch <= '힣' for ch in text)

# ----------------------------
# 랜덤 텍스트 생성기
# ----------------------------

def random_text():
    # 3~15 글자 랜덤 문자열
    length = random.randint(3, 15)
    return ''.join(random.choices(CHARSET, k=length))

# ----------------------------
# Synthetic 이미지 생성
# ----------------------------
def generate_image(text, img_id):
    #  한글 포함 여부에 따라 폰트 분리
    if has_korean(text):
        font_path = random.choice(FONT_KOR)
    else:
        font_path = random.choice(FONT_ENG)

    font_size = random.randint(22, 42)
    font = ImageFont.truetype(font_path, font_size)

    img = Image.new("RGB", (240, 70), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = random.randint(0, max(0, img.width - w))
    y = random.randint(0, max(0, img.height - h))
    draw.text((x, y), text, font=font, fill=(0, 0, 0))

    img = np.array(img)

    # 현실 왜곡 시뮬레이션
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    if random.random() < 0.4:
        img = cv2.add(img, np.random.randint(0, 30, img.shape, dtype=np.uint8))
    if random.random() < 0.3:
        angle = random.uniform(-5, 5)
        M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1.0)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=(255,255,255))

    filename = f"{img_id:05d}.jpg"
    cv2.imwrite(os.path.join(OUTPUT_DIR, "images", filename), img)
    return filename

# ----------------------------
#  데이터셋 생성
# ----------------------------
def generate_dataset(n=6000):
    labels = []
    for i in range(n):
        text = random_text()
        filename = generate_image(text, i)
        labels.append(f"{filename}\t{text}")

    with open(os.path.join(OUTPUT_DIR, "labels.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(labels))

    # alphabet.txt 자동 생성
    unique_chars = sorted(set("".join([t.split("\t")[1] for t in labels])))
    with open(os.path.join(OUTPUT_DIR, "alphabet.txt"), "w", encoding="utf-8") as f:
        f.write("".join(unique_chars))

    print(f" {n} synthetic OCR samples generated in {OUTPUT_DIR}/images")
    print(f" Unique chars: {len(unique_chars)} (saved to alphabet.txt)")

# ----------------------------
if __name__ == "__main__":
    generate_dataset(8000)
