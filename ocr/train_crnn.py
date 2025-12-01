import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ocr.crnn_model import CRNNModel


# -----------------------------
#  Dataset 정의
# -----------------------------
class OCRDataset(Dataset):
    def __init__(self, image_dir, label_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []

        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                path, text = line.strip().split('\t')
                self.samples.append((os.path.join(image_dir, path), text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"이미지 로드 실패: {img_path}")

        # CRNN 입력 크기 통일
        image = cv2.resize(image, (100, 32))
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32) / 255.0

        return torch.tensor(image), text


# -----------------------------
#  CTC용 문자열 인코더
# -----------------------------
class LabelConverter:
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.char_to_idx = {c: i + 1 for i, c in enumerate(alphabet)}  # 0은 blank
        self.idx_to_char = {i + 1: c for i, c in enumerate(alphabet)}

    def encode(self, text_list):
        encoded = []
        lengths = []

        for t in text_list:
            filtered = [self.char_to_idx[c] for c in t if c in self.char_to_idx]
            encoded.extend(filtered)
            lengths.append(len(filtered))

        return torch.IntTensor(encoded), torch.IntTensor(lengths)

    def decode(self, preds):
        texts = []
        for pred in preds:
            text = ''.join([self.idx_to_char.get(c, '') for c in pred])
            texts.append(text)
        return texts


# -----------------------------
#  학습 루프
# -----------------------------
def train_crnn():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #  완성형 한글 전체 + 영문 + 숫자 + 기호
    kor = [chr(i) for i in range(0xAC00, 0xD7A4)]  # 가~힣
    eng = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    num = list("0123456789")
    sym = list("@.-_()+/:,;&!?₩$# ")
    alphabet = kor + eng + num + sym

    print(f"총 문자 수: {len(alphabet)}")

    converter = LabelConverter(alphabet)

    dataset = OCRDataset(
        image_dir="C:/Users/user/Desktop/CV/tools/dataset/images",
        label_path="C:/Users/user/Desktop/CV/tools/dataset/labels.txt",
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    model = CRNNModel(num_classes=len(alphabet) + 1).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float('inf')

    for epoch in range(50):
        model.train()
        total_loss = 0

        for i, (images, texts) in enumerate(dataloader):
            images = images.to(device)

            # Forward
            logits = model(images)
            log_probs = logits.permute(1, 0, 2).log_softmax(2)

            T, N, C = log_probs.size()
            input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long).to(device)
            targets, lengths = converter.encode(texts)
            targets = targets.to(device)
            lengths = lengths.to(device)

            loss = criterion(log_probs, targets, input_lengths, lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % 10 == 0:
                print(f"[Epoch {epoch + 1}][{i}/{len(dataloader)}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} 평균 손실: {avg_loss:.4f}")

        # 모델 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs("weights", exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "alphabet": alphabet
            }, "weights/crnn_best.pth")
            print(" Best model updated and saved!")

    print("학습 완료.")


if __name__ == "__main__":
    train_crnn()
