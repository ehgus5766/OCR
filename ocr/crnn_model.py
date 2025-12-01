import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNNModel(nn.Module):
    def __init__(self, num_classes):
        """
        Args:
            num_classes (int): len(CHARSET) + 1 (for CTC blank)
        """
        super(CRNNModel, self).__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),          # 1/2

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),          # 1/4

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)), # 1/8 height 유지

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)), # 1/16 height 유지

            nn.Conv2d(512, 512, 2, 1, 0),
            nn.ReLU(True),
        )

        # RNN (Bidirectional LSTM)
        self.lstm = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_classes)  # ← 출력 차원 조정

    def forward(self, x):
        # Input shape: (B, 1, H, W)
        x = self.cnn(x)
        b, c, h, w = x.size()
        assert h == 1, f"expected height=1, got {h}"
        x = x.squeeze(2).permute(0, 2, 1)  # (B, W, C)

        x, _ = self.lstm(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=2)  # (B, W, num_classes)

    def decode(self, preds, charset):
        """
        Greedy decoding with given charset.
        Args:
            preds: (T, num_classes)
            charset: list of characters (same order used for training)
        """
        preds = preds.argmax(2).squeeze(0).cpu().numpy()
        result = []
        prev = -1
        for c in preds:
            if c != prev and c != 0:  # CTC blank=0
                result.append(charset[c-1])  # charset index shift
            prev = c
        return ''.join(result)
