# vision/card_classifier.py
from dataclasses import dataclass
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2

RANKS = ["A","K","Q","J","10","9","8","7","6","5","4","3","2"]
SUITS = ["c","d","h","s"]

class TinyCornerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.0),  # set 0 for inference
        )
        self.rank_head = nn.Linear(256, 13)
        self.suit_head = nn.Linear(256, 4)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return self.rank_head(x), self.suit_head(x)

@dataclass
class CardPrediction:
    label: str           # e.g. "Qh"
    rank_conf: float
    suit_conf: float
    card_conf: float     # simple combined confidence

class CardClassifier:
    """
    Classifies an already-cropped card-corner image (BGR numpy) into rank+suit.
    """

    def __init__(self, weights_path: str, device: str = "cpu", input_size: int = 96):
        self.device = torch.device(device)
        self.model = TinyCornerNet().to(self.device)
        ckpt = torch.load(weights_path, map_location=self.device)

        # supports either raw state_dict or {"model_state": state_dict}
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        self.model.load_state_dict(state)
        self.model.eval()

        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ])

    @torch.no_grad()
    def predict_corner(self, corner_bgr: np.ndarray) -> CardPrediction:
        # Convert BGR -> RGB because PIL expects RGB
        corner_rgb = cv2.cvtColor(corner_bgr, cv2.COLOR_BGR2RGB)

        x = self.tf(corner_rgb).unsqueeze(0).to(self.device)  # [1,3,H,W]

        rank_logits, suit_logits = self.model(x)

        rank_probs = F.softmax(rank_logits, dim=1)[0]
        suit_probs = F.softmax(suit_logits, dim=1)[0]

        r_idx = int(torch.argmax(rank_probs))
        s_idx = int(torch.argmax(suit_probs))

        rank = RANKS[r_idx]
        suit = SUITS[s_idx]

        rank_conf = float(rank_probs[r_idx])
        suit_conf = float(suit_probs[s_idx])
        card_conf = rank_conf * suit_conf  # simple combine

        return CardPrediction(label=f"{rank}{suit}", rank_conf=rank_conf, suit_conf=suit_conf, card_conf=card_conf)
