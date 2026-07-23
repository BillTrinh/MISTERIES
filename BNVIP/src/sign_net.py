"""Small GRU classifier for isolated dynamic signs (keypoint sequences)."""
import torch.nn as nn

from sequence_utils import SIGN_DIM


class SignGRU(nn.Module):
    def __init__(self, n_classes, dim=SIGN_DIM, hidden=128, layers=1, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(dim, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):                 # x: (B, T, 126)
        out, _ = self.gru(x)
        return self.head(out[:, -1])      # last timestep -> logits
