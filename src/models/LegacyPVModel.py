import torch

from ..Game import BOARD_SIZE
from .AbstractPVModel import AbstractPVModel


class LegacyPVModel(AbstractPVModel):

    def __init__(self):
        super().__init__()

        self.backbone = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=2, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * BOARD_SIZE**3, 128),
            torch.nn.ReLU(),
        )
        self.policy_head = torch.nn.Linear(128, BOARD_SIZE**2)
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        features = self.backbone(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value
