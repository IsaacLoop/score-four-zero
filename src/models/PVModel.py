import torch

from ..Game import BOARD_SIZE
from .AbstractPVModel import AbstractPVModel


class PVModel(AbstractPVModel):

    def __init__(self):
        super().__init__()

        self.backbone = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=2, out_channels=24, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(24),
            torch.nn.ReLU(),

            torch.nn.Conv3d(in_channels=24, out_channels=48, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(48),
            torch.nn.ReLU(),

            torch.nn.Conv3d(in_channels=48, out_channels=96, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(96),
            torch.nn.ReLU(),
            
            torch.nn.Flatten(),
            # Note to myself: 92% of all trainable parameters are there. Should be improved in run-3!
            torch.nn.Linear(96 * BOARD_SIZE**3, 384),
            torch.nn.BatchNorm1d(384),
            torch.nn.ReLU(),
        )
        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(384, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, BOARD_SIZE**2),
        )
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(384, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        features = self.backbone(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value
