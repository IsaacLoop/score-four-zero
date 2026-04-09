import torch

from ..Game import BOARD_SIZE, TOTAL_CELLS
from .AbstractPVModel import AbstractPVModel


MODEL_DIM = 256  # refered to as d in dimensional comments
NUM_HEADS = 8
NUM_LAYERS = 4
FEEDFORWARD_DIM = 1024
NUM_CELL_STATES = 3


class PVModel(AbstractPVModel):

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.input_dropout = torch.nn.Dropout(p=dropout)
        self.final_norm = torch.nn.LayerNorm(MODEL_DIM)

        self.cell_embedding = torch.nn.Embedding(NUM_CELL_STATES, MODEL_DIM)  # ([0, 1, or 2]) -> (1, d)
        self.position_embedding = torch.nn.Parameter(
            torch.zeros(1, TOTAL_CELLS, MODEL_DIM)
        )  # (1, 64, d)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=MODEL_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=FEEDFORWARD_DIM,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=NUM_LAYERS,
        )  # (b, n, d) -> (b, n, d) | n typically 64 in our case

        self.policy_head = torch.nn.Linear(
            TOTAL_CELLS * MODEL_DIM,
            BOARD_SIZE**2,
        )  # (b, 64d) -> (b, 16)
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(TOTAL_CELLS * MODEL_DIM, MODEL_DIM),
            torch.nn.GELU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(MODEL_DIM, 1),
            torch.nn.Tanh(),
        )  # (b, 64d) -> (b, 1)

    def forward(self, x):
        cell_state_ids = (x[:, 0] + 2.0 * x[:, 1]).to(dtype=torch.long)  # (b, 64) all values are 0, 1, or 2
        tokens = self.cell_embedding(cell_state_ids.reshape(x.shape[0], TOTAL_CELLS))  # (b, 64, d)
        tokens = self.input_dropout(tokens + self.position_embedding) 
        encoded_tokens = self.final_norm(self.encoder(tokens))  # (b, 64, d)
        policy_features = encoded_tokens.reshape(
            x.shape[0],
            TOTAL_CELLS * MODEL_DIM,
        )  # (b, 64d)
        policy_logits = self.policy_head(policy_features) # (b, 16)

        value = self.value_head(policy_features) # (b, 1)
        return policy_logits, value  # (b, 16), (b, 1)
