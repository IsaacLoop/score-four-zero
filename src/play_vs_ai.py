import numpy as np
import torch

from src.Env import Env
from src.MCTS import MCTS
from src.PolicyValueModel import PolicyValueModel
from src.views import AsciiView

MODEL_CHECKPOINT = "checkpoints/iteration_0406.pt"
NUM_SIMULATIONS = 25
DEVICE = "cpu"

model = PolicyValueModel().to(DEVICE)
model.load_state_dict(
    torch.load(MODEL_CHECKPOINT, map_location=DEVICE)["model_state_dict"]
)
model.eval()

env = Env()
mcts = MCTS(
    model,
    num_simulations=NUM_SIMULATIONS,
    c_puct=1.5,
    add_exploration_noise=False,
)
view = AsciiView(env.game)

while not env.is_terminal():
    if env.current_player() == -1:
        view.update()
        print("You: X, AI: O")
        action = input("Enter your move as xy: ")
        print(action)
        action = int(action[0]) * 4 + int(action[1])
    else:
        root = mcts.run(root_env=env)
        pi = mcts.visit_counts_to_policy(root=root, temperature=0.0)
        action = int(np.argmax(pi))
        print(f"AI plays: x={action//4}, y={action%4}")
    env.step(action)
view.update()
print("You: X, AI: O")
winner = env.winner()
if winner == 0:
    print("It's a draw!")
elif winner == 1:
    print("AI wins!")
else:
    print("You win! Congratulations!")