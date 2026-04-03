import argparse
from pathlib import Path

import numpy as np
import torch

from src.Env import Env
from src.MCTS import MCTS
from src.PolicyValueModel import PolicyValueModel
from src.views import AsciiView


def resolve_checkpoint_path(checkpoint):
    checkpoint_dir = Path("checkpoints")

    if checkpoint == "last":
        checkpoint_paths = sorted(checkpoint_dir.glob("iteration_*.pt"))
        if not checkpoint_paths:
            raise FileNotFoundError("No checkpoints found in checkpoints/.")
        return checkpoint_paths[-1]

    checkpoint_number = int(checkpoint)
    checkpoint_path = checkpoint_dir / f"iteration_{checkpoint_number}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def play_vs_ai(checkpoint="last"):
    checkpoint_path = resolve_checkpoint_path(checkpoint)
    NUM_SIMULATIONS = 25
    DEVICE = "cpu"

    model = PolicyValueModel().to(DEVICE)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=DEVICE)["model_state_dict"]
    )
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print()

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
            view.update(minimal=True)
            print()
            print("You: X | AI: O")
            action = input("Enter your move as xy: ")
            print(f"You play: x={action[0]}, y={action[1]}")
            action = int(action[0]) * 4 + int(action[1])
        else:
            root = mcts.run(root_env=env)
            pi = mcts.visit_counts_to_policy(root=root, temperature=0.0)
            action = int(np.argmax(pi))
            print(f"AI plays: x={action//4}, y={action%4}")
            print()
        env.step(action)
    view.update(minimal=True)
    print()
    print("You: X | AI: O")
    winner = env.winner()
    if winner == 0:
        print("It's a draw!")
    elif winner == 1:
        print("AI wins!")
    else:
        print("You win! Congratulations!")


def main():
    parser = argparse.ArgumentParser(description="Play against a trained model.")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="last",
        help='Checkpoint number to load, or "last" for the most recent one.',
    )
    args = parser.parse_args()
    play_vs_ai(args.checkpoint)


if __name__ == "__main__":
    main()
