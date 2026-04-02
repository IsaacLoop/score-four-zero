"""
Note:
- the whole parallelization aspect of this file has been done by a LLM
- i couldn't explain it exactly and i don't care
- it worked before - it works the same now, but faster => good enough for me
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import mse_loss
from torch.optim import AdamW
from tqdm import tqdm

from .Env import Env
from .MCTS import MCTS
from .PolicyValueModel import PolicyValueModel
from .ReplayBuffer import ReplayBuffer

_SELF_PLAY_WORKER_INITIALIZED = False


def play_one_self_play_game(
    env: Env,
    mcts: MCTS,
    num_sampling_moves: int = 8,
):
    """
    Play one game of self-play, returning a list of training examples (state, MCTS policy, game outcome).
    """
    trajectory = []
    env.reset()

    move_index = 0

    while not env.is_terminal():
        root = mcts.run(root_env=env)
        if move_index < num_sampling_moves:
            temperature = 1.0
        else:
            temperature = 0.0
        pi = mcts.visit_counts_to_policy(root=root, temperature=temperature)

        player = env.current_player()
        x = env.canonical_state(perspective_player=player)

        trajectory.append((x, pi, player))

        action = np.random.choice(len(pi), p=pi)
        env.step(action)

        move_index += 1

    examples = []

    for x, pi, player in trajectory:
        z = env.terminal_value(perspective_player=player)
        examples.append((x, pi, z))

    return examples


def _split_games_across_workers(total_games: int, num_workers: int):
    num_workers = max(1, min(total_games, num_workers))
    base_games_per_worker, extra_games = divmod(total_games, num_workers)

    return [
        base_games_per_worker + (1 if worker_index < extra_games else 0)
        for worker_index in range(num_workers)
        if base_games_per_worker + (1 if worker_index < extra_games else 0) > 0
    ]


def _cpu_model_state_dict(model: PolicyValueModel):
    return {
        parameter_name: parameter.detach().cpu()
        for parameter_name, parameter in model.state_dict().items()
    }


def _play_self_play_games_worker(
    model_state_dict,
    num_games: int,
    num_sampling_moves: int,
    num_simulations: int,
    c_puct: float,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    seed: int,
):
    global _SELF_PLAY_WORKER_INITIALIZED

    if not _SELF_PLAY_WORKER_INITIALIZED:
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        _SELF_PLAY_WORKER_INITIALIZED = True

    np.random.seed(seed)
    torch.manual_seed(seed)

    model = PolicyValueModel()
    model.load_state_dict(model_state_dict)
    model.eval()

    mcts = MCTS(
        model=model,
        num_simulations=num_simulations,
        c_puct=c_puct,
        add_exploration_noise=True,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
    )

    examples = []

    for _ in range(num_games):
        env = Env()
        examples.extend(
            play_one_self_play_game(
                env=env,
                mcts=mcts,
                num_sampling_moves=num_sampling_moves,
            )
        )

    return examples


def train():
    NUM_ITERATIONS = 1000
    GAMES_PER_ITERATION = 32
    TRAIN_STEPS_PER_ITERATION = 128
    BATCH_SIZE = 128
    REPLAY_BUFFER_SIZE = 100_000

    NUM_SIMULATIONS = 100
    C_PUCT = 1.5
    DIRICHLET_ALPHA = 0.3
    DIRICHLET_EPSILON = 0.25
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1.0
    NUM_SAMPLING_MOVES = 8
    SELF_PLAY_WORKERS = min(24, GAMES_PER_ITERATION, os.cpu_count() or 1)

    CHECKPOINT_DIR = Path("checkpoints")
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} " + ("🥰" if device.type == "cuda" else "😢"))

    model = PolicyValueModel().to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_SIZE)

    worker_context = get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=SELF_PLAY_WORKERS,
        mp_context=worker_context,
    ) as executor:
        for iteration in tqdm(
            range(NUM_ITERATIONS),
            desc="Iterations",
            position=0,
        ):

            # 1 - Self-play
            model.eval()
            model_state_dict = _cpu_model_state_dict(model)
            worker_game_counts = _split_games_across_workers(
                total_games=GAMES_PER_ITERATION,
                num_workers=SELF_PLAY_WORKERS,
            )
            future_to_num_games = {}

            for worker_index, num_games in enumerate(worker_game_counts):
                future = executor.submit(
                    _play_self_play_games_worker,
                    model_state_dict,
                    num_games,
                    NUM_SAMPLING_MOVES,
                    NUM_SIMULATIONS,
                    C_PUCT,
                    DIRICHLET_ALPHA,
                    DIRICHLET_EPSILON,
                    iteration * SELF_PLAY_WORKERS + worker_index,
                )
                future_to_num_games[future] = num_games

            with tqdm(
                total=GAMES_PER_ITERATION,
                desc="Self-play",
                position=1,
                leave=False,
            ) as self_play_bar:
                for future in as_completed(future_to_num_games):
                    replay_buffer.push_examples(future.result())
                    self_play_bar.update(future_to_num_games[future])

            if len(replay_buffer) < BATCH_SIZE:
                continue

            # 2 - Training
            model.train()
            for _ in tqdm(
                range(TRAIN_STEPS_PER_ITERATION),
                desc="Training",
                position=2,
                leave=False,
            ):
                x_batch, pi_batch, z_batch = replay_buffer.sample_batch(BATCH_SIZE)

                x_batch = torch.as_tensor(x_batch, dtype=torch.float32, device=device)
                pi_batch = torch.as_tensor(
                    pi_batch,
                    dtype=torch.float32,
                    device=device,
                )
                z_batch = torch.as_tensor(
                    z_batch,
                    dtype=torch.float32,
                    device=device,
                )

                optimizer.zero_grad(set_to_none=True)
                policy_logits_batch, value_batch = model(x_batch)

                # Hard to understand, so basically:
                # 1) if the target (pi) says an action is important,
                # 2) and the model gives it a small probability (softmax of logits),
                # 3) then punish this hard by turning the model probability into a big negative number (-log)
                policy_loss = (
                    -(pi_batch * torch.log_softmax(policy_logits_batch, dim=-1))
                    .sum(dim=-1)
                    .mean()
                )

                value_loss = mse_loss(value_batch.squeeze(-1), z_batch)

                loss = policy_loss + value_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

            # Final - Saving
            checkpoint_path = CHECKPOINT_DIR / f"iteration_{iteration + 1:04d}.pt"
            torch.save(
                {
                    "iteration": iteration + 1,
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )


if __name__ == "__main__":
    train()
