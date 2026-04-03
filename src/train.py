import os
from pathlib import Path

import torch
from torch.nn.functional import mse_loss
from torch.optim import AdamW
from tqdm import tqdm

from .elo_parallel import ParallelEloPool
from .PolicyValueModel import PolicyValueModel
from .ReplayBuffer import ReplayBuffer
from .self_play_parallel import ParallelSelfPlayPool


def train():
    NUM_ITERATIONS = 5000
    GAMES_PER_ITERATION = 32
    TRAIN_STEPS_PER_ITERATION = 128
    BATCH_SIZE = 128
    REPLAY_BUFFER_SIZE = 100_000

    NUM_SIMULATIONS_TRAINING = 100
    C_PUCT = 1.5
    DIRICHLET_ALPHA = 0.3
    DIRICHLET_EPSILON = 0.25
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1.0
    NUM_SAMPLING_MOVES = 8
    N_PARALLEL_WORKERS = min(24, GAMES_PER_ITERATION, os.cpu_count() or 1)
    SELF_PLAY_WORKER_MAX_TASKS = 100

    STARTING_ELO = 500.0
    NUM_EVALUATION_FIGHTS = 100
    ELO_EVALUATION_WORKER_MAX_TASKS = 100
    NUM_SIMULATIONS_EVALUATION = 25

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
    self_play_pool = ParallelSelfPlayPool(
        max_workers=N_PARALLEL_WORKERS,
        num_simulations=NUM_SIMULATIONS_TRAINING,
        c_puct=C_PUCT,
        dirichlet_alpha=DIRICHLET_ALPHA,
        dirichlet_epsilon=DIRICHLET_EPSILON,
        max_tasks_per_child=SELF_PLAY_WORKER_MAX_TASKS,
    )
    self_play_pool.open()

    elo_evaluation_pool = ParallelEloPool(
        num_simulations=NUM_SIMULATIONS_EVALUATION,
        max_workers=N_PARALLEL_WORKERS,
        c_puct=C_PUCT,
        max_tasks_per_child=ELO_EVALUATION_WORKER_MAX_TASKS,
    )
    elo_evaluation_pool.open()
    elos = []
    model_paths = []

    for iteration in tqdm(
        range(NUM_ITERATIONS),
        desc="Iterations",
        position=0,
    ):

        # 1 - Self-play
        model.eval()
        replay_buffer.push_examples(
            self_play_pool.generate_examples(
                model=model,
                total_games=GAMES_PER_ITERATION,
                num_sampling_moves=NUM_SAMPLING_MOVES,
                iteration=iteration,
                desc="Self-play",
                position=1,
                leave=False,
            )
        )

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

        # 3 - Evaluating
        checkpoint_path = CHECKPOINT_DIR / f"iteration_{iteration + 1:04d}.pt"
        torch.save(
            {
                "iteration": iteration + 1,
                "model_state_dict": model.state_dict(),
            },
            checkpoint_path,
        )
        model_paths.append(checkpoint_path)
        elos.append(STARTING_ELO)

        if len(model_paths) >= 2:
            elos = elo_evaluation_pool.evaluate(
                elos,
                model_paths,
                NUM_EVALUATION_FIGHTS // 2,
                always_last=True,
                desc="Eval latest",
                position=3,
                leave=False,
            )
            elos = elo_evaluation_pool.evaluate(
                elos,
                model_paths,
                NUM_EVALUATION_FIGHTS // 2,
                always_last=False,
                desc="Eval pool",
                position=4,
                leave=False,
            )

    self_play_pool.close()
    elo_evaluation_pool.close()


if __name__ == "__main__":
    train()
