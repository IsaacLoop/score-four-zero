import argparse
import os
from pathlib import Path

import torch
from torch.nn.functional import mse_loss
from torch.optim import AdamW
from tqdm import tqdm

from .elo_history import checkpoint_iteration
from .PolicyValueModel import PolicyValueModel
from .ReplayBuffer import ReplayBuffer
from .self_play_parallel import ParallelSelfPlayPool


def learning_rate_at_step(
    step: int,
    total_steps: int,
    peak_lr: float,
    final_lr: float,
    hold_fraction: float = 0.20,
    decay_end_fraction: float = 0.95,
):
    if total_steps <= 1:
        return final_lr

    hold_steps = int(total_steps * hold_fraction)
    decay_end_step = int(total_steps * decay_end_fraction)

    if step < hold_steps:
        return peak_lr

    if step >= decay_end_step:
        return final_lr

    decay_span = max(1, decay_end_step - hold_steps)
    progress = (step - hold_steps) / decay_span
    return peak_lr + (final_lr - peak_lr) * progress


def latest_checkpoint_path(checkpoint_dir: Path):
    checkpoint_paths = list(checkpoint_dir.glob("iteration_*.pt"))
    if not checkpoint_paths:
        return None

    return max(checkpoint_paths, key=checkpoint_iteration)


def train(
    delete_existing_checkpoints: bool = False,
    num_iterations: int = 20_000,
    resume: bool = False,
):
    NUM_ITERATIONS = num_iterations
    GAMES_PER_ITERATION = 32
    TRAIN_STEPS_PER_ITERATION = 128
    BATCH_SIZE = 128
    REPLAY_BUFFER_SIZE = 100_000

    NUM_SIMULATIONS_TRAINING = 100
    C_PUCT = 1.5
    DIRICHLET_ALPHA = 0.3
    DIRICHLET_EPSILON = 0.25
    LEARNING_RATE = 3e-4
    FINAL_LEARNING_RATE = 3e-5
    LR_HOLD_FRACTION = 0.20
    LR_DECAY_END_FRACTION = 0.95
    CHECKPOINT_EVERY_ITERATIONS = 10
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1.0
    NUM_SAMPLING_MOVES = 8
    N_PARALLEL_WORKERS = min(24, GAMES_PER_ITERATION, os.cpu_count() or 1)
    SELF_PLAY_WORKER_MAX_TASKS = 100

    CHECKPOINT_DIR = Path("checkpoints")
    TRAINING_CONFIG = {
        "games_per_iteration": GAMES_PER_ITERATION,
        "train_steps_per_iteration": TRAIN_STEPS_PER_ITERATION,
        "batch_size": BATCH_SIZE,
        "replay_buffer_size": REPLAY_BUFFER_SIZE,
        "num_simulations_training": NUM_SIMULATIONS_TRAINING,
        "c_puct": C_PUCT,
        "dirichlet_alpha": DIRICHLET_ALPHA,
        "dirichlet_epsilon": DIRICHLET_EPSILON,
        "learning_rate": LEARNING_RATE,
        "final_learning_rate": FINAL_LEARNING_RATE,
        "lr_hold_fraction": LR_HOLD_FRACTION,
        "lr_decay_end_fraction": LR_DECAY_END_FRACTION,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "num_sampling_moves": NUM_SAMPLING_MOVES,
    }

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    if delete_existing_checkpoints:
        for checkpoint_path in CHECKPOINT_DIR.glob("iteration_*.pt"):
            checkpoint_path.unlink()

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

    total_train_steps = NUM_ITERATIONS * TRAIN_STEPS_PER_ITERATION
    global_train_step = 0
    current_lr = LEARNING_RATE
    start_iteration = 0

    if resume:
        checkpoint_path = latest_checkpoint_path(CHECKPOINT_DIR)
        if checkpoint_path is None:
            print("No checkpoints found to resume from. Starting fresh.")
        else:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False,
            )
            checkpoint_training_config = checkpoint.get("training_config")
            if checkpoint_training_config != TRAINING_CONFIG:
                raise RuntimeError(
                    "Latest checkpoint is not compatible with current training settings."
                )

            try:
                model.load_state_dict(checkpoint["model_state_dict"])
            except RuntimeError as e:
                raise RuntimeError(
                    "Latest checkpoint is not compatible with the current model architecture."
                ) from e

            optimizer_state_dict = checkpoint.get("optimizer_state_dict")
            if optimizer_state_dict is not None:
                optimizer.load_state_dict(optimizer_state_dict)

            start_iteration = checkpoint_iteration(checkpoint_path) + 1
            global_train_step = int(
                checkpoint.get(
                    "global_train_step",
                    start_iteration * TRAIN_STEPS_PER_ITERATION,
                )
            )
            print(
                f"Resuming from {checkpoint_path.name} at iteration {start_iteration}."
            )

    current_lr = learning_rate_at_step(
        step=global_train_step,
        total_steps=total_train_steps,
        peak_lr=LEARNING_RATE,
        final_lr=FINAL_LEARNING_RATE,
        hold_fraction=LR_HOLD_FRACTION,
        decay_end_fraction=LR_DECAY_END_FRACTION,
    )
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    iteration_bar = tqdm(
        range(start_iteration, NUM_ITERATIONS),
        desc="Iterations",
        position=0,
    )
    iteration_bar.set_postfix({"lr": f"{current_lr:.4e}"})

    for iteration in iteration_bar:

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

            current_lr = learning_rate_at_step(
                step=global_train_step,
                total_steps=total_train_steps,
                peak_lr=LEARNING_RATE,
                final_lr=FINAL_LEARNING_RATE,
                hold_fraction=LR_HOLD_FRACTION,
                decay_end_fraction=LR_DECAY_END_FRACTION,
            )
            iteration_bar.set_postfix({"lr": f"{current_lr:.4e}"})
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

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
            global_train_step += 1

        # 3 - Checkpointing
        if iteration % CHECKPOINT_EVERY_ITERATIONS == 0:
            checkpoint_path = CHECKPOINT_DIR / f"iteration_{iteration}.pt"
            torch.save(
                {
                    "iteration": iteration + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_train_step": global_train_step,
                    "training_config": TRAINING_CONFIG,
                },
                checkpoint_path,
            )

    self_play_pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Score Four Zero model.")
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=20_000,
        help="Number of training iterations to run. Default: 20000.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the highest compatible iteration_*.pt checkpoint.",
    )
    parser.add_argument(
        "--delete-existing-checkpoints",
        action="store_true",
        help="Delete existing iteration_*.pt checkpoints before training starts.",
    )
    args = parser.parse_args()
    if args.resume and args.delete_existing_checkpoints:
        parser.error(
            "--resume and --delete-existing-checkpoints cannot be used together"
        )
    train(
        delete_existing_checkpoints=args.delete_existing_checkpoints,
        num_iterations=args.num_iterations,
        resume=args.resume,
    )
