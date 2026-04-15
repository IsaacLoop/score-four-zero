import argparse
import math
import os
import shutil
from pathlib import Path

import torch
from torch.nn.functional import mse_loss
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .ReplayBuffer import ReplayBuffer
from .elo_parallel import (
    FightPoolError,
    parallel_fight_path_results,
    remap_fight_result,
)
from .models import PVModel
from .self_play_parallel import ParallelSelfPlayPool


def learning_rate_at_step(
    step: int,
    total_steps: int,
    peak_lr: float,
    final_lr: float,
    warmup_fraction: float = 0.0,
):
    if total_steps <= 1:
        return peak_lr

    warmup_steps = max(0, min(total_steps - 1, int(total_steps * warmup_fraction)))
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * ((step + 1) / warmup_steps)

    cosine_total_steps = max(1, total_steps - warmup_steps)
    cosine_step = min(cosine_total_steps - 1, max(0, step - warmup_steps))
    progress = cosine_step / max(1, cosine_total_steps - 1)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final_lr + (peak_lr - final_lr) * cosine_factor


def format_learning_rate(value: float):
    if value == 0.0:
        return "0"
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def checkpoint_iteration_from_path(checkpoint_path: Path) -> int:
    return int(checkpoint_path.stem.split("_")[-1])


def sorted_iteration_checkpoint_paths(checkpoint_dir: Path):
    return sorted(
        checkpoint_dir.glob("iteration_*.pt"),
        key=checkpoint_iteration_from_path,
    )


def previous_checkpoint_winrate(
    previous_checkpoint_path: Path,
    current_checkpoint_path: Path,
    num_simulations: int,
    max_workers: int,
    c_puct: float,
) -> float:
    path_matchups = []
    swapped_flags = []

    for fight_index in (0, 1):
        swapped = fight_index % 2 == 1
        swapped_flags.append(swapped)
        if swapped:
            path_matchups.append(
                (
                    str(previous_checkpoint_path),
                    str(current_checkpoint_path),
                )
            )
        else:
            path_matchups.append(
                (
                    str(current_checkpoint_path),
                    str(previous_checkpoint_path),
                )
            )

    results = parallel_fight_path_results(
        path_matchups=path_matchups,
        num_simulations=num_simulations,
        max_workers=max_workers,
        c_puct=c_puct,
        desc="Evaluation",
        position=3,
        leave=False,
        retry_desc="Evaluation pool",
    )

    score = 0.0
    for result, swapped in zip(results, swapped_flags):
        remapped_result = remap_fight_result(result, swapped)
        if remapped_result == -1:
            score += 1.0
        elif remapped_result == 0:
            score += 0.5

    return score / 2.0


def train(
    num_iterations: int,
    delete_existing_checkpoints: bool = False,
    workers: int = 8,
    skip_evaluation: bool = False,
    resume: bool = False,
):
    NUM_ITERATIONS = num_iterations
    GAMES_PER_ITERATION = 512
    TRAIN_STEPS_PER_ITERATION = 2048
    BATCH_SIZE = 512
    REPLAY_BUFFER_SIZE = 1_000_000

    NUM_SIMULATIONS_TRAINING = 100
    C_PUCT = 1.5
    DIRICHLET_ALPHA = 0.3
    DIRICHLET_EPSILON = 0.25
    LEARNING_RATE = 5e-4
    FINAL_LEARNING_RATE = 0.0
    WARMUP_FRACTION = 0.01
    CHECKPOINT_EVERY_ITERATIONS = 1
    EVAL_CHECKPOINT_GAPS = (2, 5, 10)
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1.0
    NUM_SAMPLING_MOVES = 8
    SELF_PLAY_WORKERS = workers
    LIVE_GAMES_PER_WORKER = 64
    EVALUATOR_MAX_BATCH_SIZE = 1024
    EVALUATOR_MAX_WAIT_S = 0.001
    CPU_TAIL_FRACTION = 0.02
    EVALUATION_WORKERS = min(8, workers, os.cpu_count() or 1)
    TENSORBOARD_DIR = Path("tb_logs")

    CHECKPOINT_DIR = Path("checkpoints")

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    if resume and delete_existing_checkpoints:
        raise ValueError(
            "--resume and --delete-existing-checkpoints cannot be used together."
        )
    if delete_existing_checkpoints:
        for checkpoint_path in CHECKPOINT_DIR.glob("iteration_*.pt"):
            checkpoint_path.unlink()
    if TENSORBOARD_DIR.exists() and not resume:
        shutil.rmtree(TENSORBOARD_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PVModel().to(device)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fused=device.type == "cuda",
    )
    replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_SIZE)
    writer = SummaryWriter(TENSORBOARD_DIR)
    self_play_pool = ParallelSelfPlayPool(
        num_simulations=NUM_SIMULATIONS_TRAINING,
        c_puct=C_PUCT,
        dirichlet_alpha=DIRICHLET_ALPHA,
        dirichlet_epsilon=DIRICHLET_EPSILON,
        max_workers=SELF_PLAY_WORKERS,
        live_games_per_worker=LIVE_GAMES_PER_WORKER,
        evaluator_max_batch_size=EVALUATOR_MAX_BATCH_SIZE,
        evaluator_max_wait_s=EVALUATOR_MAX_WAIT_S,
        cpu_tail_fraction=CPU_TAIL_FRACTION,
    )
    self_play_pool.open()

    existing_checkpoint_paths = sorted_iteration_checkpoint_paths(CHECKPOINT_DIR)
    start_iteration = 0
    saved_checkpoint_paths = list(existing_checkpoint_paths)

    if resume and existing_checkpoint_paths:
        latest_checkpoint_path = existing_checkpoint_paths[-1]
        latest_checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        model.load_state_dict(latest_checkpoint["model_state_dict"])
        start_iteration = checkpoint_iteration_from_path(latest_checkpoint_path) + 1

    total_train_steps = NUM_ITERATIONS * TRAIN_STEPS_PER_ITERATION
    global_train_step = start_iteration * TRAIN_STEPS_PER_ITERATION
    current_lr = LEARNING_RATE

    current_lr = learning_rate_at_step(
        step=global_train_step,
        total_steps=total_train_steps,
        peak_lr=LEARNING_RATE,
        final_lr=FINAL_LEARNING_RATE,
        warmup_fraction=WARMUP_FRACTION,
    )
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    print("- Device: " + ("Cuda 🥰" if device.type == "cuda" else "CPU 🥹"))
    print(f"- Trainable parameters: {trainable_parameters:,}")
    print(f"- Self-play games per iteration: {GAMES_PER_ITERATION:,}")
    print(f"- Self-play workers: {SELF_PLAY_WORKERS:,}")
    print(f"- Live games per worker: {LIVE_GAMES_PER_WORKER:,}")
    print(f"- Evaluator max batch size: {EVALUATOR_MAX_BATCH_SIZE:,}")
    print(f"- Evaluation workers: {EVALUATION_WORKERS:,}")
    print(f"- Total iterations: {NUM_ITERATIONS:,}")
    if resume:
        if existing_checkpoint_paths:
            print(
                "- Resume: "
                f"loaded {existing_checkpoint_paths[-1].name} and continuing from iteration {start_iteration:,}."
            )
        else:
            print("- Resume: no checkpoint found, starting from scratch.")
    print(
        "- LR schedule: "
        f"linear warmup over {WARMUP_FRACTION:.0%}, then cosine decay from {format_learning_rate(LEARNING_RATE)} "
        f"to {format_learning_rate(FINAL_LEARNING_RATE)}."
    )
    print()

    if start_iteration >= NUM_ITERATIONS:
        print("Training target already reached; nothing to do.")
        self_play_pool.close()
        writer.close()
        return

    if resume and existing_checkpoint_paths:
        model.eval()
        with tqdm(
            total=REPLAY_BUFFER_SIZE,
            desc="Replay fill",
            position=0,
        ) as replay_fill_bar:
            replay_fill_bar.update(len(replay_buffer))
            fill_round = 0
            while len(replay_buffer) < REPLAY_BUFFER_SIZE:
                previous_buffer_size = len(replay_buffer)
                replay_buffer.push_examples(
                    self_play_pool.generate_examples(
                        model=model,
                        total_games=GAMES_PER_ITERATION,
                        num_sampling_moves=NUM_SAMPLING_MOVES,
                        iteration=start_iteration + fill_round,
                        desc="Replay fill self-play",
                        position=1,
                        leave=False,
                    )
                )
                replay_fill_bar.update(len(replay_buffer) - previous_buffer_size)
                fill_round += 1

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
        loss_total_cum = 0.0
        loss_policy_cum = 0.0
        loss_value_cum = 0.0
        lr_cum = 0.0
        replay_buffer_size_cum = 0.0

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
                warmup_fraction=WARMUP_FRACTION,
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

            loss_total_cum += float(loss.item())
            loss_policy_cum += float(policy_loss.item())
            loss_value_cum += float(value_loss.item())
            lr_cum += float(current_lr)
            replay_buffer_size_cum += float(len(replay_buffer))

        writer.add_scalar(
            "t/loss_total",
            float(loss_total_cum / TRAIN_STEPS_PER_ITERATION),
            iteration,
        )
        writer.add_scalar(
            "t/loss_policy",
            float(loss_policy_cum / TRAIN_STEPS_PER_ITERATION),
            iteration,
        )
        writer.add_scalar(
            "t/loss_value",
            float(loss_value_cum / TRAIN_STEPS_PER_ITERATION),
            iteration,
        )
        writer.add_scalar("t/lr", float(lr_cum / TRAIN_STEPS_PER_ITERATION), iteration)
        writer.add_scalar(
            "t/replay_buffer_size",
            float(replay_buffer_size_cum / TRAIN_STEPS_PER_ITERATION),
            iteration,
        )

        # Saving checkpoints
        if iteration % CHECKPOINT_EVERY_ITERATIONS == 0:
            checkpoint_path = CHECKPOINT_DIR / f"iteration_{iteration}.pt"
            torch.save(
                {
                    "iteration": iteration + 1,
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )

            # Evaluating
            if not skip_evaluation:
                checkpoint_winrates = {}
                for evaluation_checkpoint_gap in EVAL_CHECKPOINT_GAPS:
                    if len(saved_checkpoint_paths) < evaluation_checkpoint_gap:
                        continue

                    try:
                        winrate = previous_checkpoint_winrate(
                            saved_checkpoint_paths[-evaluation_checkpoint_gap],
                            checkpoint_path,
                            num_simulations=NUM_SIMULATIONS_TRAINING,
                            max_workers=EVALUATION_WORKERS,
                            c_puct=C_PUCT,
                        )
                    except FightPoolError:
                        continue

                    checkpoint_winrates[f"gap_{evaluation_checkpoint_gap}"] = float(
                        winrate
                    )

                if checkpoint_winrates:
                    writer.add_scalars(
                        "e/checkpoint_winrate",
                        checkpoint_winrates,
                        iteration,
                    )

            saved_checkpoint_paths.append(checkpoint_path)

    final_checkpoint_step = NUM_ITERATIONS
    final_checkpoint_path = CHECKPOINT_DIR / f"iteration_{final_checkpoint_step}.pt"
    torch.save(
        {
            "iteration": NUM_ITERATIONS,
            "model_state_dict": model.state_dict(),
        },
        final_checkpoint_path,
    )

    self_play_pool.close()
    writer.close()


if __name__ == "__main__":
    NUM_ITERATIONS = 1_000
    NUM_WORKERS = 8

    parser = argparse.ArgumentParser(description="Train the Score Four Zero model.")
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=NUM_ITERATIONS,
        help=f"Number of training iterations to run. Default: {NUM_ITERATIONS:,}.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=NUM_WORKERS,
        help=f"Number of CPU self-play workers to use. Also used as the evaluation worker cap. Default: {NUM_WORKERS}.",
    )
    parser.add_argument(
        "--delete-existing-checkpoints",
        action="store_true",
        help="Delete existing iteration_*.pt checkpoints before training starts.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip checkpoint-vs-checkpoint evaluation during training. Default: disabled.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest iteration_*.pt checkpoint, refill the replay buffer with that model, and continue training.",
    )
    args = parser.parse_args()
    train(
        delete_existing_checkpoints=args.delete_existing_checkpoints,
        num_iterations=args.num_iterations,
        workers=args.workers,
        skip_evaluation=args.skip_evaluation,
        resume=args.resume,
    )
