import json
from json import JSONDecodeError
from datetime import datetime, timezone
from pathlib import Path


def checkpoint_iteration(checkpoint_path) -> int:
    checkpoint_path = Path(checkpoint_path)
    try:
        return int(checkpoint_path.stem.split("_")[-1])
    except ValueError:
        return -1


def build_elo_snapshot(iteration: int, model_paths, elos):
    model_iterations = [checkpoint_iteration(path) for path in model_paths]

    if elos:
        latest_index = len(elos) - 1
        best_index = max(range(len(elos)), key=lambda idx: elos[idx])
        latest_model_iteration = model_iterations[latest_index]
        latest_model_elo = float(elos[latest_index])
        best_model_iteration = model_iterations[best_index]
        best_elo = float(elos[best_index])
    else:
        latest_model_iteration = None
        latest_model_elo = None
        best_model_iteration = None
        best_elo = None

    return {
        "iteration": int(iteration),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_models": len(model_iterations),
        "model_iterations": model_iterations,
        "elos": [float(elo) for elo in elos],
        "latest_model_iteration": latest_model_iteration,
        "latest_model_elo": latest_model_elo,
        "best_model_iteration": best_model_iteration,
        "best_elo": best_elo,
    }


def append_elo_snapshot(log_path, iteration: int, model_paths, elos):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot = build_elo_snapshot(iteration, model_paths, elos)
    with log_path.open("a", encoding="utf-8") as log_file:
        json.dump(snapshot, log_file)
        log_file.write("\n")

    return snapshot


def reset_elo_history(log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")


def load_elo_snapshots(log_path):
    log_path = Path(log_path)
    if not log_path.exists():
        return []

    snapshots = []
    with log_path.open("r", encoding="utf-8") as log_file:
        for line in log_file:
            line = line.strip()
            if not line:
                continue
            try:
                snapshots.append(json.loads(line))
            except JSONDecodeError:
                continue

    return snapshots
