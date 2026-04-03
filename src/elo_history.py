"""
Entirely vibe-coded file (gpt-5). Tiny helper file, but rating snapshot glue is too boring to do manually in 2026 lol
"""

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
