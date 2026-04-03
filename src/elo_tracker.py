"""
Entirely vibe-coded file (gpt-5.4). Dashboard plumbing is too much of a pain to do manually in 2026 for non-critical code lol
"""

import argparse
import copy
import json
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

from .elo_history import build_elo_snapshot, checkpoint_iteration
from .elo_parallel import (
    ParallelFightPool,
    compute_new_elos,
    remap_fight_result,
    sample_matchup,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DASHBOARD_HTML_PATH = PROJECT_ROOT / "dashboard" / "elo_tracker_dashboard.html"


class LiveEloTracker:

    def __init__(
        self,
        checkpoint_dir: Path,
        *,
        num_simulations: int = 25,
        max_workers: int = 6,
        c_puct: float = 1.5,
        elo_k: float = 20.0,
        matchup_distance_scale: float = 200.0,
        matchup_min_weight: float = 0.05,
        primary_target_count: int = 50,
        snapshot_interval_matches: int = 10,
        checkpoint_stable_age_s: float = 2.0,
        idle_sleep_s: float = 1.0,
        max_tasks_per_child: int = 100,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.num_simulations = num_simulations
        self.max_workers = max_workers
        self.c_puct = c_puct
        self.elo_k = elo_k
        self.matchup_distance_scale = matchup_distance_scale
        self.matchup_min_weight = matchup_min_weight
        self.primary_target_count = primary_target_count
        self.snapshot_interval_matches = snapshot_interval_matches
        self.checkpoint_stable_age_s = checkpoint_stable_age_s
        self.idle_sleep_s = idle_sleep_s
        self.max_tasks_per_child = max_tasks_per_child

        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        self.model_paths = []
        self.elos = []
        self.fight_counts = []
        self.primary_selection_counts = []
        self._known_model_paths = set()

        self.total_matches = 0
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.last_update_at = None
        self.snapshot_history = []
        self._next_snapshot_match_count = self.snapshot_interval_matches
        self._snapshot = self._build_snapshot_unlocked()

    def stop(self):
        self._stop_event.set()

    def get_snapshot(self):
        with self._lock:
            return dict(self._snapshot)

    def get_history(self, after: int = -1):
        with self._lock:
            return {
                "snapshots": [
                    copy.deepcopy(snapshot)
                    for snapshot in self.snapshot_history[after + 1 :]
                ],
                "total_snapshots": len(self.snapshot_history),
            }

    def run_forever(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        with ParallelFightPool(
            model_paths=(),
            num_simulations=self.num_simulations,
            max_workers=self.max_workers,
            c_puct=self.c_puct,
            max_tasks_per_child=self.max_tasks_per_child,
        ) as fight_pool:
            while not self._stop_event.is_set():
                new_iterations = self._discover_new_models()
                for iteration, initial_elo in new_iterations:
                    print(
                        f"Tracking checkpoint iteration {iteration} "
                        f"with initial Elo {initial_elo:.1f}"
                    )

                batch = self._sample_batch()
                if not batch:
                    time.sleep(self.idle_sleep_s)
                    continue

                path_matchups = [
                    self._path_matchup_from_sample(sample)
                    for sample in batch
                ]

                try:
                    for matchup_index, result in fight_pool.iter_fight_path_results(
                        path_matchups,
                        desc=None,
                    ):
                        self._apply_single_result(batch[matchup_index], result)
                except Exception as exc:
                    print(f"Match batch failed: {exc}")
                    time.sleep(self.idle_sleep_s)
                    continue

    def _build_snapshot_unlocked(self):
        snapshot = build_elo_snapshot(
            iteration=self.total_matches,
            model_paths=self.model_paths,
            elos=self.elos,
        )

        latest_index = len(self.elos) - 1 if self.elos else None
        mean_elo = float(np.mean(self.elos)) if self.elos else None
        max_elo = float(max(self.elos)) if self.elos else None

        snapshot.update(
            {
                "total_matches": int(self.total_matches),
                "mean_elo": mean_elo,
                "max_elo": max_elo,
                "elo_k": float(self.elo_k),
                "fight_counts": [int(count) for count in self.fight_counts],
                "started_at": self.started_at,
                "last_update_at": self.last_update_at,
                "primary_target_count": int(self.primary_target_count),
                "snapshot_interval_matches": int(self.snapshot_interval_matches),
                "under_sampled_models": int(
                    sum(
                        count < self.primary_target_count
                        for count in self.fight_counts
                    )
                ),
                "min_primary_selections": (
                    int(min(self.primary_selection_counts))
                    if self.primary_selection_counts
                    else None
                ),
                "max_primary_selections": (
                    int(max(self.primary_selection_counts))
                    if self.primary_selection_counts
                    else None
                ),
            }
        )
        return snapshot

    def _checkpoint_paths(self):
        checkpoint_paths = self.checkpoint_dir.glob("iteration_*.pt")
        return sorted(
            checkpoint_paths,
            key=checkpoint_iteration,
        )

    def _initial_elo_for_new_model_unlocked(self):
        return 500.0

    def _discover_new_models(self):
        new_models = []
        current_time = time.time()

        for checkpoint_path in self._checkpoint_paths():
            try:
                checkpoint_stat = checkpoint_path.stat()
            except FileNotFoundError:
                continue

            if current_time - checkpoint_stat.st_mtime < self.checkpoint_stable_age_s:
                continue

            normalized_path = str(checkpoint_path.resolve())
            with self._lock:
                if normalized_path in self._known_model_paths:
                    continue

                initial_elo = self._initial_elo_for_new_model_unlocked()
                self._known_model_paths.add(normalized_path)
                self.model_paths.append(normalized_path)
                self.elos.append(initial_elo)
                self.fight_counts.append(0)
                self.primary_selection_counts.append(0)
                self.last_update_at = datetime.now(timezone.utc).isoformat()
                self._snapshot = self._build_snapshot_unlocked()

            new_models.append((checkpoint_iteration(normalized_path), initial_elo))

        return new_models

    def _sample_primary_index_unlocked(self):
        counts = np.asarray(self.primary_selection_counts, dtype=np.float64)
        weights = 1.0 + np.clip(
            self.primary_target_count - counts,
            a_min=0.0,
            a_max=None,
        )
        probabilities = weights / weights.sum()
        idx = int(np.random.choice(len(counts), p=probabilities))
        self.primary_selection_counts[idx] += 1
        return idx

    def _sample_batch(self):
        with self._lock:
            if len(self.model_paths) < 2:
                return []

            batch = []
            for _ in range(self.max_workers):
                idx1 = self._sample_primary_index_unlocked()
                idx1, idx2 = sample_matchup(
                    self.elos,
                    distance_scale=self.matchup_distance_scale,
                    min_weight=self.matchup_min_weight,
                    forced_idx1=idx1,
                )
                swapped = bool(np.random.rand() < 0.5)
                batch.append((idx1, idx2, swapped))

            return batch

    def _path_matchup_from_sample(self, sample):
        idx1, idx2, swapped = sample

        with self._lock:
            if swapped:
                return self.model_paths[idx2], self.model_paths[idx1]
            return self.model_paths[idx1], self.model_paths[idx2]

    def _apply_single_result(self, sample, result):
        idx1, idx2, swapped = sample

        with self._lock:
            normalized_result = remap_fight_result(result, swapped)
            self.elos[idx1], self.elos[idx2] = compute_new_elos(
                self.elos[idx1],
                self.elos[idx2],
                normalized_result,
                k=self.elo_k,
            )
            self.fight_counts[idx1] += 1
            self.fight_counts[idx2] += 1
            self.total_matches += 1

            self.last_update_at = datetime.now(timezone.utc).isoformat()
            self._snapshot = self._build_snapshot_unlocked()
            while self.total_matches >= self._next_snapshot_match_count:
                self.snapshot_history.append(copy.deepcopy(self._snapshot))
                self._next_snapshot_match_count += self.snapshot_interval_matches


def make_handler(tracker: LiveEloTracker):

    class EloTrackerHandler(BaseHTTPRequestHandler):

        def do_GET(self):
            parsed_url = urlsplit(self.path)

            if parsed_url.path in ("/", "/index.html"):
                self._send_html(DASHBOARD_HTML_PATH.read_text(encoding="utf-8"))
                return

            if parsed_url.path == "/api/elo-state":
                self._send_json(tracker.get_snapshot())
                return

            if parsed_url.path == "/api/elo-history":
                query = parse_qs(parsed_url.query)
                after = int(query.get("after", ["-1"])[0])
                self._send_json(tracker.get_history(after=after))
                return

            if parsed_url.path == "/health":
                snapshot = tracker.get_snapshot()
                self._send_json(
                    {
                        "status": "ok",
                        "tracked_models": snapshot["num_models"],
                        "total_matches": snapshot["total_matches"],
                    }
                )
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def log_message(self, format, *args):
            return

        def _send_html(self, html: str):
            body = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return EloTrackerHandler


def main():
    parser = argparse.ArgumentParser(description="Track checkpoint Elo live.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--num-simulations", type=int, default=25)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--elo-k", type=float, default=20.0)
    parser.add_argument("--matchup-distance-scale", type=float, default=200.0)
    parser.add_argument("--matchup-min-weight", type=float, default=0.05)
    parser.add_argument("--primary-target-count", type=int, default=50)
    parser.add_argument("--snapshot-interval-matches", type=int, default=10)
    parser.add_argument("--checkpoint-stable-age-s", type=float, default=2.0)
    parser.add_argument("--idle-sleep-s", type=float, default=1.0)
    args = parser.parse_args()

    if args.snapshot_interval_matches <= 0:
        parser.error("--snapshot-interval-matches must be positive")

    tracker = LiveEloTracker(
        checkpoint_dir=args.checkpoint_dir.resolve(),
        num_simulations=args.num_simulations,
        max_workers=args.workers,
        c_puct=args.c_puct,
        elo_k=args.elo_k,
        matchup_distance_scale=args.matchup_distance_scale,
        matchup_min_weight=args.matchup_min_weight,
        primary_target_count=args.primary_target_count,
        snapshot_interval_matches=args.snapshot_interval_matches,
        checkpoint_stable_age_s=args.checkpoint_stable_age_s,
        idle_sleep_s=args.idle_sleep_s,
    )
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(tracker),
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print(f"Serving live Elo tracker at http://{args.host}:{args.port}")
    print(f"Watching checkpoints in {args.checkpoint_dir.resolve()}")

    try:
        tracker.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
