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
WORKER_JS_PATH = PROJECT_ROOT / "dashboard" / "elo_chart_worker.js"
MIN_WORKERS = 1
MAX_WORKERS = 24


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
        snapshot_interval_matches: int = 10,
        checkpoint_stable_age_s: float = 2.0,
        idle_sleep_s: float = 1.0,
        max_tasks_per_child: int = 100,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.num_simulations = num_simulations
        self.requested_max_workers = self._normalize_max_workers(max_workers)
        self.active_max_workers = self.requested_max_workers
        self.c_puct = c_puct
        self.elo_k = elo_k
        self.matchup_distance_scale = matchup_distance_scale
        self.matchup_min_weight = matchup_min_weight
        self.snapshot_interval_matches = snapshot_interval_matches
        self.checkpoint_stable_age_s = checkpoint_stable_age_s
        self.idle_sleep_s = idle_sleep_s
        self.max_tasks_per_child = max_tasks_per_child

        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        self.model_paths = []
        self.elos = []
        self.fight_counts = []
        self._known_model_paths = set()

        self.total_matches = 0
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.last_update_at = None
        self.snapshot_history = []
        self._next_snapshot_match_count = self.snapshot_interval_matches
        self._chart_version = 0
        self._chart_snapshot = self._build_chart_snapshot_unlocked()
        self._summary_snapshot = self._build_summary_snapshot_unlocked(
            self._chart_snapshot
        )

    @staticmethod
    def _normalize_max_workers(max_workers: int) -> int:
        normalized_workers = int(max_workers)
        if not (MIN_WORKERS <= normalized_workers <= MAX_WORKERS):
            raise ValueError(
                f"max_workers must be between {MIN_WORKERS} and {MAX_WORKERS}"
            )
        return normalized_workers

    def stop(self):
        self._stop_event.set()

    def get_snapshot(self):
        with self._lock:
            return dict(self._summary_snapshot)

    def get_chart_snapshot(self):
        with self._lock:
            return dict(self._chart_snapshot)

    def get_history(self, after: int = -1):
        with self._lock:
            return {
                "snapshots": [
                    copy.deepcopy(snapshot)
                    for snapshot in self.snapshot_history[after + 1 :]
                ],
                "total_snapshots": len(self.snapshot_history),
            }

    def set_max_workers(self, max_workers: int):
        normalized_workers = self._normalize_max_workers(max_workers)

        with self._lock:
            if normalized_workers == self.requested_max_workers:
                return dict(self._summary_snapshot)

            self.requested_max_workers = normalized_workers
            self._refresh_metadata_unlocked()
            return dict(self._summary_snapshot)

    def run_forever(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        fight_pool = None

        try:
            while not self._stop_event.is_set():
                self._discover_new_models()
                desired_workers = self._get_requested_max_workers()

                if fight_pool is None or desired_workers != self.active_max_workers:
                    if fight_pool is not None:
                        fight_pool.close()

                    fight_pool = ParallelFightPool(
                        model_paths=(),
                        num_simulations=self.num_simulations,
                        max_workers=desired_workers,
                        c_puct=self.c_puct,
                        max_tasks_per_child=self.max_tasks_per_child,
                    )
                    fight_pool.open()
                    self._set_active_max_workers(desired_workers)

                batch = self._sample_batch(batch_size=desired_workers)
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
        finally:
            if fight_pool is not None:
                fight_pool.close()

    def _build_chart_snapshot_unlocked(self):
        snapshot = build_elo_snapshot(
            iteration=self.total_matches,
            model_paths=self.model_paths,
            elos=self.elos,
        )

        mean_elo = float(np.mean(self.elos)) if self.elos else None
        max_elo = float(max(self.elos)) if self.elos else None
        fight_counts_array = np.asarray(self.fight_counts, dtype=np.float64)
        fight_count_mean = (
            float(np.mean(fight_counts_array))
            if fight_counts_array.size
            else None
        )
        fight_count_p01 = (
            float(np.percentile(fight_counts_array, 1))
            if fight_counts_array.size
            else None
        )
        fight_count_median = (
            float(np.median(fight_counts_array))
            if fight_counts_array.size
            else None
        )
        fight_count_p25 = (
            float(np.percentile(fight_counts_array, 25))
            if fight_counts_array.size
            else None
        )
        fight_count_p75 = (
            float(np.percentile(fight_counts_array, 75))
            if fight_counts_array.size
            else None
        )
        fight_count_max = (
            float(np.max(fight_counts_array))
            if fight_counts_array.size
            else None
        )

        snapshot.update(
            {
                "total_matches": int(self.total_matches),
                "mean_elo": mean_elo,
                "max_elo": max_elo,
                "elo_k": float(self.elo_k),
                "requested_max_workers": int(self.requested_max_workers),
                "active_max_workers": int(self.active_max_workers),
                "fight_counts": [int(count) for count in self.fight_counts],
                "fight_count_mean": fight_count_mean,
                "fight_count_p01": fight_count_p01,
                "fight_count_median": fight_count_median,
                "fight_count_p25": fight_count_p25,
                "fight_count_p75": fight_count_p75,
                "fight_count_max": fight_count_max,
                "started_at": self.started_at,
                "last_update_at": self.last_update_at,
                "snapshot_interval_matches": int(self.snapshot_interval_matches),
                "chart_version": int(self._chart_version),
            }
        )
        return snapshot

    def _build_summary_snapshot_unlocked(self, chart_snapshot):
        return {
            key: value
            for key, value in chart_snapshot.items()
            if key not in {"model_iterations", "elos", "fight_counts"}
        }

    def _refresh_snapshots_unlocked(self):
        self._chart_version += 1
        self._refresh_metadata_unlocked()

    def _refresh_metadata_unlocked(self):
        self._chart_snapshot = self._build_chart_snapshot_unlocked()
        self._summary_snapshot = self._build_summary_snapshot_unlocked(
            self._chart_snapshot
        )

    def _get_requested_max_workers(self):
        with self._lock:
            return int(self.requested_max_workers)

    def _set_active_max_workers(self, max_workers: int):
        with self._lock:
            normalized_workers = self._normalize_max_workers(max_workers)
            if normalized_workers == self.active_max_workers:
                return

            self.active_max_workers = normalized_workers
            self._refresh_metadata_unlocked()

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
                self.last_update_at = datetime.now(timezone.utc).isoformat()
                self._refresh_snapshots_unlocked()

            new_models.append((checkpoint_iteration(normalized_path), initial_elo))

        return new_models

    def _sample_batch(self, batch_size: int):
        with self._lock:
            if len(self.model_paths) < 2:
                return []

            batch = []
            for _ in range(batch_size):
                idx1, idx2 = sample_matchup(
                    self.elos,
                    distance_scale=self.matchup_distance_scale,
                    min_weight=self.matchup_min_weight,
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
            self._refresh_snapshots_unlocked()
            while self.total_matches >= self._next_snapshot_match_count:
                self.snapshot_history.append(copy.deepcopy(self._chart_snapshot))
                self._next_snapshot_match_count += self.snapshot_interval_matches


def make_handler(tracker: LiveEloTracker):

    class EloTrackerHandler(BaseHTTPRequestHandler):

        def do_GET(self):
            parsed_url = urlsplit(self.path)

            if parsed_url.path in ("/", "/index.html"):
                self._send_html(DASHBOARD_HTML_PATH.read_text(encoding="utf-8"))
                return

            if parsed_url.path == "/elo-chart-worker.js":
                self._send_javascript(WORKER_JS_PATH.read_text(encoding="utf-8"))
                return

            if parsed_url.path == "/api/elo-state":
                self._send_json(tracker.get_snapshot())
                return

            if parsed_url.path == "/api/elo-live-chart":
                self._send_json(tracker.get_chart_snapshot())
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

        def do_POST(self):
            parsed_url = urlsplit(self.path)

            if parsed_url.path == "/api/max-workers":
                try:
                    payload = self._read_json()
                    snapshot = tracker.set_max_workers(payload["max_workers"])
                except (KeyError, TypeError, json.JSONDecodeError):
                    self._send_json(
                        {"error": "Request body must be valid JSON with max_workers."},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                except ValueError as exc:
                    self._send_json(
                        {"error": str(exc)},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return

                self._send_json(snapshot)
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def log_message(self, format, *args):
            return

        def _read_json(self):
            content_length = int(self.headers.get("Content-Length", "0"))
            request_body = self.rfile.read(content_length)
            if not request_body:
                raise json.JSONDecodeError("Empty body", "", 0)
            return json.loads(request_body.decode("utf-8"))

        def _send_html(self, html: str):
            body = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload, *, status: HTTPStatus = HTTPStatus.OK):
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_javascript(self, source: str):
            body = source.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/javascript; charset=utf-8")
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
    parser.add_argument("--snapshot-interval-matches", type=int, default=10)
    parser.add_argument("--checkpoint-stable-age-s", type=float, default=2.0)
    parser.add_argument("--idle-sleep-s", type=float, default=1.0)
    args = parser.parse_args()

    if args.snapshot_interval_matches <= 0:
        parser.error("--snapshot-interval-matches must be positive")
    if not (MIN_WORKERS <= args.workers <= MAX_WORKERS):
        parser.error(
            f"--workers must be between {MIN_WORKERS} and {MAX_WORKERS}"
        )

    tracker = LiveEloTracker(
        checkpoint_dir=args.checkpoint_dir.resolve(),
        num_simulations=args.num_simulations,
        max_workers=args.workers,
        c_puct=args.c_puct,
        elo_k=args.elo_k,
        matchup_distance_scale=args.matchup_distance_scale,
        matchup_min_weight=args.matchup_min_weight,
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
