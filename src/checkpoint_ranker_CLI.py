"""
Entirely vibe-coded file (gpt-5.4 xhigh reasoning).
"""

import argparse
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
DEFAULT_RESUME_STATE_PATH = PROJECT_ROOT / "artifacts" / "elo_tracker_resume_state.json"
DEFAULT_ANCHOR_DIR = PROJECT_ROOT / "anchors" / "legacy_pv_model"
DEFAULT_ANCHOR_MANIFEST_PATH = DEFAULT_ANCHOR_DIR / "anchors.json"
DASHBOARD_HTML_PATH = PROJECT_ROOT / "dashboard" / "checkpoint_ranker_dashboard.html"
WORKER_JS_PATH = PROJECT_ROOT / "dashboard" / "elo_chart_worker.js"
MIN_WORKERS = 1
MAX_WORKERS = 24
TRACKER_STATE_SCHEMA_VERSION = 1
DEFAULT_SNAPSHOT_MIN_INTERVAL_S = 0.5
DEFAULT_LIVE_REFRESH_INTERVAL_S = 0.1
DEFAULT_FIGHT_TASK_BATCH_SIZE = 8
DEFAULT_ANCHOR_FIGHT_PROBABILITY = 0.01


def load_legacy_anchors():
    try:
        manifest = json.loads(DEFAULT_ANCHOR_MANIFEST_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Legacy anchor manifest not found at {DEFAULT_ANCHOR_MANIFEST_PATH}"
        ) from exc

    anchor_entries = manifest.get("anchors")
    if not isinstance(anchor_entries, list):
        raise ValueError("Legacy anchor manifest is malformed.")

    anchors = []
    for anchor_entry in anchor_entries:
        checkpoint_name = anchor_entry["checkpoint"]
        checkpoint_path = (DEFAULT_ANCHOR_DIR / checkpoint_name).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Legacy anchor checkpoint not found: {checkpoint_path}")

        anchors.append(
            {
                "name": Path(checkpoint_name).stem,
                "path": str(checkpoint_path),
                "iteration": int(anchor_entry["iteration"]),
                "elo": float(anchor_entry["elo"]),
            }
        )

    return anchors


class LiveCheckpointRanker:

    def __init__(
        self,
        checkpoint_dir: Path,
        *,
        resume_state_path: Path = DEFAULT_RESUME_STATE_PATH,
        persist_resume_state: bool = False,
        num_simulations: int = 25,
        max_workers: int = 6,
        c_puct: float = 1.5,
        elo_k: float = 10.0,
        matchup_distance_scale: float = 200.0,
        matchup_min_weight: float = 0.05,
        snapshot_interval_matches: int = 10,
        snapshot_min_interval_s: float = DEFAULT_SNAPSHOT_MIN_INTERVAL_S,
        live_refresh_interval_s: float = DEFAULT_LIVE_REFRESH_INTERVAL_S,
        checkpoint_stable_age_s: float = 2.0,
        idle_sleep_s: float = 1.0,
        max_tasks_per_child: int = 100,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.resume_state_path = resume_state_path
        self.persist_resume_state = persist_resume_state
        self.num_simulations = num_simulations
        self.requested_max_workers = self._normalize_max_workers(max_workers)
        self.active_max_workers = self.requested_max_workers
        self.c_puct = c_puct
        self.elo_k = elo_k
        self.matchup_distance_scale = matchup_distance_scale
        self.matchup_min_weight = matchup_min_weight
        self.snapshot_interval_matches = snapshot_interval_matches
        self.snapshot_min_interval_s = snapshot_min_interval_s
        self.live_refresh_interval_s = live_refresh_interval_s
        self.checkpoint_stable_age_s = checkpoint_stable_age_s
        self.idle_sleep_s = idle_sleep_s
        self.max_tasks_per_child = max_tasks_per_child

        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        self.anchors = load_legacy_anchors()
        self.anchor_names = [anchor["name"] for anchor in self.anchors]
        self.anchor_paths = [anchor["path"] for anchor in self.anchors]
        self.anchor_iterations = [anchor["iteration"] for anchor in self.anchors]
        self.anchor_elos = [anchor["elo"] for anchor in self.anchors]
        self.anchor_fight_probability = DEFAULT_ANCHOR_FIGHT_PROBABILITY
        self.anchor_match_results_total = 0

        self.model_paths = []
        self.elos = []
        self.fight_counts = []
        self.recent_avg_elo_last_100 = []
        self.recent_avg_elo_warmup_counts = []
        self._known_model_paths = set()

        self.total_matches = 0
        self.cached_match_results_total = 0
        self.uncached_match_results_total = 0
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.last_update_at = None
        self.snapshot_history = []
        self._next_snapshot_match_count = self.snapshot_interval_matches
        self._next_snapshot_time = time.perf_counter()
        self._next_live_refresh_time = self._next_snapshot_time
        self._chart_version = 0
        self.fight_task_batch_size = DEFAULT_FIGHT_TASK_BATCH_SIZE
        self._chart_snapshot = self._build_chart_snapshot_unlocked()
        self._summary_snapshot = self._build_summary_snapshot_unlocked(
            self._chart_snapshot
        )

    def _resume_settings_snapshot(self):
        return {
            "checkpoint_dir": str(self.checkpoint_dir),
            "num_simulations": int(self.num_simulations),
            "c_puct": float(self.c_puct),
            "elo_k": float(self.elo_k),
            "matchup_distance_scale": float(self.matchup_distance_scale),
            "matchup_min_weight": float(self.matchup_min_weight),
            "snapshot_interval_matches": int(self.snapshot_interval_matches),
            "anchors": [
                {
                    "path": path,
                    "iteration": int(iteration),
                    "elo": float(elo),
                }
                for path, iteration, elo in zip(
                    self.anchor_paths,
                    self.anchor_iterations,
                    self.anchor_elos,
                )
            ],
            "anchor_fight_probability": float(self.anchor_fight_probability),
        }

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

    def delete_resume_state(self):
        try:
            self.resume_state_path.unlink()
        except FileNotFoundError:
            return

    def try_resume_previous_run(self):
        try:
            state = json.loads(self.resume_state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return False, "no saved checkpoint ranker state"
        except (OSError, json.JSONDecodeError):
            self.delete_resume_state()
            return False, "saved checkpoint ranker state could not be read"

        is_valid, reason, normalized_model_paths = self._validate_resume_state(state)
        if not is_valid:
            self.delete_resume_state()
            return False, reason

        with self._lock:
            self.model_paths = normalized_model_paths
            self.elos = [float(elo) for elo in state["elos"]]
            self.fight_counts = [int(count) for count in state["fight_counts"]]
            self.recent_avg_elo_last_100 = [float(elo) for elo in self.elos]
            self.recent_avg_elo_warmup_counts = [0 for _ in self.model_paths]
            self._known_model_paths = set(self.model_paths)
            self.total_matches = int(state["total_matches"])
            self.cached_match_results_total = int(
                state.get("cached_match_results_total", 0)
            )
            self.uncached_match_results_total = int(
                state.get("uncached_match_results_total", 0)
            )
            self.anchor_match_results_total = int(
                state.get("anchor_match_results_total", 0)
            )
            self.started_at = state.get("started_at") or self.started_at
            self.last_update_at = state.get("last_update_at")
            self.snapshot_history = list(state.get("snapshot_history", []))
            self._chart_version = int(state.get("chart_version", 0))

            next_snapshot_match_count = int(
                state.get(
                    "next_snapshot_match_count",
                    self.snapshot_interval_matches,
                )
            )
            while next_snapshot_match_count <= self.total_matches:
                next_snapshot_match_count += self.snapshot_interval_matches
            self._next_snapshot_match_count = next_snapshot_match_count
            self._refresh_metadata_unlocked()

        return True, "resumed previous checkpoint ranker state"

    def _validate_resume_state(self, state):
        if not isinstance(state, dict):
            return False, "saved checkpoint ranker state is malformed", []

        if state.get("schema_version") != TRACKER_STATE_SCHEMA_VERSION:
            return False, "saved checkpoint ranker state schema changed", []

        if state.get("settings") != self._resume_settings_snapshot():
            return False, "checkpoint ranker settings changed since the saved run", []

        model_paths = state.get("model_paths")
        elos = state.get("elos")
        fight_counts = state.get("fight_counts")
        snapshot_history = state.get("snapshot_history", [])

        if not isinstance(model_paths, list):
            return False, "saved checkpoint ranker state has invalid model paths", []
        if not isinstance(elos, list) or not isinstance(fight_counts, list):
            return False, "saved checkpoint ranker state has invalid rating data", []
        if len(model_paths) != len(elos) or len(model_paths) != len(fight_counts):
            return False, "saved checkpoint ranker state has inconsistent array lengths", []
        if not isinstance(snapshot_history, list):
            return False, "saved checkpoint ranker history is malformed", []

        normalized_model_paths = [str(Path(path).resolve()) for path in model_paths]
        current_paths = [str(path.resolve()) for path in self._checkpoint_paths()]

        if current_paths[: len(normalized_model_paths)] != normalized_model_paths:
            return False, "checkpoint set changed since the saved run", []

        return True, None, normalized_model_paths

    def _build_persisted_state_unlocked(self):
        return {
            "schema_version": TRACKER_STATE_SCHEMA_VERSION,
            "settings": self._resume_settings_snapshot(),
            "started_at": self.started_at,
            "last_update_at": self.last_update_at,
            "total_matches": int(self.total_matches),
            "cached_match_results_total": int(self.cached_match_results_total),
            "uncached_match_results_total": int(self.uncached_match_results_total),
            "anchor_match_results_total": int(self.anchor_match_results_total),
            "next_snapshot_match_count": int(self._next_snapshot_match_count),
            "chart_version": int(self._chart_version),
            "model_paths": list(self.model_paths),
            "elos": [float(elo) for elo in self.elos],
            "fight_counts": [int(count) for count in self.fight_counts],
            "snapshot_history": list(self.snapshot_history),
        }

    def _persist_resume_state(self):
        if not self.persist_resume_state:
            return

        with self._lock:
            state = self._build_persisted_state_unlocked()

        self.resume_state_path.parent.mkdir(parents=True, exist_ok=True)
        temp_state_path = self.resume_state_path.with_suffix(
            self.resume_state_path.suffix + ".tmp"
        )
        temp_state_path.write_text(
            json.dumps(state, separators=(",", ":")),
            encoding="utf-8",
        )
        temp_state_path.replace(self.resume_state_path)

    def get_snapshot(self):
        return dict(self._summary_snapshot)

    def get_chart_snapshot(self):
        return dict(self._chart_snapshot)

    def get_history(self, after: int = -1):
        with self._lock:
            return {
                "snapshots": list(self.snapshot_history[after + 1 :]),
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
                        task_batch_size=self.fight_task_batch_size,
                    )
                    fight_pool.open()
                    self._set_active_max_workers(desired_workers)

                batch = self._sample_batch(
                    batch_size=desired_workers * self.fight_task_batch_size
                )
                if not batch:
                    time.sleep(self.idle_sleep_s)
                    continue

                path_matchups = [
                    self._path_matchup_from_sample(sample)
                    for sample in batch
                ]

                try:
                    batch_results = []
                    for matchup_index, result, from_cache in fight_pool.iter_fight_path_results(
                        path_matchups,
                        desc=None,
                        include_cache_status=True,
                    ):
                        batch_results.append((batch[matchup_index], result, from_cache))
                    self._apply_batch_results(batch_results)
                except Exception as exc:
                    print(f"Match batch failed: {exc}")
                    time.sleep(self.idle_sleep_s)
                    continue
        finally:
            self._persist_resume_state()
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
                "cached_match_results_total": int(self.cached_match_results_total),
                "uncached_match_results_total": int(self.uncached_match_results_total),
                "anchor_match_results_total": int(self.anchor_match_results_total),
                "mean_elo": mean_elo,
                "max_elo": max_elo,
                "elo_k": float(self.elo_k),
                "requested_max_workers": int(self.requested_max_workers),
                "active_max_workers": int(self.active_max_workers),
                "fight_counts": [int(count) for count in self.fight_counts],
                "recent_avg_elo_last_100": list(self.recent_avg_elo_last_100),
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
                "num_anchors": len(self.anchor_paths),
                "anchor_fight_probability": float(self.anchor_fight_probability),
                "anchor_names": list(self.anchor_names),
                "anchor_iterations": [int(iteration) for iteration in self.anchor_iterations],
                "anchor_elos": [float(elo) for elo in self.anchor_elos],
            }
        )
        return snapshot

    def _build_summary_snapshot_unlocked(self, chart_snapshot):
        return {
            key: value
            for key, value in chart_snapshot.items()
            if key
            not in {
                "model_iterations",
                "elos",
                "fight_counts",
                "recent_avg_elo_last_100",
                "anchor_names",
                "anchor_iterations",
                "anchor_elos",
            }
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
        should_persist = False

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
                self.recent_avg_elo_last_100.append(float(initial_elo))
                self.recent_avg_elo_warmup_counts.append(0)
                self.last_update_at = datetime.now(timezone.utc).isoformat()
                self._refresh_snapshots_unlocked()
                should_persist = True

            new_models.append((checkpoint_iteration(normalized_path), initial_elo))

        if should_persist:
            self._persist_resume_state()

        return new_models

    def _sample_batch(self, batch_size: int):
        with self._lock:
            if len(self.model_paths) < 2:
                return []

            batch = []
            for _ in range(batch_size):
                idx1 = int(np.random.randint(len(self.model_paths)))
                swapped = bool(np.random.rand() < 0.5)
                if self.anchor_paths and np.random.rand() < self.anchor_fight_probability:
                    anchor_idx = self._closest_anchor_index_unlocked(idx1)
                    batch.append(("anchor", idx1, anchor_idx, swapped))
                    continue

                _, idx2 = sample_matchup(
                    self.elos,
                    distance_scale=self.matchup_distance_scale,
                    min_weight=self.matchup_min_weight,
                    forced_idx1=idx1,
                )
                batch.append(("model", idx1, idx2, swapped))

            return batch

    def _closest_anchor_index_unlocked(self, model_index: int):
        current_elo = float(self.elos[model_index])
        closest_anchor_index = min(
            range(len(self.anchor_elos)),
            key=lambda anchor_index: abs(self.anchor_elos[anchor_index] - current_elo),
        )
        return int(closest_anchor_index)

    def _path_matchup_from_sample(self, sample):
        sample_kind, idx1, idx2, swapped = sample

        with self._lock:
            if sample_kind == "anchor":
                if swapped:
                    return self.anchor_paths[idx2], self.model_paths[idx1]
                return self.model_paths[idx1], self.anchor_paths[idx2]

            if swapped:
                return self.model_paths[idx2], self.model_paths[idx1]
            return self.model_paths[idx1], self.model_paths[idx2]

    def _apply_batch_results(self, batch_results):
        if not batch_results:
            return

        should_persist = False
        now = time.perf_counter()

        with self._lock:
            for sample, result, from_cache in batch_results:
                sample_kind, idx1, idx2, swapped = sample
                normalized_result = remap_fight_result(result, swapped)
                if sample_kind == "anchor":
                    self.elos[idx1] = compute_new_elos(
                        self.elos[idx1],
                        self.anchor_elos[idx2],
                        normalized_result,
                        k=self.elo_k,
                    )[0]
                    self.fight_counts[idx1] += 1
                    self.anchor_match_results_total += 1
                else:
                    self.elos[idx1], self.elos[idx2] = compute_new_elos(
                        self.elos[idx1],
                        self.elos[idx2],
                        normalized_result,
                        k=self.elo_k,
                    )
                    self.fight_counts[idx1] += 1
                    self.fight_counts[idx2] += 1
                self.total_matches += 1
                if from_cache:
                    self.cached_match_results_total += 1
                else:
                    self.uncached_match_results_total += 1
                self._record_recent_elo_unlocked(idx1)
                if sample_kind != "anchor":
                    self._record_recent_elo_unlocked(idx2)

            self.last_update_at = datetime.now(timezone.utc).isoformat()
            should_take_replay_snapshot = (
                self.total_matches >= self._next_snapshot_match_count
                and now >= self._next_snapshot_time
            )
            should_refresh_live_snapshot = now >= self._next_live_refresh_time

            if should_refresh_live_snapshot or should_take_replay_snapshot:
                self._refresh_snapshots_unlocked()
                self._next_live_refresh_time = now + self.live_refresh_interval_s

            if should_take_replay_snapshot:
                self.snapshot_history.append(
                    {
                        key: value
                        for key, value in self._chart_snapshot.items()
                        if key != "recent_avg_elo_last_100"
                    }
                )
                while self._next_snapshot_match_count <= self.total_matches:
                    self._next_snapshot_match_count += self.snapshot_interval_matches
                self._next_snapshot_time = now + self.snapshot_min_interval_s
                should_persist = True

        if should_persist:
            self._persist_resume_state()

    def _record_recent_elo_unlocked(self, model_index: int):
        current_elo = float(self.elos[model_index])
        warmup_count = self.recent_avg_elo_warmup_counts[model_index]

        if warmup_count < 100:
            next_count = warmup_count + 1
            previous_average = self.recent_avg_elo_last_100[model_index]
            self.recent_avg_elo_last_100[model_index] = (
                previous_average * warmup_count + current_elo
            ) / next_count
            self.recent_avg_elo_warmup_counts[model_index] = next_count
            return

        self.recent_avg_elo_last_100[model_index] = (
            self.recent_avg_elo_last_100[model_index] * 0.99
            + current_elo * 0.01
        )


def make_handler(ranker: LiveCheckpointRanker):

    class CheckpointRankerHandler(BaseHTTPRequestHandler):

        def do_GET(self):
            parsed_url = urlsplit(self.path)

            if parsed_url.path in ("/", "/index.html"):
                self._send_html(DASHBOARD_HTML_PATH.read_text(encoding="utf-8"))
                return

            if parsed_url.path == "/elo-chart-worker.js":
                self._send_javascript(WORKER_JS_PATH.read_text(encoding="utf-8"))
                return

            if parsed_url.path == "/api/elo-state":
                self._send_json(ranker.get_snapshot())
                return

            if parsed_url.path == "/api/elo-live-chart":
                self._send_json(ranker.get_chart_snapshot())
                return

            if parsed_url.path == "/api/elo-history":
                query = parse_qs(parsed_url.query)
                after = int(query.get("after", ["-1"])[0])
                self._send_json(ranker.get_history(after=after))
                return

            if parsed_url.path == "/health":
                snapshot = ranker.get_snapshot()
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
                    snapshot = ranker.set_max_workers(payload["max_workers"])
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

    return CheckpointRankerHandler


def main():
    parser = argparse.ArgumentParser(description="Rank checkpoints live with Elo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--num-simulations", type=int, default=25)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--elo-k", type=float, default=10.0)
    parser.add_argument("--matchup-distance-scale", type=float, default=200.0)
    parser.add_argument("--matchup-min-weight", type=float, default=0.05)
    parser.add_argument("--snapshot-interval-matches", type=int, default=10)
    parser.add_argument("--checkpoint-stable-age-s", type=float, default=2.0)
    parser.add_argument("--idle-sleep-s", type=float, default=1.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the previous checkpoint ranker run if the saved state still matches the checkpoint set and settings.",
    )
    args = parser.parse_args()

    if args.snapshot_interval_matches <= 0:
        parser.error("--snapshot-interval-matches must be positive")
    if not (MIN_WORKERS <= args.workers <= MAX_WORKERS):
        parser.error(
            f"--workers must be between {MIN_WORKERS} and {MAX_WORKERS}"
        )

    ranker = LiveCheckpointRanker(
        checkpoint_dir=args.checkpoint_dir.resolve(),
        resume_state_path=DEFAULT_RESUME_STATE_PATH,
        persist_resume_state=args.resume,
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

    if not args.resume:
        ranker.delete_resume_state()
    else:
        resumed, resume_message = ranker.try_resume_previous_run()
        if resumed:
            print("Resumed previous live checkpoint ranker state.")
        else:
            print(f"Starting fresh checkpoint ranker state: {resume_message}.")

    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(ranker),
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print(f"Serving live checkpoint ranker at http://{args.host}:{args.port}")
    print(f"Watching checkpoints in {args.checkpoint_dir.resolve()}")

    try:
        ranker.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        ranker.stop()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
