"""
Entirely vibe-coded file (gpt-5.4 xhigh reasoning).
"""

from concurrent.futures.process import BrokenProcessPool
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .Env import Env
from .MCTS import MCTS
from .models import LegacyPVModel, PVModel

_WORKER_NUM_SIMULATIONS = None
_WORKER_C_PUCT = None
_WORKER_TEMPERATURE = None
_LEGACY_ANCHOR_DIR = (Path(__file__).resolve().parent.parent / "anchors" / "legacy_pv_model").resolve()


class FightPoolError(RuntimeError):
    pass


def _init_fight_worker(num_simulations: int, c_puct: float, temperature: float):
    global _WORKER_NUM_SIMULATIONS, _WORKER_C_PUCT, _WORKER_TEMPERATURE

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    _WORKER_NUM_SIMULATIONS = num_simulations
    _WORKER_C_PUCT = c_puct
    _WORKER_TEMPERATURE = temperature


def _load_model(model_path: str):
    checkpoint = torch.load(model_path, map_location="cpu")
    resolved_model_path = Path(model_path).resolve()
    if resolved_model_path.parent == _LEGACY_ANCHOR_DIR:
        model = LegacyPVModel()
    else:
        model = PVModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _fight_worker(model_path_1: str, model_path_2: str):
    model1 = _load_model(model_path_1)
    model2 = _load_model(model_path_2)

    env = Env()
    mcts1 = MCTS(
        model1,
        num_simulations=_WORKER_NUM_SIMULATIONS,
        c_puct=_WORKER_C_PUCT,
        add_exploration_noise=False,
    )
    mcts2 = MCTS(
        model2,
        num_simulations=_WORKER_NUM_SIMULATIONS,
        c_puct=_WORKER_C_PUCT,
        add_exploration_noise=False,
    )
    root1 = None
    root2 = None

    while not env.is_terminal():
        if env.current_player() == -1:
            root1 = mcts1.run(root_env=env, root=root1)
            pi = mcts1.visit_counts_to_policy(root=root1, temperature=_WORKER_TEMPERATURE)
            if _WORKER_TEMPERATURE > 0.0:
                action = int(np.random.choice(len(pi), p=pi))
            else:
                action = int(pi.argmax())
        else:
            root2 = mcts2.run(root_env=env, root=root2)
            pi = mcts2.visit_counts_to_policy(root=root2, temperature=_WORKER_TEMPERATURE)
            if _WORKER_TEMPERATURE > 0.0:
                action = int(np.random.choice(len(pi), p=pi))
            else:
                action = int(pi.argmax())

        env.step(action)
        if root1 is not None:
            root1 = root1.children.get(action)
            if root1 is not None:
                root1.prior = 1.0
        if root2 is not None:
            root2 = root2.children.get(action)
            if root2 is not None:
                root2.prior = 1.0

    return env.winner()


def _fight_worker_batch(path_matchups):
    return [_fight_worker(model_path_1, model_path_2) for model_path_1, model_path_2 in path_matchups]


def compute_new_elos(elo1, elo2, result, k=20.0):
    expected1 = 1.0 / (1.0 + 10.0 ** ((elo2 - elo1) / 400.0))
    expected2 = 1.0 - expected1

    if result == -1:
        score1 = 1.0
    elif result == 0:
        score1 = 0.5
    else:
        score1 = 0.0

    score2 = 1.0 - score1

    new_elo1 = elo1 + k * (score1 - expected1)
    new_elo2 = elo2 + k * (score2 - expected2)
    return new_elo1, new_elo2


def remap_fight_result(result, swapped: bool):
    if not swapped or result == 0:
        return result

    return -result


def sample_matchup(
    elos,
    distance_scale: float = 200.0,
    min_weight: float = 0.05,
    forced_idx1: int | None = None,
):
    if len(elos) < 2:
        raise ValueError("At least two models are required to sample a matchup.")

    if forced_idx1 is None:
        idx1 = int(np.random.randint(len(elos)))
    else:
        idx1 = int(forced_idx1)

    distances = np.abs(np.asarray(elos, dtype=np.float64) - float(elos[idx1]))
    weights = np.exp(-distances / distance_scale) + min_weight
    weights[idx1] = 0.0
    probabilities = weights / weights.sum()
    idx2 = int(np.random.choice(len(elos), p=probabilities))
    return idx1, idx2


def _worker_fallback_counts(max_workers: int, fallback_worker_limits=(8, 4, 1)):
    worker_counts = []

    normalized_max_workers = max(1, int(max_workers))
    worker_counts.append(normalized_max_workers)

    for fallback_limit in fallback_worker_limits:
        normalized_worker_count = max(1, min(normalized_max_workers, int(fallback_limit)))
        if normalized_worker_count not in worker_counts:
            worker_counts.append(normalized_worker_count)

    return worker_counts


def parallel_fight_path_results(
    path_matchups,
    num_simulations: int,
    max_workers: int,
    c_puct: float = 1.5,
    temperature: float = 0.0,
    desc: str | None = "Fights",
    position: int | None = None,
    leave: bool = False,
    max_tasks_per_child: int | None = None,
    task_batch_size: int = 1,
    retry_desc: str = "Fight pool",
):
    worker_counts = _worker_fallback_counts(max_workers)
    last_exception = None

    for worker_count in worker_counts:
        try:
            with ParallelFightPool(
                num_simulations=num_simulations,
                max_workers=worker_count,
                c_puct=c_puct,
                temperature=temperature,
                max_tasks_per_child=max_tasks_per_child,
                task_batch_size=task_batch_size,
            ) as fight_pool:
                return fight_pool.fight_path_results(
                    path_matchups,
                    desc=desc,
                    position=position,
                    leave=leave,
                )
        except (BrokenProcessPool, OSError, PermissionError, SystemError) as exc:
            last_exception = exc

    raise FightPoolError(
        f"{retry_desc} failed after retrying with fewer workers."
    ) from last_exception


class ParallelFightPool:

    def __init__(
        self,
        num_simulations: int,
        max_workers: int = 24,
        c_puct: float = 1.5,
        temperature: float = 0.0,
        max_tasks_per_child: int | None = None,
        task_batch_size: int = 1,
    ):
        self.num_simulations = num_simulations
        self.max_workers = max_workers
        self.c_puct = c_puct
        self.temperature = float(temperature)
        self.max_tasks_per_child = max_tasks_per_child
        self.task_batch_size = max(1, int(task_batch_size))
        self.executor = None
        self.pending_futures = {}
        self.result_cache = {}
        self.cache_enabled = self.temperature == 0.0

    def open(self):
        if self.executor is not None:
            return self

        self.executor = ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=get_context("spawn"),
            initializer=_init_fight_worker,
            initargs=(self.num_simulations, self.c_puct, self.temperature),
            max_tasks_per_child=self.max_tasks_per_child,
        )
        self.pending_futures = {}
        return self

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
        self.pending_futures = {}

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def available_worker_slots(self):
        if self.executor is None:
            raise RuntimeError("ParallelFightPool must be opened before use.")
        return max(0, self.max_workers - len(self.pending_futures))

    def has_pending_fights(self):
        return bool(self.pending_futures)

    def submit_path_matchup(self, path_matchup):
        if self.executor is None:
            raise RuntimeError("ParallelFightPool must be opened before use.")

        future = self.executor.submit(_fight_worker, *path_matchup)
        self.pending_futures[future] = path_matchup

    def collect_completed_path_results(self, timeout: float | None = None):
        if self.executor is None:
            raise RuntimeError("ParallelFightPool must be opened before use.")
        if not self.pending_futures:
            return []

        done_futures, _ = wait(
            tuple(self.pending_futures),
            timeout=timeout,
            return_when=FIRST_COMPLETED,
        )

        completed_path_results = []
        for future in done_futures:
            path_matchup = self.pending_futures.pop(future)
            result = future.result()
            if self.cache_enabled:
                self.result_cache[path_matchup] = result
            completed_path_results.append((path_matchup, result))

        return completed_path_results

    def iter_fight_path_results(
        self,
        path_matchups,
        desc: str | None = "Fights",
        position: int | None = None,
        leave: bool = False,
    ):
        if self.executor is None:
            raise RuntimeError("ParallelFightPool must be opened before use.")

        cached_results = []
        future_to_chunk = {}
        uncached_chunk = []
        uncached_chunks = []

        for matchup_index, path_matchup in enumerate(path_matchups):
            cached_result = self.result_cache.get(path_matchup) if self.cache_enabled else None
            if cached_result is not None:
                cached_results.append((matchup_index, cached_result))
                continue

            uncached_chunk.append((matchup_index, path_matchup))
            if len(uncached_chunk) >= self.task_batch_size:
                uncached_chunks.append(list(uncached_chunk))
                uncached_chunk.clear()

        if uncached_chunk:
            uncached_chunks.append(list(uncached_chunk))

        progress_bar = None
        if desc is not None:
            progress_bar = tqdm(
                total=len(path_matchups),
                desc=desc,
                position=position,
                leave=leave,
            )

        try:
            for matchup_index, cached_result in cached_results:
                if progress_bar is not None:
                    progress_bar.update(1)
                yield matchup_index, cached_result

            next_chunk_to_submit = 0

            while (
                next_chunk_to_submit < len(uncached_chunks)
                and len(future_to_chunk) < self.max_workers
            ):
                chunk = uncached_chunks[next_chunk_to_submit]
                chunk_path_matchups = [path_matchup for _, path_matchup in chunk]
                future = self.executor.submit(_fight_worker_batch, chunk_path_matchups)
                future_to_chunk[future] = chunk
                next_chunk_to_submit += 1

            while future_to_chunk:
                done_futures, _ = wait(
                    tuple(future_to_chunk),
                    return_when=FIRST_COMPLETED,
                )
                for future in done_futures:
                    chunk = future_to_chunk.pop(future)
                    results = future.result()
                    for (matchup_index, path_matchup), result in zip(chunk, results):
                        if self.cache_enabled:
                            self.result_cache[path_matchup] = result
                        if progress_bar is not None:
                            progress_bar.update(1)
                        yield matchup_index, result

                    if next_chunk_to_submit < len(uncached_chunks):
                        next_chunk = uncached_chunks[next_chunk_to_submit]
                        next_chunk_path_matchups = [
                            path_matchup for _, path_matchup in next_chunk
                        ]
                        next_future = self.executor.submit(
                            _fight_worker_batch,
                            next_chunk_path_matchups,
                        )
                        future_to_chunk[next_future] = next_chunk
                        next_chunk_to_submit += 1
        finally:
            if progress_bar is not None:
                progress_bar.close()

    def fight_path_results(
        self,
        path_matchups,
        desc: str | None = "Fights",
        position: int | None = None,
        leave: bool = False,
    ):
        results = [None] * len(path_matchups)
        for matchup_index, result in self.iter_fight_path_results(
            path_matchups,
            desc=desc,
            position=position,
            leave=leave,
        ):
            results[matchup_index] = result

        return results
