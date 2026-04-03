"""
Entirely vibe-coded file (gpt-5.4 xhigh reasoning). Parallelization is too much of a pain to do manually in 2026 for non-critical code lol
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np
import torch
from tqdm import tqdm

from .Env import Env
from .MCTS import MCTS
from .PolicyValueModel import PolicyValueModel

_WORKER_MODEL_CACHE = {}
_WORKER_NUM_SIMULATIONS = None
_WORKER_C_PUCT = None


def _init_fight_worker(num_simulations: int, c_puct: float):
    global _WORKER_MODEL_CACHE, _WORKER_NUM_SIMULATIONS, _WORKER_C_PUCT

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    _WORKER_MODEL_CACHE = {}
    _WORKER_NUM_SIMULATIONS = num_simulations
    _WORKER_C_PUCT = c_puct


def _get_cached_model(model_path: str):
    model = _WORKER_MODEL_CACHE.get(model_path)
    if model is None:
        checkpoint = torch.load(model_path, map_location="cpu")
        model = PolicyValueModel()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        _WORKER_MODEL_CACHE[model_path] = model
    return model


def _fight_worker(model_path_1: str, model_path_2: str):
    model1 = _get_cached_model(model_path_1)
    model2 = _get_cached_model(model_path_2)

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

    while not env.is_terminal():
        if env.current_player() == -1:
            root = mcts1.run(root_env=env)
            pi = mcts1.visit_counts_to_policy(root=root, temperature=0.0)
            action = int(pi.argmax())
            env.step(action)
        else:
            root = mcts2.run(root_env=env)
            pi = mcts2.visit_counts_to_policy(root=root, temperature=0.0)
            action = int(pi.argmax())
            env.step(action)

    return env.winner()


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


def parallel_fight_results(
    model_paths,
    matchups,
    num_simulations: int,
    max_workers: int = 24,
    c_puct: float = 1.5,
    desc: str = "Fights",
):
    with ParallelFightPool(
        model_paths=model_paths,
        num_simulations=num_simulations,
        max_workers=max_workers,
        c_puct=c_puct,
    ) as fight_pool:
        return fight_pool.fight_results(matchups, desc=desc)


class ParallelFightPool:

    def __init__(
        self,
        model_paths,
        num_simulations: int,
        max_workers: int = 24,
        c_puct: float = 1.5,
        max_tasks_per_child: int = 100,
    ):
        self.model_paths = tuple(str(path) for path in model_paths)
        self.num_simulations = num_simulations
        self.max_workers = max_workers
        self.c_puct = c_puct
        self.max_tasks_per_child = max_tasks_per_child
        self.executor = None

    def open(self):
        if self.executor is not None:
            return self

        self.executor = ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=get_context("spawn"),
            initializer=_init_fight_worker,
            initargs=(self.num_simulations, self.c_puct),
            max_tasks_per_child=self.max_tasks_per_child,
        )
        return self

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def fight_results(self, matchups, desc: str | None = "Fights"):
        path_matchups = [
            (self.model_paths[idx1], self.model_paths[idx2])
            for idx1, idx2 in matchups
        ]
        return self.fight_path_results(path_matchups, desc=desc)

    def iter_fight_path_results(self, path_matchups, desc: str | None = "Fights"):
        if self.executor is None:
            raise RuntimeError("ParallelFightPool must be opened before use.")

        future_to_matchup_index = {
            self.executor.submit(_fight_worker, path1, path2): matchup_index
            for matchup_index, (path1, path2) in enumerate(path_matchups)
        }

        completed_futures = as_completed(future_to_matchup_index)
        if desc is not None:
            completed_futures = tqdm(
                completed_futures,
                total=len(future_to_matchup_index),
                desc=desc,
            )

        for future in completed_futures:
            matchup_index = future_to_matchup_index[future]
            yield matchup_index, future.result()

    def fight_path_results(self, path_matchups, desc: str | None = "Fights"):
        results = [None] * len(path_matchups)
        for matchup_index, result in self.iter_fight_path_results(
            path_matchups,
            desc=desc,
        ):
            results[matchup_index] = result

        return results


class ParallelEloPool:

    def __init__(
        self,
        num_simulations: int,
        max_workers: int = 24,
        c_puct: float = 1.5,
        elo_k: float = 20.0,
        matchmaking_distance_scale: float = 200.0,
        matchmaking_min_weight: float = 0.05,
        max_tasks_per_child: int = 100,
    ):
        self.max_workers = max_workers
        self.elo_k = elo_k
        self.matchmaking_distance_scale = matchmaking_distance_scale
        self.matchmaking_min_weight = matchmaking_min_weight
        self.fight_pool = ParallelFightPool(
            model_paths=(),
            num_simulations=num_simulations,
            max_workers=max_workers,
            c_puct=c_puct,
            max_tasks_per_child=max_tasks_per_child,
        )

    def open(self):
        self.fight_pool.open()
        return self

    def close(self):
        self.fight_pool.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def evaluate(
        self,
        elos,
        model_paths,
        num_fights: int,
        always_last: bool = False,
        desc: str = "Evaluation",
        position: int = 3,
        leave: bool = False,
    ):
        if len(model_paths) < 2 or num_fights <= 0:
            return list(elos)

        updated_elos = list(elos)
        model_paths = [str(path) for path in model_paths]
        fights_remaining = num_fights

        with tqdm(
            total=num_fights,
            desc=desc,
            position=position,
            leave=leave,
        ) as evaluation_bar:
            while fights_remaining > 0:
                batch_size = min(self.max_workers, fights_remaining)
                matchup_indices = []
                swapped_flags = []

                for _ in range(batch_size):
                    idx1, idx2 = sample_matchup(
                        updated_elos,
                        distance_scale=self.matchmaking_distance_scale,
                        min_weight=self.matchmaking_min_weight,
                        forced_idx1=(len(updated_elos) - 1) if always_last else None,
                    )
                    matchup_indices.append((idx1, idx2))
                    swapped_flags.append(bool(always_last and np.random.rand() < 0.5))

                path_matchups = []
                for (idx1, idx2), swapped in zip(matchup_indices, swapped_flags):
                    if swapped:
                        path_matchups.append((model_paths[idx2], model_paths[idx1]))
                    else:
                        path_matchups.append((model_paths[idx1], model_paths[idx2]))
                results = self.fight_pool.fight_path_results(path_matchups, desc=None)

                for (idx1, idx2), swapped, result in zip(matchup_indices, swapped_flags, results):
                    remapped_result = remap_fight_result(result, swapped)
                    updated_elos[idx1], updated_elos[idx2] = compute_new_elos(
                        updated_elos[idx1],
                        updated_elos[idx2],
                        remapped_result,
                        k=self.elo_k,
                    )

                fights_remaining -= batch_size
                evaluation_bar.update(batch_size)

        return updated_elos
