"""
Entirely vibe-coded file (gpt-4 xhigh reasoning). Parallelization is too much of a pain to do manually in 2026 for non-critical code lol
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import torch
from tqdm import tqdm

from .Env import Env
from .MCTS import MCTS
from .PolicyValueModel import PolicyValueModel


_WORKER_MODEL_PATHS = ()
_WORKER_MODEL_CACHE = {}
_WORKER_NUM_SIMULATIONS = None
_WORKER_C_PUCT = None


def _init_fight_worker(model_paths, num_simulations: int, c_puct: float):
    global _WORKER_MODEL_PATHS, _WORKER_MODEL_CACHE, _WORKER_NUM_SIMULATIONS, _WORKER_C_PUCT

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    _WORKER_MODEL_PATHS = tuple(model_paths)
    _WORKER_MODEL_CACHE = {}
    _WORKER_NUM_SIMULATIONS = num_simulations
    _WORKER_C_PUCT = c_puct


def _get_cached_model(model_index: int):
    model = _WORKER_MODEL_CACHE.get(model_index)
    if model is None:
        checkpoint = torch.load(_WORKER_MODEL_PATHS[model_index], map_location="cpu")
        model = PolicyValueModel()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        _WORKER_MODEL_CACHE[model_index] = model
    return model


def _fight_worker(model_index_1: int, model_index_2: int):
    model1 = _get_cached_model(model_index_1)
    model2 = _get_cached_model(model_index_2)

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
    ):
        self.model_paths = tuple(model_paths)
        self.num_simulations = num_simulations
        self.max_workers = max_workers
        self.c_puct = c_puct
        self.executor = None

    def __enter__(self):
        self.executor = ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=get_context("spawn"),
            initializer=_init_fight_worker,
            initargs=(self.model_paths, self.num_simulations, self.c_puct),
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    def fight_results(self, matchups, desc: str | None = "Fights"):
        if self.executor is None:
            raise RuntimeError("ParallelFightPool must be entered before use.")

        results = [None] * len(matchups)
        future_to_matchup_index = {
            self.executor.submit(_fight_worker, idx1, idx2): matchup_index
            for matchup_index, (idx1, idx2) in enumerate(matchups)
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
            results[matchup_index] = future.result()

        return results
