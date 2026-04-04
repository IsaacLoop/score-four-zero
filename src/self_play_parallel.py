"""
Parallel self-play helpers used by the training loop. Parallelization aspects kinda vibe-coded.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np
import torch
from tqdm import tqdm

from .Env import Env
from .MCTS import MCTS
from .PolicyValueModel import PolicyValueModel

_WORKER_NUM_SIMULATIONS = None
_WORKER_C_PUCT = None
_WORKER_DIRICHLET_ALPHA = None
_WORKER_DIRICHLET_EPSILON = None


def play_one_self_play_game(
    env: Env,
    mcts: MCTS,
    num_sampling_moves: int = 8,
):
    """
    Play one self-play game and return training examples (state, policy, value).
    """
    trajectory = []
    env.reset()

    move_index = 0

    while not env.is_terminal():
        root = mcts.run(root_env=env)
        temperature = 1.0 if move_index < num_sampling_moves else 0.0
        pi = mcts.visit_counts_to_policy(root=root, temperature=temperature)

        player = env.current_player()
        x = env.canonical_state(perspective_player=player)

        trajectory.append((x, pi, player))

        action = np.random.choice(len(pi), p=pi)
        env.step(action)
        move_index += 1

    examples = []
    for x, pi, player in trajectory:
        z = env.terminal_value(perspective_player=player)
        examples.append((x, pi, z))

    return examples

# Everything below is about the parallelism, which is a vibe-coded optimization feature.

def _split_games_across_workers(total_games: int, num_workers: int):
    num_workers = max(1, min(total_games, num_workers))
    base_games_per_worker, extra_games = divmod(total_games, num_workers)

    return [
        base_games_per_worker + (1 if worker_index < extra_games else 0)
        for worker_index in range(num_workers)
        if base_games_per_worker + (1 if worker_index < extra_games else 0) > 0
    ]


def _cpu_model_state_dict(model: PolicyValueModel):
    return {
        parameter_name: parameter.detach().cpu().numpy().copy()
        for parameter_name, parameter in model.state_dict().items()
    }


def _init_self_play_worker(
    num_simulations: int,
    c_puct: float,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
):
    global _WORKER_NUM_SIMULATIONS
    global _WORKER_C_PUCT
    global _WORKER_DIRICHLET_ALPHA
    global _WORKER_DIRICHLET_EPSILON

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    _WORKER_NUM_SIMULATIONS = num_simulations
    _WORKER_C_PUCT = c_puct
    _WORKER_DIRICHLET_ALPHA = dirichlet_alpha
    _WORKER_DIRICHLET_EPSILON = dirichlet_epsilon


def _play_self_play_games_worker(
    model_state_dict,
    num_games: int,
    num_sampling_moves: int,
    seed: int,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = PolicyValueModel()
    model.load_state_dict(
        {
            parameter_name: torch.from_numpy(parameter_value)
            for parameter_name, parameter_value in model_state_dict.items()
        }
    )
    model.eval()

    mcts = MCTS(
        model=model,
        num_simulations=_WORKER_NUM_SIMULATIONS,
        c_puct=_WORKER_C_PUCT,
        add_exploration_noise=True,
        dirichlet_alpha=_WORKER_DIRICHLET_ALPHA,
        dirichlet_epsilon=_WORKER_DIRICHLET_EPSILON,
    )

    examples = []
    for _ in range(num_games):
        env = Env()
        examples.extend(
            play_one_self_play_game(
                env=env,
                mcts=mcts,
                num_sampling_moves=num_sampling_moves,
            )
        )

    return examples


class ParallelSelfPlayPool:

    def __init__(
        self,
        max_workers: int,
        num_simulations: int,
        c_puct: float,
        dirichlet_alpha: float,
        dirichlet_epsilon: float,
        max_tasks_per_child: int = 100,
    ):
        self.max_workers = max_workers
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.max_tasks_per_child = max_tasks_per_child
        self.executor = None

    def open(self):
        if self.executor is not None:
            return self

        self.executor = ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=get_context("spawn"),
            initializer=_init_self_play_worker,
            initargs=(
                self.num_simulations,
                self.c_puct,
                self.dirichlet_alpha,
                self.dirichlet_epsilon,
            ),
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

    def generate_examples(
        self,
        model: PolicyValueModel,
        total_games: int,
        num_sampling_moves: int,
        iteration: int,
        desc: str = "Self-play",
        position: int = 1,
        leave: bool = False,
    ):
        if self.executor is None:
            raise RuntimeError("ParallelSelfPlayPool must be entered before use.")

        model_state_dict = _cpu_model_state_dict(model)
        worker_game_counts = _split_games_across_workers(total_games, self.max_workers)

        future_to_num_games = {}
        for worker_index, num_games in enumerate(worker_game_counts):
            future = self.executor.submit(
                _play_self_play_games_worker,
                model_state_dict,
                num_games,
                num_sampling_moves,
                iteration * self.max_workers + worker_index,
            )
            future_to_num_games[future] = num_games

        examples = []
        with tqdm(
            total=total_games,
            desc=desc,
            position=position,
            leave=leave,
        ) as self_play_bar:
            for future in as_completed(future_to_num_games):
                examples.extend(future.result())
                self_play_bar.update(future_to_num_games[future])

        return examples
