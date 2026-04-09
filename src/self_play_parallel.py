"""
CPU workers run MCTS and environment updates.
One central GPU evaluator owns the model and micro-batches leaf inference.
"""

import queue
import time
import traceback
from multiprocessing import get_context

import numpy as np
from tqdm import tqdm

from .Env import Env
from .GpuBatchEvaluator import GpuBatchEvaluator, GpuBatchEvaluatorClient
from .MCTS import MCTS, Node


class _LiveSelfPlayGame:

    def __init__(self, seed: int):
        self.env = Env()
        self.env.reset()
        self.rng = np.random.default_rng(seed)
        self.trajectory = []
        self.move_index = 0
        self.root = None


class SelfPlayWorker:
    """
    Small long-lived worker that keeps several games alive and advances them
    together. Tree search stays on CPU. Leaf evaluation goes through the shared
    GPU evaluator client.
    """

    def __init__(
        self,
        worker_id: int,
        *,
        initial_game_indices: list[int],
        global_total_games: int,
        live_games: int,
        iteration: int,
        num_sampling_moves: int,
        num_simulations: int,
        c_puct: float,
        dirichlet_alpha: float,
        dirichlet_epsilon: float,
        request_queue,
        response_queue,
        result_queue,
        progress_queue,
        game_index_queue,
    ):
        self.worker_id = int(worker_id)
        self.initial_game_indices = list(initial_game_indices)
        self.global_total_games = int(global_total_games)
        self.live_games = int(live_games)
        self.iteration = int(iteration)
        self.num_sampling_moves = int(num_sampling_moves)
        self.result_queue = result_queue
        self.progress_queue = progress_queue
        self.game_index_queue = game_index_queue
        self.mcts_helper = MCTS(
            model=None,
            num_simulations=num_simulations,
            c_puct=c_puct,
            add_exploration_noise=False,
        )
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.gpu_evaluator = GpuBatchEvaluatorClient(
            worker_id=self.worker_id,
            request_queue=request_queue,
            response_queue=response_queue,
        )
        self.cpu_search_backprop_s = 0.0

    def run(self):
        examples = []
        games_completed = 0
        live_games = []

        while self.initial_game_indices and len(live_games) < self.live_games:
            live_game = self._start_live_game_from_index(self.initial_game_indices.pop())
            live_games.append(live_game)

        while len(live_games) < self.live_games:
            live_game = self._start_live_game_from_queue()
            if live_game is None:
                break
            live_games.append(live_game)

        while live_games:
            self._run_batched_mcts(live_games)

            next_live_games = []
            for live_game in live_games:
                temperature = (
                    1.0 if live_game.move_index < self.num_sampling_moves else 0.0
                )
                pi = self.mcts_helper.visit_counts_to_policy(
                    root=live_game.root,
                    temperature=temperature,
                )

                player = live_game.env.current_player()
                x = live_game.env.canonical_state(perspective_player=player)
                live_game.trajectory.append((x, pi, player))

                action = int(live_game.rng.choice(len(pi), p=pi))
                live_game.env.step(action)
                live_game.move_index += 1
                live_game.root = live_game.root.children.get(action)
                if live_game.root is not None:
                    live_game.root.prior = 1.0

                if live_game.env.is_terminal():
                    for x, pi, player in live_game.trajectory:
                        z = live_game.env.terminal_value(perspective_player=player)
                        examples.append((x, pi, z))
                    games_completed += 1

                    replacement_live_game = self._start_live_game_from_queue()
                    if replacement_live_game is not None:
                        next_live_games.append(replacement_live_game)
                    self.progress_queue.put(1)
                else:
                    next_live_games.append(live_game)

            live_games = next_live_games

        self.result_queue.put(
            {
                "worker_id": self.worker_id,
                "examples": examples,
                "stats": {
                    "cpu_search_backprop_s": float(self.cpu_search_backprop_s),
                    "evaluator_wait_s": float(self.gpu_evaluator.get_stats()["wait_s"]),
                    "evaluator_requests": int(
                        self.gpu_evaluator.get_stats()["requests"]
                    ),
                    "games": int(games_completed),
                    "examples": int(len(examples)),
                },
            }
        )

    def _start_live_game_from_index(self, global_game_index: int):
        return _LiveSelfPlayGame(
            seed=self.iteration * max(1, self.global_total_games) + int(global_game_index)
        )

    def _start_live_game_from_queue(self):
        try:
            global_game_index = self.game_index_queue.get_nowait()
        except queue.Empty:
            return None
        return self._start_live_game_from_index(global_game_index)

    def _add_exploration_noise(self, root: Node, rng):
        if len(root.children) == 0:
            return

        noise = rng.dirichlet([self.dirichlet_alpha] * len(root.children))
        for child, noise_value in zip(root.children.values(), noise):
            child.prior = (
                (1 - self.dirichlet_epsilon) * child.prior
                + self.dirichlet_epsilon * float(noise_value)
            )

    def _expand_pending_nodes(self, pending_nodes, backpropagate: bool):
        if not pending_nodes:
            return

        x_batch = np.stack(
            [
                pending_node["env"].canonical_state(
                    perspective_player=pending_node["node"].player_to_play
                )
                for pending_node in pending_nodes
            ]
        )
        policy_logits_batch, value_batch = self.gpu_evaluator.evaluate(x_batch)

        for pending_node, policy_logits, value in zip(
            pending_nodes,
            policy_logits_batch,
            value_batch,
        ):
            self.mcts_helper.expand_from_evaluation(
                node=pending_node["node"],
                env=pending_node["env"],
                policy_logits=policy_logits,
                value=float(value),
            )

            if backpropagate:
                self.mcts_helper._backpropagate(
                    pending_node["path"],
                    float(value),
                )

            while pending_node["applied_actions"]:
                pending_node["env"].undo_action(pending_node["applied_actions"].pop())

            if backpropagate:
                self.cpu_search_backprop_s += (
                    time.perf_counter() - pending_node["cpu_started_at"]
                )

    def _run_batched_mcts(self, live_games):
        root_expansions = []
        for live_game in live_games:
            if live_game.root is None:
                live_game.root = Node(
                    prior=1.0,
                    player_to_play=live_game.env.current_player(),
                )

            if not live_game.root.is_expanded:
                root_expansions.append(
                    {
                        "node": live_game.root,
                        "env": live_game.env,
                        "path": [live_game.root],
                        "applied_actions": [],
                    }
                )

        self._expand_pending_nodes(root_expansions, backpropagate=False)

        for live_game in live_games:
            self._add_exploration_noise(
                root=live_game.root,
                rng=live_game.rng,
            )

        for _ in range(self.mcts_helper.num_simulations):
            pending_nodes = []

            for live_game in live_games:
                node = live_game.root
                path = [node]
                applied_actions = []
                cpu_started_at = time.perf_counter()

                while node.is_expanded and len(node.children) > 0:
                    action, child = self.mcts_helper._select_child(node)
                    live_game.env.step(action)
                    applied_actions.append(action)
                    node = child
                    path.append(node)

                    if live_game.env.is_terminal():
                        break

                if live_game.env.is_terminal():
                    value = live_game.env.terminal_value(node.player_to_play)
                    self.mcts_helper._backpropagate(path, value)
                    while applied_actions:
                        live_game.env.undo_action(applied_actions.pop())
                    self.cpu_search_backprop_s += time.perf_counter() - cpu_started_at
                else:
                    pending_nodes.append(
                        {
                            "node": node,
                            "env": live_game.env,
                            "path": path,
                            "applied_actions": applied_actions,
                            "cpu_started_at": cpu_started_at,
                        }
                    )

            self._expand_pending_nodes(pending_nodes, backpropagate=True)


def _run_self_play_worker(
    worker_id: int,
    *,
    initial_game_indices: list[int],
    global_total_games: int,
    live_games: int,
    iteration: int,
    num_sampling_moves: int,
    num_simulations: int,
    c_puct: float,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    request_queue,
    response_queue,
    result_queue,
    progress_queue,
    game_index_queue,
):
    try:
        worker = SelfPlayWorker(
            worker_id=worker_id,
            initial_game_indices=initial_game_indices,
            global_total_games=global_total_games,
            live_games=live_games,
            iteration=iteration,
            num_sampling_moves=num_sampling_moves,
            num_simulations=num_simulations,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            request_queue=request_queue,
            response_queue=response_queue,
            result_queue=result_queue,
            progress_queue=progress_queue,
            game_index_queue=game_index_queue,
        )
        worker.run()
    except Exception:
        result_queue.put(
            {
                "worker_id": worker_id,
                "error": traceback.format_exc(),
            }
        )


class ParallelSelfPlayPool:

    def __init__(
        self,
        num_simulations: int,
        c_puct: float,
        dirichlet_alpha: float,
        dirichlet_epsilon: float,
        max_workers: int = 8,
        live_games_per_worker: int = 64,
        evaluator_max_batch_size: int = 1024,
        evaluator_max_wait_s: float = 0.001,
    ):
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.max_workers = int(max_workers)
        self.live_games_per_worker = int(live_games_per_worker)
        self.evaluator_max_batch_size = int(evaluator_max_batch_size)
        self.evaluator_max_wait_s = float(evaluator_max_wait_s)
        self.mp_context = get_context("spawn")

    def open(self):
        return self

    def close(self):
        return None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def generate_examples(
        self,
        model,
        total_games: int,
        num_sampling_moves: int,
        iteration: int,
        desc: str = "Self-play",
        position: int = 1,
        leave: bool = False,
    ):
        if total_games <= 0:
            return []

        worker_count = max(
            1,
            min(
                self.max_workers,
                total_games,
            ),
        )
        total_live_games = min(total_games, worker_count * self.live_games_per_worker)
        request_queue = self.mp_context.Queue()
        response_queues = [self.mp_context.Queue() for _ in range(worker_count)]
        result_queue = self.mp_context.Queue()
        progress_queue = self.mp_context.Queue()
        game_index_queue = self.mp_context.Queue()
        worker_processes = []
        initial_game_count = min(total_games, total_live_games)
        base_initial_games_per_worker = initial_game_count // worker_count
        extra_initial_games = initial_game_count % worker_count
        next_game_index = 0
        initial_game_indices_per_worker = []
        for worker_id in range(worker_count):
            worker_initial_game_count = (
                base_initial_games_per_worker
                + (1 if worker_id < extra_initial_games else 0)
            )
            initial_game_indices_per_worker.append(
                list(
                    range(
                        next_game_index,
                        next_game_index + worker_initial_game_count,
                    )
                )
            )
            next_game_index += worker_initial_game_count

        for game_index in range(initial_game_count, total_games):
            game_index_queue.put(game_index)

        self_play_started_at = time.perf_counter()
        evaluator = GpuBatchEvaluator(
            model=model,
            request_queue=request_queue,
            response_queues=response_queues,
            max_batch_size=self.evaluator_max_batch_size,
            max_wait_s=self.evaluator_max_wait_s,
        )
        evaluator.start()

        try:
            for worker_id in range(worker_count):
                process = self.mp_context.Process(
                    target=_run_self_play_worker,
                    kwargs={
                        "worker_id": worker_id,
                        "initial_game_indices": initial_game_indices_per_worker[
                            worker_id
                        ],
                        "global_total_games": total_games,
                        "live_games": self.live_games_per_worker,
                        "iteration": iteration,
                        "num_sampling_moves": num_sampling_moves,
                        "num_simulations": self.num_simulations,
                        "c_puct": self.c_puct,
                        "dirichlet_alpha": self.dirichlet_alpha,
                        "dirichlet_epsilon": self.dirichlet_epsilon,
                        "request_queue": request_queue,
                        "response_queue": response_queues[worker_id],
                        "result_queue": result_queue,
                        "progress_queue": progress_queue,
                        "game_index_queue": game_index_queue,
                    },
                )
                process.start()
                worker_processes.append(process)

            examples = []
            completed_workers = 0
            worker_stats = []

            with tqdm(
                total=total_games,
                desc=desc,
                position=position,
                leave=leave,
            ) as self_play_bar:
                while completed_workers < worker_count:
                    progress_delta = 0
                    while True:
                        try:
                            progress_delta += int(progress_queue.get_nowait())
                        except queue.Empty:
                            break
                    if progress_delta:
                        self_play_bar.update(progress_delta)

                    try:
                        worker_result = result_queue.get(timeout=0.1)
                    except queue.Empty:
                        if all(
                            process.exitcode is not None for process in worker_processes
                        ) and completed_workers < worker_count:
                            raise RuntimeError(
                                "A self-play worker exited unexpectedly."
                            )
                        continue

                    if worker_result.get("error"):
                        raise RuntimeError(worker_result["error"])

                    examples.extend(worker_result["examples"])
                    worker_stats.append(worker_result["stats"])
                    completed_workers += 1

                progress_delta = 0
                while True:
                    try:
                        progress_delta += int(progress_queue.get_nowait())
                    except queue.Empty:
                        break
                if progress_delta:
                    self_play_bar.update(progress_delta)
            return examples
        finally:
            evaluator.close()
            for process in worker_processes:
                process.join()
            for response_queue in response_queues:
                response_queue.close()
            request_queue.close()
            result_queue.close()
            progress_queue.close()
            game_index_queue.close()
