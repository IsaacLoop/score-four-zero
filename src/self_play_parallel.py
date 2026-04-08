"""
Single-process self-play helpers that batch model inference across many live games. Parallelization aspects kinda vibe-coded.
"""

import numpy as np
import torch
from tqdm import tqdm

from .Env import Env
from .MCTS import MCTS, Node
from .models import PVModel


class _LiveSelfPlayGame:

    def __init__(self, seed: int):
        self.env = Env()
        self.env.reset()
        self.rng = np.random.default_rng(seed)
        self.trajectory = []
        self.move_index = 0


def _add_exploration_noise(
    root: Node,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    rng,
):
    if len(root.children) == 0:
        return

    noise = rng.dirichlet([dirichlet_alpha] * len(root.children))
    for child, noise_value in zip(root.children.values(), noise):
        child.prior = (
            (1 - dirichlet_epsilon) * child.prior
            + dirichlet_epsilon * float(noise_value)
        )


def _softmax_legal_policy(logits, legal_actions_mask):
    masked_logits = np.full_like(logits, -np.inf, dtype=np.float32)
    masked_logits[legal_actions_mask] = logits[legal_actions_mask]
    max_logit = np.max(masked_logits[legal_actions_mask])
    exp_logits = np.zeros_like(logits, dtype=np.float32)
    exp_logits[legal_actions_mask] = np.exp(masked_logits[legal_actions_mask] - max_logit)
    return exp_logits / np.sum(exp_logits[legal_actions_mask])


def _evaluate_pending_nodes(
    model: PVModel,
    pending_nodes,
    mcts_helper: MCTS,
):
    if not pending_nodes:
        return

    device = next(model.parameters()).device
    x_batch = np.stack(
        [
            pending_node["env"].canonical_state(
                perspective_player=pending_node["node"].player_to_play
            )
            for pending_node in pending_nodes
        ]
    )
    x_batch = torch.as_tensor(
        x_batch,
        dtype=torch.float32,
        device=device,
    )

    with torch.inference_mode():
        policy_logits_batch, value_batch = model(x_batch)

    policy_logits_batch = policy_logits_batch.detach().cpu().numpy()
    value_batch = value_batch.squeeze(-1).detach().cpu().numpy()

    for pending_node, policy_logits, value in zip(
        pending_nodes,
        policy_logits_batch,
        value_batch,
    ):
        node = pending_node["node"]
        env = pending_node["env"]
        legal_actions_mask = env.legal_actions_mask()
        priors = _softmax_legal_policy(policy_logits, legal_actions_mask)

        for action, prior in enumerate(priors):
            if legal_actions_mask[action]:
                node.children[action] = Node(
                    prior=float(prior),
                    player_to_play=-node.player_to_play,
                )

        node.is_expanded = True
        mcts_helper._backpropagate(pending_node["path"], float(value))


def _run_batched_mcts(
    live_games,
    model: PVModel,
    mcts_helper: MCTS,
    add_exploration_noise: bool,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
):
    roots = [
        Node(prior=1.0, player_to_play=live_game.env.current_player())
        for live_game in live_games
    ]

    _evaluate_pending_nodes(
        model=model,
        pending_nodes=[
            {
                "node": root,
                "env": live_game.env,
                "path": [root],
            }
            for live_game, root in zip(live_games, roots)
        ],
        mcts_helper=mcts_helper,
    )

    if add_exploration_noise:
        for live_game, root in zip(live_games, roots):
            _add_exploration_noise(
                root=root,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_epsilon=dirichlet_epsilon,
                rng=live_game.rng,
            )

    for _ in range(mcts_helper.num_simulations):
        pending_nodes = []

        for live_game, root in zip(live_games, roots):
            env = live_game.env.clone()
            node = root
            path = [node]

            while node.is_expanded and len(node.children) > 0:
                action, child = mcts_helper._select_child(node)
                env.step(action)
                node = child
                path.append(node)

                if env.is_terminal():
                    break

            if env.is_terminal():
                value = env.terminal_value(node.player_to_play)
                mcts_helper._backpropagate(path, value)
            else:
                pending_nodes.append(
                    {
                        "node": node,
                        "env": env,
                        "path": path,
                    }
                )

        _evaluate_pending_nodes(
            model=model,
            pending_nodes=pending_nodes,
            mcts_helper=mcts_helper,
        )

    return roots


class ParallelSelfPlayPool:

    def __init__(
        self,
        num_simulations: int,
        c_puct: float,
        dirichlet_alpha: float,
        dirichlet_epsilon: float,
    ):
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

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
        model: PVModel,
        total_games: int,
        num_sampling_moves: int,
        iteration: int,
        desc: str = "Self-play",
        position: int = 1,
        leave: bool = False,
    ):
        if total_games <= 0:
            return []

        live_games = [
            _LiveSelfPlayGame(seed=iteration * total_games + game_index)
            for game_index in range(total_games)
        ]
        mcts_helper = MCTS(
            model=model,
            num_simulations=self.num_simulations,
            c_puct=self.c_puct,
            add_exploration_noise=False,
        )

        examples = []
        with tqdm(
            total=total_games,
            desc=desc,
            position=position,
            leave=leave,
        ) as self_play_bar:
            while live_games:
                roots = _run_batched_mcts(
                    live_games=live_games,
                    model=model,
                    mcts_helper=mcts_helper,
                    add_exploration_noise=True,
                    dirichlet_alpha=self.dirichlet_alpha,
                    dirichlet_epsilon=self.dirichlet_epsilon,
                )

                next_live_games = []
                for live_game, root in zip(live_games, roots):
                    temperature = (
                        1.0 if live_game.move_index < num_sampling_moves else 0.0
                    )
                    pi = mcts_helper.visit_counts_to_policy(
                        root=root,
                        temperature=temperature,
                    )

                    player = live_game.env.current_player()
                    x = live_game.env.canonical_state(perspective_player=player)
                    live_game.trajectory.append((x, pi, player))

                    action = int(live_game.rng.choice(len(pi), p=pi))
                    live_game.env.step(action)
                    live_game.move_index += 1

                    if live_game.env.is_terminal():
                        for x, pi, player in live_game.trajectory:
                            z = live_game.env.terminal_value(perspective_player=player)
                            examples.append((x, pi, z))
                        self_play_bar.update(1)
                    else:
                        next_live_games.append(live_game)

                live_games = next_live_games

        return examples
