import numpy as np
import torch

from .Env import Env


class Node:

    def __init__(self, prior: float, player_to_play: int):
        self.prior = prior
        self.player_to_play = player_to_play

        self.visit_count = 0
        self.value_sum = 0.0

        self.children = {}
        self.is_expanded = False

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:

    def __init__(
        self,
        model,
        num_simulations: int,
        c_puct: float,
        add_exploration_noise: bool = False,
        dirichlet_alpha: float = None,
        dirichlet_epsilon: float = None,
    ):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.add_exploration_noise = add_exploration_noise
        if add_exploration_noise:
            assert not (
                dirichlet_alpha is None or dirichlet_epsilon is None
            ), "Dirichlet parameters must be provided when add_exploration_noise is True."
        else:
            assert (
                dirichlet_alpha is None and dirichlet_epsilon is None
            ), "Dirichlet parameters must be None when add_exploration_noise is False."
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def run(self, root_env: Env):
        root = Node(prior=1.0, player_to_play=root_env.current_player())

        self._expand_node(root, root_env)

        if self.add_exploration_noise:
            self._add_exploration_noise(root)

        for _ in range(self.num_simulations):
            env = root_env.clone()
            node = root
            path = [node]

            # 1 - Selection
            while node.is_expanded and len(node.children) > 0:
                action, child = self._select_child(node)
                env.step(action)
                node = child
                path.append(node)

                if env.is_terminal():
                    break

            # 2 - Expansion
            if env.is_terminal():
                value = env.terminal_value(node.player_to_play)
            else:
                value = self._expand_node(node, env)

            # 3 - Backpropagation (not ML backpropagation!!)
            self._backpropagate(path, value)

    def _expand_node(self, node: Node, env: Env) -> float:
        x = env.canonical_state(perspective_player=node.player_to_play)
        policy_logits, value = self.model.forward(x)
        policy_logits = policy_logits.reshape(-1)
        legal_actions_mask = torch.as_tensor(
            env.legal_actions_mask(),
            dtype=torch.bool,
            device=policy_logits.device,
        )
        masked_logits = policy_logits.masked_fill(~legal_actions_mask, float("-inf"))
        priors = torch.softmax(masked_logits, dim=0)

        for a, prior in enumerate(priors):
            if legal_actions_mask[a].item():
                node.children[a] = Node(
                    prior=float(prior.item()),
                    player_to_play=-node.player_to_play,
                )

        node.is_expanded = True
        return float(value.item())
