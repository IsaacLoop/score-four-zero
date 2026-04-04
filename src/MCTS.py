import numpy as np
import torch

from .Env import Env
from .Game import BOARD_SIZE


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
        c_puct: float,  # "Prior - Upper Confidence bound applied to Trees"
        add_exploration_noise: bool = False,
        dirichlet_alpha: float = None,
        dirichlet_epsilon: float = None,
    ):
        self.model = model
        self.device = next(model.parameters()).device
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

        return root

    def _expand_node(self, node: Node, env: Env) -> float:
        x = torch.as_tensor(
            env.canonical_state(perspective_player=node.player_to_play),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.inference_mode():
            policy_logits, value = self.model(x)
        policy_logits = policy_logits.squeeze(0)
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
        return float(value.squeeze(0).item())

    def _add_exploration_noise(self, root: Node):
        if len(root.children) == 0:
            return

        noise = np.random.dirichlet([self.dirichlet_alpha] * len(root.children))

        for child, noise_value in zip(root.children.values(), noise):
            child.prior = (
                1 - self.dirichlet_epsilon
            ) * child.prior + self.dirichlet_epsilon * float(noise_value)

    def _select_child(self, node: Node):
        """
        Choose action maximizing Q + U

        Q is the value of a child
        U is the exploration bonus based on prior and visit counter
        """
        best_score = float("-inf")
        best_action = None
        best_child = None

        parent_visits_sqrt = np.sqrt(node.visit_count + 1)

        for action, child in node.children.items():
            q = (
                -child.value()
            )  # minus because next move is from the PoV of the opponent. gonna happen a lot

            u = self.c_puct * child.prior * parent_visits_sqrt / (1 + child.visit_count)

            score = q + u

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _backpropagate(self, path: list[Node], value: float):
        for node in path[::-1]:
            node.value_sum += value
            node.visit_count += 1
            value = -value

    def visit_counts_to_policy(self, root: Node, temperature: float):
        """
        Higher temperature means more exploration. 0 means greedy.
        """
        visits = np.zeros(BOARD_SIZE**2, dtype=np.float32)

        for action, child in root.children.items():
            visits[action] = child.visit_count

        if temperature == 0.0:
            pi = np.zeros(BOARD_SIZE**2, dtype=np.float32)
            if root.children:
                best_action = max(
                    root.children.items(),
                    key=lambda item: item[1].visit_count,
                )[0]
            else:
                best_action = int(np.argmax(visits))
            pi[best_action] = 1.0
            return pi

        pi = visits ** (1 / temperature)
        visit_total = np.sum(pi)
        if visit_total == 0:
            pi = np.zeros(BOARD_SIZE**2, dtype=np.float32)
            if root.children:
                legal_actions = list(root.children)
                pi[legal_actions] = 1.0 / len(legal_actions)
            return pi
        return pi / visit_total
