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
        eval_cache=None,
        model_cache_key=None,
    ):
        self.model = model
        self.device = (
            next(model.parameters()).device if model is not None else None
        )
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
        self.eval_cache = eval_cache
        self.model_cache_key = model_cache_key

    def run(self, root_env: Env, root: Node = None):
        if root is None:
            root = Node(prior=1.0, player_to_play=root_env.current_player())
        else:
            assert root.player_to_play == root_env.current_player()

        if not root.is_expanded:
            self._expand_node(root, root_env)

        if self.add_exploration_noise:
            self._add_exploration_noise(root)

        for _ in range(self.num_simulations):
            node = root
            path = [node]
            applied_actions = []

            try:
                # 1 - Selection
                while node.is_expanded and len(node.children) > 0:
                    action, child = self._select_child(node)
                    root_env.step(action)
                    applied_actions.append(action)
                    node = child
                    path.append(node)

                    if root_env.is_terminal():
                        break

                # 2 - Expansion
                if root_env.is_terminal():
                    value = root_env.terminal_value(node.player_to_play)
                else:
                    value = self._expand_node(node, root_env)

                # 3 - Backpropagation (not ML backpropagation!!)
                self._backpropagate(path, value)
            finally:
                while applied_actions:
                    root_env.undo_action(applied_actions.pop())

        return root

    def evaluate_state(self, env: Env, perspective_player: int):
        assert self.model is not None, "MCTS.evaluate_state requires a model."
        canonical_state = env.canonical_state(perspective_player=perspective_player)

        model_eval_cache = None
        state_key = None
        if self.eval_cache is not None and self.model_cache_key is not None:
            model_eval_cache = self.eval_cache.setdefault(self.model_cache_key, {})
            state_key = canonical_state.tobytes()
            cached_evaluation = model_eval_cache.get(state_key)
            if cached_evaluation is not None:
                return cached_evaluation

        x = torch.as_tensor(
            canonical_state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.inference_mode():
            policy_logits, value = self.model(x)
        evaluation = (
            policy_logits.squeeze(0).detach().cpu().numpy(),
            float(value.squeeze(0).item()),
        )

        if model_eval_cache is not None:
            model_eval_cache[state_key] = evaluation

        return evaluation

    def expand_from_evaluation(self, node: Node, env: Env, policy_logits, value: float):
        legal_actions_mask = env.legal_actions_mask()
        masked_logits = np.full(policy_logits.shape, -np.inf, dtype=np.float32)
        masked_logits[legal_actions_mask] = policy_logits[legal_actions_mask]
        max_logit = np.max(masked_logits[legal_actions_mask])
        exp_logits = np.zeros_like(masked_logits, dtype=np.float32)
        exp_logits[legal_actions_mask] = np.exp(
            masked_logits[legal_actions_mask] - max_logit
        )
        priors = exp_logits / np.sum(exp_logits[legal_actions_mask])

        node.children.clear()
        for action, prior in enumerate(priors):
            if legal_actions_mask[action]:
                node.children[action] = Node(
                    prior=float(prior),
                    player_to_play=-node.player_to_play,
                )

        node.is_expanded = True
        return float(value)

    def _expand_node(self, node: Node, env: Env) -> float:
        policy_logits, value = self.evaluate_state(
            env=env,
            perspective_player=node.player_to_play,
        )
        return self.expand_from_evaluation(node, env, policy_logits, value)

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
