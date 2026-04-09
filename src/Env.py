import numpy as np

from .Game import BOARD_SIZE, Game, GameState


class Env:

    def __init__(self):
        self.game = Game()

    def reset(self):
        self.game = Game()

    def step(self, action: int):
        ok = self.game.make_action(action)
        if not ok:
            raise ValueError(f"Invalid action: {action}")

    def undo_action(self, action: int):
        ok = self.game.undo_action(action)
        if not ok:
            raise ValueError(f"Cannot undo action: {action}")

    def legal_actions_mask(self):
        """
        Boolean mask with one entry per action.
        """
        return (self.game.column_heights < BOARD_SIZE).reshape(-1)

    def is_terminal(self):
        return self.game.game_state != GameState.IN_PROGRESS

    def winner(self):
        assert self.is_terminal(), "Game is not over yet."
        if self.game.game_state == GameState.PLAYER_1_WINS:
            return -1
        if self.game.game_state == GameState.PLAYER_2_WINS:
            return 1
        return 0

    def terminal_value(self, perspective_player):
        winner = self.winner()
        if winner == 0:
            return 0
        if winner == perspective_player:
            return 1
        return -1

    def current_player(self):
        return self.game.current_player

    def canonical_state(self, perspective_player):
        """
        Plane 0: current player's beads
        Plane 1: opponent's beads
        """
        current_player_state = self.game.board == perspective_player
        opponent_state = self.game.board == -perspective_player
        return np.stack([current_player_state, opponent_state], axis=0).astype(
            np.float32
        )
