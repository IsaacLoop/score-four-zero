from copy import deepcopy
import numpy as np

from .Game import BOARD_SIZE, Game, GameState


class Env:

    def __init__(self):
        self.game = Game()

    def reset(self):
        self.game = Game()

    def step(self, action: int):
        ok = self.game.make_move(action // BOARD_SIZE, action % BOARD_SIZE)
        if not ok:
            raise ValueError(f"Invalid action: {action}")

    def legal_actions_mask(self):
        """
        1 if the action is legal, 0 otherwise.
        """
        mask = [0] * (BOARD_SIZE * BOARD_SIZE)
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if self.game.column_heights[x][y] < BOARD_SIZE:
                    mask[x * BOARD_SIZE + y] = 1
        return mask

    def is_terminal(self):
        return self.game.game_state != GameState.IN_PROGRESS

    def winner(self):
        assert self.is_terminal(), "Game is not over yet."
        if self.game.game_state == GameState.PLAYER_1_WINS:
            return -1
        elif self.game.game_state == GameState.PLAYER_2_WINS:
            return 1
        else:
            return 0
        
    def terminal_value(self, perspective_player):
        winner = self.winner()
        if winner == 0:
            return 0
        elif winner == perspective_player:
            return 1
        else:
            return -1

    def current_player(self):
        return self.game.current_player

    def clone(self):
        new_env = Env()
        new_env.game = deepcopy(self.game)
        return new_env

    def canonical_state(self, perspective_player):
        """
        Plane 0: current player's beads
        Plane 1: opponent's beads
        """
        player_1_state = np.array(self.game.board) == -1
        player_2_state = np.array(self.game.board) == 1
        if perspective_player == -1:
            return np.stack([player_1_state, player_2_state], axis=0).astype(int)
        else:
            return np.stack([player_2_state, player_1_state], axis=0).astype(int)