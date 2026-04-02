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
        Boolean mask with one entry per action.
        """
        column_heights = np.asarray(self.game.column_heights)
        return (column_heights < BOARD_SIZE).reshape(-1)

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
        new_env = Env.__new__(Env)
        new_game = Game.__new__(Game)

        new_game.board = [
            [column[:] for column in plane]
            for plane in self.game.board
        ]
        new_game.column_heights = [row[:] for row in self.game.column_heights]
        new_game.current_player = self.game.current_player
        new_game.game_state = self.game.game_state
        new_game.move_count = self.game.move_count

        new_env.game = new_game
        return new_env

    def canonical_state(self, perspective_player):
        """
        Plane 0: current player's beads
        Plane 1: opponent's beads
        """
        board = np.asarray(self.game.board, dtype=np.int8)
        current_player_state = board == perspective_player
        opponent_state = board == -perspective_player
        return np.stack([current_player_state, opponent_state], axis=0).astype(
            np.float32
        )
