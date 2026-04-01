from ..Game import BOARD_SIZE, EMPTY_CELL, PLAYER_1, PLAYER_2, GameState
from .GameView import GameView


class AsciiView(GameView):
    """Display the game as stacked ASCII layers."""

    def __init__(self, game):
        super().__init__(game)

    def update(self):
        symbols = {
            EMPTY_CELL: ".",
            PLAYER_1: "X",
            PLAYER_2: "O",
        }

        if self.game.current_player == PLAYER_1:
            player_turn = 1
        else:
            player_turn = 2

        if self.game.get_game_state() == GameState.IN_PROGRESS:
            game_state = "In progress"
        elif self.game.get_game_state() == GameState.PLAYER_1_WINS:
            game_state = "Player 1 wins"
        elif self.game.get_game_state() == GameState.PLAYER_2_WINS:
            game_state = "Player 2 wins"
        else:
            game_state = "Draw"

        print(
            "z=   "
            + "   ".join(f"{z:<7}" for z in range(BOARD_SIZE))
        )
        print(
            "x=   "
            + "   ".join("0 1 2 3" for _ in range(BOARD_SIZE))
        )

        for y in range(BOARD_SIZE - 1, -1, -1):
            row = []
            for z in range(BOARD_SIZE):
                row.append(
                    " ".join(
                        symbols[self.game.board[x][y][z]]
                        for x in range(BOARD_SIZE)
                    )
                )
            print(f"y={y}  " + "   ".join(row))

        print()
        print(f"Player's turn: {player_turn}")
        print(f"Game state: {game_state}")
