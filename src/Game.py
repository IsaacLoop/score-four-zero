from enum import IntEnum

BOARD_SIZE = 4
TOTAL_CELLS = BOARD_SIZE**3
EMPTY_CELL = 0
PLAYER_1 = 1
PLAYER_2 = 2


class GameState(IntEnum):
    IN_PROGRESS = 0
    PLAYER_1_WINS = PLAYER_1
    PLAYER_2_WINS = PLAYER_2
    DRAW = 3


def _build_masks():
    """Build every winning mask for the 4x4x4 board."""
    masks = []

    def add_mask(cells):
        """Add one winning mask from its occupied cells."""
        mask = [
            [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
            for _ in range(BOARD_SIZE)
        ]
        for x, y, z in cells:
            mask[x][y][z] = 1
        masks.append(
            tuple(
                tuple(
                    tuple(mask[x][y][z] for z in range(BOARD_SIZE))
                    for y in range(BOARD_SIZE)
                )
                for x in range(BOARD_SIZE)
            )
        )

    for y in range(BOARD_SIZE):
        for z in range(BOARD_SIZE):
            add_mask([(x, y, z) for x in range(BOARD_SIZE)])

    for x in range(BOARD_SIZE):
        for z in range(BOARD_SIZE):
            add_mask([(x, y, z) for y in range(BOARD_SIZE)])

    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            add_mask([(x, y, z) for z in range(BOARD_SIZE)])

    for z in range(BOARD_SIZE):
        add_mask([(i, i, z) for i in range(BOARD_SIZE)])
        add_mask([(i, BOARD_SIZE - 1 - i, z) for i in range(BOARD_SIZE)])

    for y in range(BOARD_SIZE):
        add_mask([(i, y, i) for i in range(BOARD_SIZE)])
        add_mask([(i, y, BOARD_SIZE - 1 - i) for i in range(BOARD_SIZE)])

    for x in range(BOARD_SIZE):
        add_mask([(x, i, i) for i in range(BOARD_SIZE)])
        add_mask([(x, i, BOARD_SIZE - 1 - i) for i in range(BOARD_SIZE)])

    add_mask([(i, i, i) for i in range(BOARD_SIZE)])
    add_mask([(i, i, BOARD_SIZE - 1 - i) for i in range(BOARD_SIZE)])
    add_mask([(i, BOARD_SIZE - 1 - i, i) for i in range(BOARD_SIZE)])
    add_mask([(i, BOARD_SIZE - 1 - i, BOARD_SIZE - 1 - i) for i in range(BOARD_SIZE)])

    return tuple(masks)


MASKS = _build_masks()


def _build_winning_lines_by_cell():
    lines_by_cell = {
        (x, y, z): []
        for x in range(BOARD_SIZE)
        for y in range(BOARD_SIZE)
        for z in range(BOARD_SIZE)
    }

    for mask in MASKS:
        line = tuple(
            (x, y, z)
            for x in range(BOARD_SIZE)
            for y in range(BOARD_SIZE)
            for z in range(BOARD_SIZE)
            if mask[x][y][z] == 1
        )
        for cell in line:
            lines_by_cell[cell].append(line)

    return {cell: tuple(lines) for cell, lines in lines_by_cell.items()}


WINNING_LINES_BY_CELL = _build_winning_lines_by_cell()


class Game:
    """
    Class representing a game of "Score Four", a kind of 3D "Connect Four" with a grid of 4x4x4.
    Beads necessarily fall to the lowest available position in the column, and the first player to align four of their beads in a row (horizontally, vertically, or diagonally) wins the game.
    """

    def __init__(self):
        # dimensions are: horizontal, other horizontal, vertical
        self.board = [
            [[EMPTY_CELL for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
            for _ in range(BOARD_SIZE)
        ]
        self.column_heights = [
            [0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)
        ]
        self.current_player = PLAYER_1
        self.state = GameState.IN_PROGRESS
        self.move_count = 0

    def make_move(self, x: int, y: int) -> bool:
        """
        Make a move for the current player at the specified (x, y) coordinates.
        The bead will fall to the lowest available position in the column.
        Returns True if the move was successful, False if the move cannot be made.
        """
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            raise ValueError("Coordinates must be between 0 and 3 inclusive.")
        if self.state != GameState.IN_PROGRESS:
            return False

        z = self.column_heights[x][y]
        if z >= BOARD_SIZE:
            return False

        player = self.current_player
        self.board[x][y][z] = player
        self.column_heights[x][y] += 1
        self.move_count += 1
        self.update_state(x, y, z, player)

        if self.state == GameState.IN_PROGRESS:
            self.current_player = PLAYER_2 if player == PLAYER_1 else PLAYER_1

        return True

    def update_state(
        self, x: int, y: int, z: int, player: int | None = None
    ) -> GameState:
        """
        Update the game state after a move and return it.
        """
        if player is None:
            player = self.board[x][y][z]

        for line in WINNING_LINES_BY_CELL[(x, y, z)]:
            if all(self.board[cx][cy][cz] == player for cx, cy, cz in line):
                self.state = (
                    GameState.PLAYER_1_WINS
                    if player == PLAYER_1
                    else GameState.PLAYER_2_WINS
                )
                return self.state

        self.state = (
            GameState.DRAW if self.move_count == TOTAL_CELLS else GameState.IN_PROGRESS
        )
        return self.state

    def check_state(self) -> GameState:
        """
        Return the current game state.
        """
        return self.state
