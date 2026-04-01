from abc import ABC, abstractmethod

from ..Game import Game

# i've got plans to have a prettier 3D renderer,
# but for now just ascii will do fine!

class GameView(ABC):
    def __init__(self, game: Game):
        self.game = game

    @abstractmethod
    def update(self):
        pass
