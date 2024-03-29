import random

class NNet:
    def __init__(self, game):
        self.board_size = game.board_size

    def predict(self, board):
        return [random.random() for i in range(3 * self.board_size ** 2)], 1
