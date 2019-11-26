

class NNet:
    def __init__(self, game):
        self.board_size = game.board_size
        pass

    def predict(self, board):
        return [0.4 for i in range(3 * self.board_size ** 2)], 1
        pass
