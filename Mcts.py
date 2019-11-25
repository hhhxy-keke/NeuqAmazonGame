import numpy as np

BLACK = -2
WHITE = 2
EMPTY = 0
ARROW = 1


class Mcts:
    def __init__(self, game, nnet, args):
        """
        蒙特卡洛树搜索类：对给定棋盘状态使用MCTS方法得到下一步最优行动
        :param game: 当前棋盘对象
        :param nnet: 神经网络
        :param args: 训练参数
        """
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Ns = {}        # stores #times board s was visited
        self.Nsa = {}
        self.action = []
        self.Ns_start = {}
        self.Ns_end = {}
        self.Ns_arrow = {}

        self.Ps = {}        # 存储该状态初始由神经网路得到策略概率 ,stores initial policy (returned by neural net)
        self.Ps_start = {}
        self.Ps_end = {}
        self.Ps_arrow = {}

        self.top_start = {}
        self.top_end = {}
        self.top_arrow = {}

        self.Game_End = {}        # 存储已知的输赢状态 eg{str(board):-1/1,.....} , stores game.getGameEnded ended for board s
        self.Vs = {}        # 存储该状态下可以的所有走法 , stores game.getValidMoves for board s
        self.episodeStep = 0

    def get_best_action(self, board, player):
        """
        :param board: 当前棋盘
        :param player: 当前玩家
        :return:
        """
        pass

    def search(self, board, player):
        """
        对状态进行一次递归的模拟搜索，增加各状态（棋盘）的访问次数N
        :param board: 棋盘当前
        :param player: 当前玩家
        :return:
        """
        board_key = self.game.to_string(board)
        if board_key not in self.Game_End:
            self.Game_End[board_key] = self.game.get_game_ended(board, player)
        if self.Game_End[board_key] != 0:
            return -self.Game_End[board_key]

        pass
