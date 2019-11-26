from Game import Game
import numpy as np
from Mcts import Mcts

# 训练模式的参数
args = dict({
    'numIters': 1000,
    'numEps': 1,
    'tempThreshold': 35,    # 探索效率
    'updateThreshold': 0.55,
    'maxlenOfQueue': 200000,
    'num_mcts_search': 1000,      # 从当前状态搜索到一个未被扩展的叶结点25次
    'arenaCompare': 40,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/models/','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

BLACK = -2
WHITE = 2
EMPTY = 0
ARROW = 1


class TrainMode:

    def __init__(self, game, nnet):
        """
        :param game: 棋盘对象
        :param nnet: 神经网络对象
        """
        self.player = WHITE
        self.game = game
        self.nnet = nnet
        # self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = Mcts(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # 类似于经验池的作用 history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def learn(self):
        pass

    # 完整下一盘游戏
    def play_one_game(self):
        steps_train_data = []
        board = self.game.get_init_board()
        play_step = 0
        while True:
            play_step += 1
            print('---------------------------')
            print('第', play_step, '步')
            print(board)
            self.mcts.episodeStep = play_step
            # 这里进行了更新
            # 在MCTS中，始终以白棋视角选择
            transformed_board = self.game.get_transformed_board(board, self.player)
            # 将翻转后的棋盘和temp传给蒙特卡洛树搜索方法得到当前的策略
            # 进行多次mcts搜索得出来概率
            next_action, steps_train_data = self.mcts.get_best_action(transformed_board, self.player)
            print(next_action)
            board, self.player = self.game.get_next_state(board, self.player, next_action)

            r = self.game.get_game_ended(board, self.player)

            if r != 0:  # 胜负已分
                return [(x[0], x[2], r*((-1)**(x[1] != self.player))) for x in steps_train_data]
                #       [(棋盘 , 策略 , 1/-1(输赢奖励)),(s,pi,z),(),().....]


if __name__ == "__main__":
    game = Game(5)
    # train = TrainMode(game, nnet)
    game.board[4][4] = 10
    print(game.board)
    print(game.get_init_board(5))
    print(game.get_action_size())
