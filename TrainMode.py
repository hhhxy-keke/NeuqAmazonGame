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
        trainExamples = []
        board = self.game.get_init_board()
        play_step = 0
        while True:
            play_step += 1
            print('---------------------------')
            print('第', play_step, '步')
            print(board)
            self.mcts.episodeStep = play_step
            # 这里进行了更新
            # 在蒙特卡洛数中，以player1移动
            transformed_board = self.game.get_transformed_board(board, self.player)
            # 将翻转后的棋盘和temp传给蒙特卡洛树搜索方法得到当前的策略
            pi = self.mcts.get_best_action(transformed_board, self.player)     # 进行多次mcts搜索得出来概率

            # 将局面和策略顺时针旋转180度，返回4个棋盘和策略组成的元组
            sym = self.game.get_symmetries(transformed_board, pi)
            for b, p in sym:
                trainExamples.append([b, self.player, p, None])
            pi_start = pi[0:self.game.board_size]
            pi_end = pi[self.game.board_size:2*self.game.board_size]
            pi_arrow = pi[2*self.game.board_size:3*self.game.board_size]
            # 深拷贝
            copy_board = np.copy(transformed_board)
            # 选择下一步最优动作
            while True:
                # 将1*100的策略概率的数组传入得到 0~99 的行动点 , action_start,end,arrow都是选出来的点 eg: 43,65....
                action_start = np.random.choice(len(pi_start), p=pi_start)
                action_end = np.random.choice(len(pi_end), p=pi_end)
                action_arrow = np.random.choice(len(pi_arrow), p=pi_arrow)
                # 加断言保证起子点有棋子，落子点和放箭点均无棋子
                assert copy_board[action_start // self.game.board_size][action_start % self.game.board_size] == self.player
                assert copy_board[action_end // self.game.board_size][action_end % self.game.board_size] == EMPTY
                assert copy_board[action_arrow // self.game.board_size][action_arrow % self.game.board_size] == EMPTY
                if self.game.is_legal_move(action_start, action_end):
                    copy_board[action_start // self.game.board_size][action_start % self.game.board_size] = EMPTY
                    copy_board[action_end//self.game.board_size][action_end % self.game.board_size] = self.player
                    if self.game.is_legal_move(action_end, action_arrow):
                        nest_action = [action_start, action_end, action_arrow]
                        break   # 跳出While True 循环
                    else:
                        copy_board[action_start // self.game.board_size][action_start % self.game.board_size] = self.player
                        copy_board[action_end // self.game.board_size][action_end % self.game.board_size] = EMPTY

            print(nest_action)
            print('---------------------------')
            board, self.player = self.game.get_next_state(board, self.player, nest_action)

            r = self.game.get_game_ended(board, self.player)

            if r != 0:  # 胜负已分
                return [(x[0], x[2], r*((-1)**(x[1] != self.player))) for x in trainExamples]
                #       [(棋盘 , 策略 , 1/-1(输赢奖励)),(s,pi,z),(),().....]


if __name__ == "__main__":
    game = Game(5)
    # train = TrainMode(game, nnet)
    game.board[4][4] = 100
    print(game.board)
    print(game.get_init_board(5))
    print(game.get_action_size())
