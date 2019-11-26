import time
from collections import deque
from random import shuffle
from Game import Game
import numpy as np
from Mcts import Mcts


BLACK = -2
WHITE = 2
EMPTY = 0
ARROW = 1


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


# 训练模式的参数
args = dotdict({
    'num_iter': 1000,          # 神经网络训练次数
    'num_play_game': 1,        # 下“num_play_game”盘棋训练一次NNet
    'max_len_queue': 200000,   # 双向列表最大长度
    'num_mcts_search': 1000,   # 从某状态模拟搜索到叶结点次数
    'max_batch_size': 20,      # NNet每次训练的最大数据量
    'Cpuct': 1,                # 置信上限函数中的“温度”超参数
    'arenaCompare': 40,
    'tempThreshold': 35,       # 探索效率
    'updateThreshold': 0.55,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/models/', 'best.pth.tar'),
})


class TrainMode:

    def __init__(self, game, nnet):
        """
        :param game: 棋盘对象
        :param nnet: 神经网络对象
        """
        self.args = args
        self.player = WHITE
        self.game = game
        self.nnet = nnet
        # self.pnet = self.nnet.__class__(self.game)  # the competitor network  # 旧网络
        self.mcts = Mcts(self.game, self.nnet, self.args)
        self.batch = []        # 每次给NNet喂的数据量,但类型不对（多维列表）
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    # 调用NNet开始训练
    def learn(self):
        for i in range(1, self.args.num_iter + 1):
            print('************ ITER： ' + str(i) + '************')
            # 每次都执行
            if not self.skipFirstSelfPlay or i > 1:
                # deque：双向队列  max_len：队列最大长度：self.args.max_len_queue
                iter_train_data = deque([], maxlen=self.args.max_len_queue)

                # 下“num_play_game”盘棋训练一次NNet
                for plays in range(self.args.num_play_game):  
                    # 重置搜索树
                    self.mcts = Mcts(self.game, self.nnet, self.args)
                    iter_train_data += self.play_one_game()

                # save the iteration examples to the history
                self.batch.append(iter_train_data)

            # 不断更新训练数据
            # 如果 训练数据 大于规定的训练长度，则将最旧的数据删除
            if len(self.batch) > self.args.max_batch_size:
                print("len(max_batch_size) =", len(self.batch),
                      " => remove the oldest batch")
                self.batch.pop(0)
            
            # 保存训练数据
            self.saveTrainExamples(i - 1)

            # 原batch是多维列表，此处标准化batch
            standard_batch = []
            for e in self.batch:
                # extend() 在列表末尾一次性追加其他序列中多个元素
                standard_batch.extend(e)
            # 打乱数据，是数据服从独立同分布（排除数据间的相关性）
            shuffle(standard_batch)

            # 这里保存的是一个temp也就是一直保存着最近一次的网络，这里是为了和最新的网络进行对弈
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

            # 开启训练
            self.nnet.train(standard_batch)

            print('PITTING AGAINST PREVIOUS VERSION')
            # 旧、新网路赢的次数 和 平局
            pwins, nwins, draws = 10, 100, 1
            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            # 如果旧网路和新网路赢得和为0 或 新网络/ 新网络＋旧网路 小于 更新阈值（0.55）则不更新，否则更新成新网络参数
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                # 如果拒绝了新模型，这老模型就能发挥作用
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                # 保存当前模型并更新最新模型
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    # 完整下一盘游戏
    def play_one_game(self):
        """
        使用Mcts完整下一盘棋
        :return: 4 * [(board, pi, z)] : 返回四个训练数据元组：（棋盘，策略，输赢）
        """
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


if __name__ == "__main__":
    game = Game(5)
    # train = TrainMode(game, nnet)
    game.board[4][4] = 10
    print(game.board)
    print(game.get_init_board(5))
    print(game.get_action_size())
    print(args.numIters)
