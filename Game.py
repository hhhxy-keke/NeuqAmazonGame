# coding=utf-8
import numpy as np

BLACK = -2
WHITE = 2
EMPTY = 0
ARROW = 1


class Game:
    directions = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]

    def __init__(self, board_size):
        """
        :param board_size: int:棋盘大小
        :return None
        """
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        # 黑棋
        self.board[0][board_size // 3] = BLACK
        self.board[0][2 * board_size // 3] = BLACK
        self.board[board_size // 3][0] = BLACK
        self.board[board_size // 3][board_size - 1] = BLACK
        # 白棋
        self.board[2 * board_size // 3][0] = WHITE
        self.board[2 * board_size // 3][board_size - 1] = WHITE
        self.board[board_size - 1][board_size // 3] = WHITE
        self.board[board_size - 1][2 * board_size // 3] = WHITE

    def get_init_board(self):
        """
        :return b.board: 返回board_size*board_size的棋盘的二维numpy数组
        """
        return self.board

    def get_board_size(self):
        """
        :return (self.board_size, self.board_size):棋盘大小的元组
        """
        return self.board_size, self.board_size

    def get_action_size(self):
        """
        :return 3 * self.board_size * self.board_size:int:返回所有的动作空间数
        """
        return 3 * self.board_size ** 2

    def is_legal_move(self, start, end):

        """
        判断start—>end是否可走
        :param start: 起点
        :param end: 落点
        :return: boolean:可走返回True else False
        """
        return True or False

    # 更新选点,落点,放箭棋盘 ——>返回:新棋盘和下次走棋方 -----:该方法没有判别起始点、落子和放箭是否符合规则的
    def get_next_state(self, board, player, action):
        """
        获得下一个状态
        :param board: n*n棋盘
        :param player:int:当前玩家
        :param action:三元组eg(67,78,99): 起始坐标,放皇后坐标,箭坐标
        :return board, player
                board:当前棋盘
                player:当前玩家:BLACK or WHITE
        """
        return board, WHITE if player == BLACK else BLACK

    def get_valid_actions(self, board, player, ps):
        """
        计算当前棋盘下所有可走的动作，同时将NN返回的预测值重新整理:将绝可能到的点概率值为零
        :param board: n*n棋盘
        :param player:int:当前玩家
        :param Ps:3 * board_size ** 2列表:使用神经网络直接预测的概率值
        :return: Ps, valids
                 Ps:整理后的3 * board_size ** 2列表;
                 valids:三元组(s,e,a)构成的列表:当前棋盘下的所有可走动作
                 Ps -->1 * 300：[0.2， 0.02， 0.23，......]   动作：valids -->[(s, e, a),......]

        """
        valids = []
        return ps, valids

    def get_game_ended(self, board, player):
        """
        返回游戏状态：输：-1/赢：1/未结束：0
        :param board: n*n棋盘
        :param player: 当前玩家
        :return: 0 or 1 or -1: 0—>未能判断输赢；1—>player赢；-1—>player输
        """
        # 这部分和原始程序不一样，需要自己写
        # 编程思路即对棋盘上的每方四个皇后分别判断是否可走，当某一方四个皇后均不可走时即输
        # 具体的过程交给你们了！

        return 0 or 1 or -1

    def get_symmetries(self, board, pi):
        """
        将局面和策略顺时针旋转180度，返回4个棋盘与策略的元组、目的是增加训练神经网络时的数据量
        :param board: n*n棋盘
        :param pi: 3 * board_size ** 2列表:策略向量
        :return: board_list:4个(棋盘, 策略)的元组
        """
        pi_board_start = np.reshape(pi[0:self.board_size**2], (self.board_size, self.board_size))
        pi_board_end = np.reshape(pi[self.board_size**2:2 * self.board_size**2], (self.board_size, self.board_size))
        pi_board_arrow = np.reshape(pi[2 * self.board_size**2:3 * self.board_size**2], (self.board_size, self.board_size))
        board_list = []
        board_list += [(board, pi)]  # 每张棋盘和一个1*300的策略数组,组成一个元组

        # 将棋盘和策略向量二维数组均左右翻转
        newB = np.fliplr(board)
        newPi_start = np.fliplr(pi_board_start)
        newPi_end = np.fliplr(pi_board_end)
        newPi_arrow = np.fliplr(pi_board_arrow)
        newPi = newPi_start
        newPi = np.append(newPi, newPi_end)
        newPi = np.append(newPi, newPi_arrow)
        board_list += [(newB, list(newPi.ravel()))]  # ravel()：多维数组转换为一维数组的功能

        # 将棋盘和策略向量在第一个维度上进行倒序.
        newB = np.flipud(board)
        newPi_start = np.flipud(pi_board_start)
        newPi_end = np.flipud(pi_board_end)
        newPi_arrow = np.flipud(pi_board_arrow)
        newPi = newPi_start
        newPi = np.append(newPi, newPi_end)
        newPi = np.append(newPi, newPi_arrow)
        board_list += [(newB, list(newPi.ravel()))]

        # 将棋盘和策略向量二维数组再左右翻转
        newB = np.fliplr(newB)
        newPi_start = np.fliplr(newPi_start)
        newPi_end = np.fliplr(newPi_end)
        newPi_arrow = np.fliplr(newPi_arrow)
        newPi = newPi_start
        newPi = np.append(newPi, newPi_end)
        newPi = np.append(newPi, newPi_arrow)
        board_list += [(newB, list(newPi.ravel()))]
        return board_list

    def to_string(self, board):
        """
        将棋盘转换成字符串，为后面使用棋盘做字典的key做准备
        :param board: n*n棋盘
        :return: str字符串
        """
        return board.tostring()


# 测试使用
game = Game(5)
b = np.copy(game.board)
print(b)
# print(game.get_action_size())
# u = np.zeros(75, dtype=int)
# print(u)
# y = game.get_symmetries(b, u)
# for b, p in y:
#     print(b,p)
