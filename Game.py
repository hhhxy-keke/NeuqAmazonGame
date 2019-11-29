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
        self.get_init_board(board_size)

    def get_init_board(self, board_size):
        """
        :return b.board: 返回board_size*board_size的棋盘的二维numpy数组
        """
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

    def is_legal_move(self, board, start, end):
        """
        判断start—>end是否可走
        :param board: 当前棋盘
        :param start: 起点
        :param end: 落点
        :return: boolean:可走返回True else False
        """
        sx = start // self.board_size  # 起点x
        sy = start % self.board_size  # 起点y
        ex = end // self.board_size  # 落点x
        ey = end % self.board_size  # 落点y
        # print(sx,sy,ex,ey)
        # 先判断是否沿左右、上下、米子方向走，沿则可能，不沿则绝不可能直接False
        if ex == sx or ey == sy or abs(ex - sx) == abs(ey - sy):
            tx = (ex - sx) // max(1, abs(ex - sx))  # +1：向右  -1：向左
            ty = (ey - sy) // max(1, abs(ey - sy))  # +1：向上  -1：向下
            t_start = -1  #
            t_end = end  #
            # 之后从start一步一步走向end，判断是否有障碍直到end
            while sx != ex or sy != ey:
                sx += tx
                sy += ty
                t_start = sx * self.board_size + sy
                if board[sx][sy] != EMPTY:  # 没走到end且遇到障碍
                    break
            if t_start == t_end:  # 如果是因为顺利走到end而break，则可以走
                if board[sx][sy] != EMPTY:
                    return False
                return True
            else:  # 如果是因为有障碍而break，则不能走
                return False
        else:
            return False

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
        b = board.copy()
        start_x, start_y = action[0] // self.board_size, action[0] % self.board_size
        end_x, end_y = action[1] // self.board_size, action[1] % self.board_size
        arrow_x, arrow_y = action[2] // self.board_size, action[2] % self.board_size

        if b[start_x][start_y] != player:
            print("Game-get_next_state: Error Start!", start_x, start_y)
        else:
            b[start_x][start_y] = EMPTY
        if b[end_x][end_y] != EMPTY:
            print("Game-get_next_state: Error End", start_x, start_y)
        else:
            b[end_x][end_y] = player
        if b[arrow_x][arrow_y] != EMPTY:
            print("Game-get_next_state: Error arrow", start_x, start_y)
        else:
            b[arrow_x][arrow_y] = ARROW

        if player == WHITE:
            player = BLACK
        else:
            player = WHITE

        return b, player

    def get_valid_actions(self, board, player, ps):
        """
        计算当前棋盘下所有可走的动作，同时将NN返回的预测值重新整理:将绝可能到的点概率值为零
        :param board: n*n棋盘
        :param player:int:当前玩家
        :param ps:3 * board_size ** 2列表:使用神经网络直接预测的概率值
        :return: Ps, all_valid_action
                 Ps:整理后的3 * board_size ** 2列表;
                 all_valid_action:三元组(s,e,a)构成的列表:当前棋盘下的所有可走动作
                 Ps -->1 * 300：[0.2， 0.02， 0.23，......]   动作：all_valid_action -->[(s, e, a),......]
        """
        b = board.copy()
        size = self.board_size  # 棋盘的尺寸

        ps_start = ps[0:size ** 2]
        ps_end = ps[size ** 2:2 * size ** 2]
        ps_arrow = ps[2 * size ** 2:3 * size ** 2]

        valid_start = np.zeros(size ** 2, dtype=int)
        valid_end = np.zeros(size ** 2, dtype=int)
        valid_arrow = np.zeros(size ** 2, dtype=int)

        all_valid_action = []  # 储存合法的步伐
        for s in range(size ** 2):  # 挑选出合法的步伐
            if b[s // size][s % size] == player:
                valid_start[s] = 1
                for e in range(size ** 2):
                    if self.is_legal_move(b, s, e):
                        valid_end[e] = 1
                        b[s // size][s % size] = EMPTY
                        b[e // size][e % size] = player
                        for a in range(size ** 2):
                            if self.is_legal_move(b, e, a):
                                valid_arrow[a] = 1
                                valid = (s, e, a)
                                all_valid_action.append(valid)
                        b[s // size][s % size] = player
                        b[e // size][e % size] = EMPTY
        for s in range(size ** 2):
            if valid_start[s] == 0:
                ps_start[s] = 0
        sum = np.sum(ps_start)
        if sum > 0:
            ps_start /= sum
        else:
            print("All start moves were masked, do workaround.")
            # ps_start[valid_start == 1] = 0.25
            for s in range(size ** 2):
                if valid_start[s] == 1:
                    ps_start[s] = 0.25

        for e in range(size ** 2):
            if valid_end[e] == 0:
                ps_end[e] = 0
        sum = np.sum(ps_end)
        if sum > 0:
            ps_end /= sum
        else:
            print("All end moves were masked, do workaround.")

        for a in range(size ** 2):
            if valid_arrow[a] == 0:
                ps_arrow[a] = 0
        sum = np.sum(ps_arrow)
        if sum > 0:
            ps_arrow /= sum
        else:
            print("All arrow moves were masked, do workaround.")

        ps[0:size ** 2] = ps_start
        ps[size ** 2:2 * size ** 2] = ps_end
        ps[2 * size ** 2:3 * size ** 2] = ps_arrow

        all_valid_action = np.array(all_valid_action)  # 转成数组
        return ps, all_valid_action

    def get_game_ended(self, board, player):
        """
        判断游戏轮到player走棋时否结束
        :param board: n*n棋盘
        :param player: 当前轮到哪位该玩家走棋
        :return: 0 or -1: 0—>未能判断输赢；-1—>player输 ：判断不了赢的情况（提前判断对手不能走有时是不准确的，可能你走完对手又能走了）
        """
        size = self.board_size
        # 记录player可走的棋子数量
        count_player = 0
        # 记录player对手可走的棋子数量
        # count_opposite_player = 0
        for i in range(size):
            for j in range(size):
                if board[i][j] == player:
                    # 对八个方向判断是否可走
                    for k in range(8):
                        m = i + self.directions[k][0]
                        n = j + self.directions[k][1]
                        if m not in range(size) or n not in range(size):
                            continue
                        if board[m][n] == EMPTY:
                            count_player += 1
                            break
                # if board[i][j] == -player:
                #     # 对八个方向判断是否可走
                #     for k in range(8):
                #         m = i + self.directions[k][0]
                #         n = j + self.directions[k][1]
                #         if m not in range(size) or n not in range(size):
                #             continue
                #         if board[m][n] == EMPTY:
                #             count_opposite_player += 1
                #             break
        # 如果当前玩家可走的棋子数为0，则当前棋子 输
        if count_player == 0:
            return -1
        # # 如果当前对手可走的棋子数为0，则当前棋子 赢
        # if count_opposite_player == 0:
        #     return 1
        return 0

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

    def get_transformed_board(self, board, player):
        """
        自博弈时将智能体始终转为白棋视角,在MCTS搜索存储各状态属性时使用
        :param board: n*n棋盘
        :param player: 当前玩家
        :return board: 转换后的棋盘
        """
        # 深拷贝
        if player == WHITE:
            return board
        board = np.copy(board)
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == WHITE:
                    board[i][j] = BLACK
                elif board[i][j] == BLACK:
                    board[i][j] = WHITE
        return board

    def to_string(self, board):
        """
        将棋盘转换成字符串，为后面使用棋盘做字典的key做准备
        :param board: n*n棋盘
        :return: str字符串
        """
        return board.tostring()

