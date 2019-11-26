import numpy as np
import math

EPS = 1e-8
BLACK = -2
WHITE = 2
EMPTY = 0
ARROW = 1


class Mcts:
    """
    蒙特卡洛树搜索类：对给定棋盘状态始终使用“白棋视角”搜索得到下一步最优行动
    """
    def __init__(self, game, nnet, args):
        """
        :param game: 当前棋盘对象
        :param nnet: 神经网络
        :param args: 训练参数
        """
        self.game = game
        self.nnet = nnet
        self.args = args
        self.episodeStep = 0

        self.Game_End = {}        # 输赢状态字典
        self.Actions = {}         # 某状态下所有可走的行动
        self.Pi = {}              # 行动时选点的概率 value: 3 * board_size 的一维列表
        self.N = {}               # 某状态的访问次数
        self.Nsa = {}             # 某状态s+动作a（下一状态）访问次数 == N[s+1]
        self.Qsa = {}             # 某状态s+动作a（下一个状态）的奖励值

        self.N_start = {}
        self.N_end = {}
        self.N_arrow = {}

    def get_best_action(self, board, player):
        """
        :param board:  当前棋盘
        :param player:  玩家
        :return best_action: 下一步最优动作
        """
        for i in range(self.args.num_mcts_search):
            self.search(board)

        # 这里将采样次数转化成对应的模拟概率
        s = self.game.to_string(board)
        # 如果一个动作存在Ns_start记录中，则将该点设为 Ns_start[(s, a)]，不存在时设为0 ////counts_start:[1:100]
        counts_start = [self.N_start[(s, a)] if (s, a) in self.N_start else 0 for a in range(self.game.board_size)]
        # softmax ，将整个1*100的起始点的采样数N转换成概率模式//[0，0，0，0.3，0，0，0，0.21，0..........]
        p_start = [x / float(self.N[s]) for x in counts_start]
        counts_end = [self.N_end[(s, a)] if (s, a) in self.N_end else 0 for a in range(self.game.board_size)]
        p_end = [x / float(self.N[s]) for x in counts_end]
        counts_arrow = [self.N_arrow[(s, a)] if (s, a) in self.N_arrow else 0 for a in range(self.game.board_size)]
        p_arrow = [x / float(self.N[s]) for x in counts_arrow]

        # 方法二：使用softmax策略选择动作
        pi = p_start
        pi = np.append(pi, p_end)
        pi = np.append(pi, p_arrow)

        # 将局面和策略顺时针旋转180度，返回4个棋盘和策略组成的元组
        steps_train_data = []
        sym = self.game.get_symmetries(board, pi)
        for boards, pis in sym:
            steps_train_data.append([boards, player, pis, None])
        pi_start = pi[0:self.game.board_size]
        pi_end = pi[self.game.board_size:2 * self.game.board_size]
        pi_arrow = pi[2 * self.game.board_size:3 * self.game.board_size]
        # 深拷贝
        copy_board = np.copy(board)
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
                copy_board[action_end // self.game.board_size][action_end % self.game.board_size] = self.player
                if self.game.is_legal_move(action_end, action_arrow):
                    best_action = [action_start, action_end, action_arrow]
                    break  # 跳出While True 循环
                else:
                    copy_board[action_start // self.game.board_size][action_start % self.game.board_size] = self.player
                    copy_board[action_end // self.game.board_size][action_end % self.game.board_size] = EMPTY

        return best_action, steps_train_data

    def search(self, board):
        """
        对状态进行一次递归的模拟搜索，添加各状态（棋盘）的访问结点信息（始终以白棋视角存储）
        :param board: 棋盘当前
        :param player: 当前玩家
        :return: None
        """
        board_key = self.game.to_string(board)
        # 判断是否胜负已分（叶子节点）
        if board_key not in self.Game_End:
            self.Game_End[board_key] = self.game.get_game_ended(board, WHITE)
        if self.Game_End[board_key] != 0:
            return -self.Game_End[board_key]

        # 判断board_key是否为新扩展的节点
        if board_key not in self.Pi:
            # 由神经网路预测策略与v([-1,1]) PS[s] 为[1:300]数组
            self.Pi[board_key], v = self.nnet.predict(board)
            # 得到在有效路径归一化后的P与动作；eg：P -->1 * 300：[0.2， 0.02， 0.23，......]   动作：legal_actions -->[[s, e, a], [98, 87, 56]......]
            # 始终寻找白棋可走的行动
            self.Pi[board_key], legal_actions = self.game.get_valid_actions(board, WHITE, self.Pi[board_key])
            # 存储该状态下所有可行动作
            self.Actions[board_key] = legal_actions
            self.N[board_key] = 0
            self.Qsa[board_key] = 0
            return -v
        legal_actions = self.Actions[board_key]
        best_uct = -float('inf')
        # 最好的行动
        best_action = -1
        psa = list()                  # 状态转移概率，长度为当前状态下可走的动作数

        # 将选点概率P转换成动作的概率
        for a in legal_actions:
            p = 0
            for i in [0, 1, 2]:
                assert self.Pi[board_key][a[i] + i * 100] > 0
                p += math.log(self.Pi[board_key][a[i] + i * 100])
            psa.append(p)
        psa = np.array(psa)
        psa = np.exp(psa) / sum(np.exp(psa))
        # 求置信上限函数：Q+c*p*((Ns/Nsa)的开方)
        for i, a in enumerate(legal_actions):              # enumerate():将一个元组加上序号，其中 i 为序号：0，1.... a为中的legal_actions元组
            if (board_key, a[0], a[1], a[2]) in self.Qsa:  # board_key:棋盘字符串，a[0], a[1], a[2]分别为起始点，落子点，放箭点
                uct = self.Qsa[(board_key, a[0], a[1], a[2])] + self.args.cpuct * psa[i] * math.sqrt(self.N[board_key])\
                      / (1 + self.Nsa[(board_key, a[0], a[1], a[2])])

            else:
                uct = self.args.cpuct * psa[i] * math.sqrt(self.N[board_key] + EPS)   # 防止乘积为0

            if uct > best_uct:
                best_uct = uct
                best_action = a

        a = best_action
        # next_player反转
        next_board, next_player = self.game.get_next_state(board, WHITE, a)
        # 下一个状态，将棋盘颜色反转 (next_player = BLACK)
        next_board = self.game.get_transformed_board(next_board, next_player)

        v = self.search(next_board)

        if (board_key, a[0], a[1], a[2]) in self.Qsa:
            self.Qsa[(board_key, a[0], a[1], a[2])] = (self.Nsa[(board_key, a[0], a[1], a[2])] * self.Qsa[(board_key, a[0], a[1], a[2])] + v) / (self.Nsa[(board_key, a[0], a[1], a[2])]+1)
            self.Nsa[(board_key, a[0], a[1], a[2])] += 1

        else:
            self.Qsa[(board_key, a[0], a[1], a[2])] = v
            self.Nsa[(board_key, a[0], a[1], a[2])] = 1

        if (board_key, a[0]) in self.N_start:
            self.N_start[(board_key, a[0])] += 1
        else:
            self.N_start[(board_key, a[0])] = 1

        if (board_key, a[1]) in self.N_end:
            self.N_end[(board_key, a[1])] += 1
        else:
            self.N_end[(board_key, a[1])] = 1

        if (board_key, a[2]) in self.N_arrow:
            self.N_arrow[(board_key, a[2])] += 1
        else:
            self.N_arrow[(board_key, a[2])] = 1

        self.N[board_key] += 1

        return -v
