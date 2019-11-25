from Game import Game

# 训练模式的参数
args = dict({
    'numIters': 1000,
    'numEps': 1,
    'tempThreshold': 35,    # 探索效率
    'updateThreshold': 0.55,
    'maxlenOfQueue': 200000,
    'num_mcts_search': 50,      # 从当前状态搜索到一个未被扩展的叶结点25次
    'arenaCompare': 40,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/models/','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})


if __name__=="__main__":
    g = Game(5)
