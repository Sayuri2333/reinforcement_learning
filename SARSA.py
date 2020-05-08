import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORLD_HEIGHT = 7  # 世界高度

WORLD_WIDTH = 10  # 世界宽度

# 每一列风的强度 风是从下往上吹的
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# 个体动作编号
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

# epsilon的大小（探索概率）
EPSILON = 0.1

# SARSA算法的初始步长
ALPHA = 0.5

# 每步的reward（只要不是到终点都是-1）
REWARD = -1.0

# 起始位置和终点位置
START = [3, 0]
GOAL = [3, 7]
# 操作列表
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]


# 根据state（当前位置）以及action（当前行动）来判断行动后位置
def step(state, action):
    i, j = state
    if action == ACTION_UP:
        return [max(i - 1 - WIND[j], 0), j]  # -1代表向上一一格 -WIND[j]代表风向上吹的格数 max保证个体不会移动到世界外部
    elif action == ACTION_DOWN:
        return [max(min(i + 1 - WIND[j], WORLD_HEIGHT - 1), 0), j]  # 内层min保证向下时不会突破底部 max保证被风吹的时候不会突破顶部
    elif action == ACTION_LEFT:
        return [max(i - WIND[j], 0), max(j - 1, 0)]  # 左右移动时被风吹的还是原来的格子的风
    elif action == ACTION_RIGHT:
        return [max(i - WIND[j], 0), min(j + 1, WORLD_WIDTH - 1)]
    else:
        assert False


# 玩一把
def episode(q_value):
    # 记录整个流程到底花了多少步
    time = 0

    # 1.初始化S为当前序列的第一个状态，设置A为ε-贪婪法在当前状态选择的动作
    # 初始化位置
    state = START

    # 使用ε-贪婪法选择一个动作
    if np.random.binomial(1, EPSILON) == 1:
        action = np.random.choice(ACTIONS)
    else:
        values_ = q_value[state[0], state[1], :]  # 根据当前的state取出各个动作对应的value
        action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])  # 根据各个动作的value选择value最高的动作

    # 循环至抵达终点
    while state != GOAL:
        # 2.在状态S执行当前动作A，得到新状态S'以及奖励R
        next_state = step(state, action)  # 使用step走一步 并获得后面的位置
        # 3.用ε-贪婪法在状态S'选择新的动作A'
        # 这部分和上面的一样
        if np.random.binomial(1, EPSILON) == 1:
            next_action = np.random.choice(ACTIONS)
        else:
            values_ = q_value[next_state[0], next_state[1], :]
            next_action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

        # 4.更新价值函数Q(S,A）
        q_value[state[0], state[1], action] += \
            ALPHA * (REWARD + q_value[next_state[0], next_state[1], next_action] -
                     q_value[state[0], state[1], action])
        # 5.S=S', A=A'
        state = next_state
        action = next_action
        time += 1
    return time


def sarsa():
    # q_value用来存储每个状态对应四个动作的value
    q_value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
    # 最大把数
    episode_limit = 500

    # 统计每把走了多少步的数组
    steps = []
    ep = 0
    while ep < episode_limit:
        steps.append(episode(q_value))
        # time = episode(q_value)
        # episodes.extend([ep] * time)
        ep += 1

    steps = np.add.accumulate(steps)  # 统计每个循环结束后一共走了的总步数

    plt.plot(steps, np.arange(1, len(steps) + 1))  # 画步数随episode分布图
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')

    plt.savefig('./sarsa.png')
    plt.close()

    # 显示每个位置对应的最佳动作矩阵
    optimal_policy = []
    for i in range(0, WORLD_HEIGHT):
        optimal_policy.append([])
        for j in range(0, WORLD_WIDTH):
            if [i, j] == GOAL:
                optimal_policy[-1].append('G')
                continue
            bestAction = np.argmax(q_value[i, j, :])
            if bestAction == ACTION_UP:
                optimal_policy[-1].append('U')
            elif bestAction == ACTION_DOWN:
                optimal_policy[-1].append('D')
            elif bestAction == ACTION_LEFT:
                optimal_policy[-1].append('L')
            elif bestAction == ACTION_RIGHT:
                optimal_policy[-1].append('R')
    print('Optimal policy is:')
    for row in optimal_policy:
        print(row)
    print('Wind strength for each column:\n{}'.format([str(w) for w in WIND]))

if __name__ == '__main__':
    sarsa()