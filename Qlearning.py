import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 世界高度
WORLD_HEIGHT = 7

# 世界宽度
WORLD_WIDTH = 10

# 每列风速
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# 可能的行动
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

# ε的值
EPSILON = 0.1

# 步长
ALPHA = 0.5

# 每一步的奖励R
REWARD = -1.0

# 起止位置以及动作列表
START = [3, 0]
GOAL = [3, 7]
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

# 根据state以及action返回执行动作后的state
def step(state, action):
    i, j = state
    if action == ACTION_UP:
        return [max(i - 1 - WIND[j], 0), j]
    elif action == ACTION_DOWN:
        return [max(min(i + 1 - WIND[j], WORLD_HEIGHT - 1), 0), j]
    elif action == ACTION_LEFT:
        return [max(i - WIND[j], 0), max(j - 1, 0)]
    elif action == ACTION_RIGHT:
        return [max(i - WIND[j], 0), min(j + 1, WORLD_WIDTH - 1)]
    else:
        assert False

# 玩一把
def episode(q_value):
    # 追踪每把步数
    time = 0

    # 初始化位置
    state = START
    # 如果不是最终位置
    while state != GOAL:
    # 基于ε-贪婪法选择一个动作
        if np.random.binomial(1, EPSILON) == 1:
            action = np.random.choice(ACTIONS)
        else:
            values_ = q_value[state[0], state[1], :]
            action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])
        # 执行动作并获得下一个状态
        next_state = step(state, action)
        # 使用贪婪法选择下一个动作
        values_ = q_value[next_state[0], next_state[1], :]
        next_action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

        # 更新动作价值函数
        q_value[state[0], state[1], action] += \
            ALPHA * (REWARD + q_value[next_state[0], next_state[1], next_action] -
                     q_value[state[0], state[1], action])
        state = next_state
        time += 1
    return time


# 其余部分与SARSA一致
def q_learning():
    q_value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
    episode_limit = 500

    steps = []
    ep = 0
    while ep < episode_limit:
        steps.append(episode(q_value))
        ep += 1

    steps = np.add.accumulate(steps)

    plt.plot(steps, np.arange(1, len(steps) + 1))
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')

    plt.savefig('./q-learning.png')
    plt.close()

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
    q_learning()