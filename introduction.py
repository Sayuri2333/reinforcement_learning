import numpy as np
import pickle

BOARD_ROWS = 3  # 行数
BOARD_COLS = 3  # 列数
BOARD_SIZE = BOARD_ROWS * BOARD_COLS  # 棋盘大小


class State:
    def __init__(self):
        # 棋盘由n*n的数组表示
        # 取值为1表示玩家1的动作
        # 取值为-1表示玩家2的动作
        # 取值为0表示空
        self.data = np.zeros((BOARD_ROWS, BOARD_COLS))  # 生成初始棋盘
        self.winner = None
        self.hash_val = None
        self.end = None

    # 根据当前状态计算hash值 每个状态是唯一的
    def hash(self):
        if self.hash_val is None:  # 如果没有当前状态
            self.hash_val = 0  # 初始化当前状态为0
            for i in self.data.reshape(BOARD_ROWS * BOARD_COLS):  # 将二维数组降维 并把其中的-1改为2
                if i == -1:
                    i = 2
                self.hash_val = self.hash_val * 3 + i  # 根据当前格子取值计算hash值
        return int(self.hash_val)  # 返回当前状态表示的hash值

    # 检查是否有玩家胜利 或者是平局
    def is_end(self):
        if self.end is not None:
            return self.end  # 如果end取值不为空 表示已经有结果 此时返回值
        results = []
        # 检查行
        for i in range(0, BOARD_ROWS):
            results.append(np.sum(self.data[i, :]))  # 将每一行的和添加到结果中
        # 检查列
        for i in range(0, BOARD_COLS):
            results.append(np.sum(self.data[:, i]))  # 将每一列的和添加到结果中

        # 检查对角线
        results.append(0)
        for i in range(0, BOARD_ROWS):
            results[-1] += self.data[i, i] # 添加左对角线的和
        results.append(0)
        for i in range(0, BOARD_ROWS):
            results[-1] += self.data[i, BOARD_ROWS - 1 - i]  # 添加右对角线的和

        for result in results:  # 如果上述值中有出现3或者-3 则判定一方胜利
            if result == 3:
                self.winner = 1
                self.end = True
                return self.end
            if result == -3:
                self.winner = -1
                self.end = True
                return self.end

        # 判定是不是和局
        sum = np.sum(np.abs(self.data))  # 对棋盘中的值取绝对值
        if sum == BOARD_ROWS * BOARD_COLS:  # 如果和等于棋盘大小 则为和局
            self.winner = 0
            self.end = True
            return self.end

        # 否则游戏不终止
        self.end = False
        return self.end

    # @symbol: 1 or -1
    # 将symbol值放入(i,j)位置 并更新状态
    def next_state(self, i, j, symbol):
        new_state = State()
        new_state.data = np.copy(self.data)
        new_state.data[i, j] = symbol
        return new_state

    # 打印棋盘
    def print(self):
        for i in range(0, BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.data[i, j] == 1:
                    token = '*'
                if self.data[i, j] == 0:
                    token = '0'
                if self.data[i, j] == -1:
                    token = 'x'
                out += token + ' | '
            print(out)
        print('-------------')


# 给定当前状态以及下一步玩家 使用all_state存储当前状态之后会出现的所有状态、对应hash值以及isEnd值
def get_all_states_impl(current_state, current_symbol, all_states):
    for i in range(0, BOARD_ROWS):
        for j in range(0, BOARD_COLS):
            if current_state.data[i][j] == 0:
                newState = current_state.next_state(i, j, current_symbol)
                newHash = newState.hash()
                if newHash not in all_states.keys():
                    isEnd = newState.is_end()
                    all_states[newHash] = (newState, isEnd)  # 用hash来做key（唯一标记）
                    if not isEnd:
                        get_all_states_impl(newState, -current_symbol, all_states)


# 获取所有状态
def get_all_states():
    current_symbol = 1
    current_state = State()
    all_states = dict()
    all_states[current_state.hash()] = (current_state, current_state.is_end())
    get_all_states_impl(current_state, current_symbol, all_states)
    return all_states


# 存储所有状态
all_states = get_all_states()


class Judger:
    # @player1: the player who will move first, its chessman will be 1
    # @player2: another player with a chessman -1
    # @feedback: if True, both players will receive rewards when game is end
    def __init__(self, player1, player2):
        self.p1 = player1  # 设置玩家
        self.p2 = player2
        self.current_player = None
        self.p1_symbol = 1  # 分配先后
        self.p2_symbol = -1
        self.p1.set_symbol(self.p1_symbol)  # 根据先后顺序两个player更新estimation值
        self.p2.set_symbol(self.p2_symbol)
        self.current_state = State()  # 初始化状态为当前状态

    def reset(self):
        self.p1.reset()  # 重置state轨迹以及greedy轨迹 但不重置estimation值
        self.p2.reset()

    def alternate(self):  # 这是一个用来循环返回p1 p2的迭代器
        while True:
            yield self.p1
            yield self.p2

    # @print: if True, print each board during the game
    def play(self, print=False):
        alternator = self.alternate()  # 玩家选择器初始化
        self.reset()  # 初始化两个玩家的state以及greedy轨迹
        current_state = State()  # 初始化状态
        self.p1.set_state(current_state)  # 将状态告知玩家
        self.p2.set_state(current_state)
        while True:
            player = next(alternator)  # 取下一个玩家
            if print:
                current_state.print()
            [i, j, symbol] = player.act()  # 根据当前状态选择一个动作
            next_state_hash = current_state.next_state(i, j, symbol).hash()  # 进入下一个状态
            current_state, is_end = all_states[next_state_hash]  # 将当前状态设置为下一个状态 并查看是否isEnd
            self.p1.set_state(current_state)  # 将状态告知玩家
            self.p2.set_state(current_state)
            if is_end:  # 如果结束了 返回胜利的玩家
                if print:
                    current_state.print()
                return current_state.winner


# AI player
class Player:
    # @step_size: 更新预测的步长
    # @epsilon: 进行探索的概率
    def __init__(self, step_size=0.1, epsilon=0.1):  # 初始化
        self.estimations = dict()  # 用来存储每个状态对应的value值
        self.step_size = step_size
        self.epsilon = epsilon
        self.states = []
        self.greedy = []

    def reset(self):  # 重置states以及greedy方法
        self.states = []
        self.greedy = []

    def set_state(self, state):  # 添加states以及对应是否greedy的值
        self.states.append(state)
        self.greedy.append(True)

    def set_symbol(self, symbol):  # 告诉AI player它是哪一个玩家 从而更新estimations中的value值
        self.symbol = symbol
        for hash_val in all_states.keys():  # 注意estimations中存储的状态包括了可能出现的所有状态
            (state, is_end) = all_states[hash_val]
            if is_end:  # 如果hash_val对应的状态是最后状态
                if state.winner == self.symbol:  # 根据赢家调整estimations的对应值
                    self.estimations[hash_val] = 1.0
                elif state.winner == 0:
                    # we need to distinguish between a tie and a lose
                    self.estimations[hash_val] = 0.5
                else:
                    self.estimations[hash_val] = 0
            else:  # 如果不是最终状态 则调整为0.5
                self.estimations[hash_val] = 0.5

    # 更新estimation的预测值
    def backup(self):
        # for debug
        # print('player trajectory')
        # for state in self.states:
        #     state.print()

        self.states = [state.hash() for state in self.states]  # 将states中的state对象变成对应hash值

        for i in reversed(range(len(self.states) - 1)):  # 对于states数组 从倒数第二个往前
            state = self.states[i]
            td_error = self.greedy[i] * (self.estimations[self.states[i + 1]] - self.estimations[state])  # 计算当前状态与下一个状态之间的value差值
            self.estimations[state] += self.step_size * td_error  # 当前状态的value值考虑到了当前状态以及下一个状态提供的value值变化

    # 根据当前状态选择一个动作
    def act(self):
        state = self.states[-1]  # 获得当前状态
        next_states = []
        next_positions = []
        for i in range(BOARD_ROWS):  # 根据当前状态 挑选可选动作 并存储动作以及对应状态的hash值
            for j in range(BOARD_COLS):
                if state.data[i, j] == 0:
                    next_positions.append([i, j])
                    next_states.append(state.next_state(i, j, self.symbol).hash())

        if np.random.rand() < self.epsilon:  # 使用epsilon-greedy法挑选动作 小于时随机挑选动作
            action = next_positions[np.random.randint(len(next_positions))]  # 随机挑选动作
            action.append(self.symbol)  # 在动作尾部添加symbol
            self.greedy[-1] = False  # 此时并没有greedy
            return action

        values = []
        # 下述操作用来选出value最高的动作
        for hash, pos in zip(next_states, next_positions):
            values.append((self.estimations[hash], pos))
        np.random.shuffle(values)
        values.sort(key=lambda x: x[0], reverse=True)
        action = values[0][1]
        action.append(self.symbol)
        return action

    def save_policy(self): # 存储estimations
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'wb') as f:
            pickle.dump(self.estimations, f)

    def load_policy(self):  # 读取estimations
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'rb') as f:
            self.estimations = pickle.load(f)


# human interface
# input a number to put a chessman
# | q | w | e |
# | a | s | d |
# | z | x | c |
class HumanPlayer:
    def __init__(self, **kwargs):
        self.symbol = None
        self.keys = ['q', 'w', 'e', 'a', 's', 'd', 'z', 'x', 'c']
        self.state = None
        return

    def reset(self):
        return

    def set_state(self, state):  # 设置state
        self.state = state

    def set_symbol(self, symbol):  # 设置symbol（先后）
        self.symbol = symbol
        return

    def backup(self, _):
        return

    def act(self):  # 根据输入值采取动作
        self.state.print()
        key = input("Input your position:")
        data = self.keys.index(key)
        i = data // int(BOARD_COLS)
        j = data % BOARD_COLS
        return (i, j, self.symbol)  # 动作尾部添加symbol


def train(epochs):  # 使用两个AI player进行训练  epoch为训练轮数
    player1 = Player(epsilon=0.01)  # 设置不贪婪概率
    player2 = Player(epsilon=0.01)
    judger = Judger(player1, player2)
    player1_win = 0.0  # 用来统计两个玩家赢的次数
    player2_win = 0.0
    for i in range(1, epochs + 1):
        winner = judger.play(print=False)  # 用judger玩一把
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        print('Epoch %d, player 1 win %.02f, player 2 win %.02f' % (i, player1_win / i, player2_win / i))
        player1.backup()  # 玩完之后改进value值
        player2.backup()
        judger.reset()  # 重置player
    player1.save_policy()  # 迭代完成后存储estimation
    player2.save_policy()


def compete(turns):  # 正式来几把 不包含backup的改进操作
    player1 = Player(epsilon=0)
    player2 = Player(epsilon=0)
    judger = Judger(player1, player2)
    player1.load_policy()
    player2.load_policy()
    player1_win = 0.0
    player2_win = 0.0
    for i in range(0, turns):
        winner = judger.play()
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        judger.reset()
    print('%d turns, player 1 win %.02f, player 2 win %.02f' % (turns, player1_win / turns, player2_win / turns))


# The game is a zero sum game. If both players are playing with an optimal strategy, every game will end in a tie.
# So we test whether the AI can guarantee at least a tie if it goes second.
def play():  # 人类玩家和电脑来几把
    while True:
        player1 = HumanPlayer()
        player2 = Player(epsilon=0)
        judger = Judger(player1, player2)
        player2.load_policy()  # 电脑玩家用训练好的policy
        winner = judger.play()
        if winner == player2.symbol:
            print("You lose!")
        elif winner == player1.symbol:
            print("You win!")
        else:
            print("It is a tie!")


if __name__ == '__main__':
    train(int(1e5))
    compete(int(1e3))
    play()
