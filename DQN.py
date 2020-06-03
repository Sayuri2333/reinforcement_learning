import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque

# DQN超参数
GAMMA = 0.9  #
INITIAL_EPSILON = 0.5  # ε的初始值
FINAL_EPSILON = 0.01  # ε的最终值
REPLAY_SIZE = 10000  # 经验回放池的大小
BATCH_SIZE = 32  # 每次提取的经验的多少


class DQN():
    # DQN Agent
    def __init__(self, env):
        # init experience replay
        self.replay_buffer = deque()  # 使用deque构筑经验回放池
        # init some parameters
        self.time_step = 0  # 步数
        self.epsilon = INITIAL_EPSILON  # 初始化ε
        self.state_dim = env.observation_space.shape[0]  # 状态空间
        self.action_dim = env.action_space.n  # 动作空间

        self.create_Q_network()  # 初始化Q网络
        self.create_training_method()  # 初始化训练方法

        # 初始化session
        self.session = tf.InteractiveSession()  # 使用tf.InteractiveSession能允许在进程启动后构建operation
        self.session.run(tf.global_variables_initializer())  # 初始化全部变量

    def create_Q_network(self):
        # network weights
        W1 = self.weight_variable([self.state_dim, 20])  # 第一层为self.state_dim -> 20
        b1 = self.bias_variable([20])
        W2 = self.weight_variable([20, self.action_dim])  # 第二层 20 -> self.action_dim 预计输出对应动作的Q-value
        b2 = self.bias_variable([self.action_dim])
        # input layer
        self.state_input = tf.placeholder("float", [None, self.state_dim])  # 输入变量，None行是因为需要miniBatch处理
        # hidden layers
        h_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)  # 第一层到第二层需要经过relu激活函数
        # Q Value layer
        self.Q_value = tf.matmul(h_layer, W2) + b2  # 输出Q-value

    def create_training_method(self):
        self.action_input = tf.placeholder("float", [None, self.action_dim])  # 动作输入，是one-hot向量
        self.y_input = tf.placeholder("float", [None])  # 更新后的y值
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)  # 对应action的Q值
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))  # 计算loss
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)  # 使用Adam法最小化loss

    def perceive(self, state, action, reward, next_state, done):  # 接收新一组经验，并判断是否需要学习
        one_hot_action = np.zeros(self.action_dim)  # 生成全0的动作向量
        one_hot_action[action] = 1  # 根据接收的action位置生成action的one-hot向量
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))  # 将经验加入经验回放池
        if len(self.replay_buffer) > REPLAY_SIZE:  # 经验回放池已满
            self.replay_buffer.popleft()  # 将最以前的经验弹出

        if len(self.replay_buffer) > BATCH_SIZE:  # 如果经验数超过了miniBatch的数量
            self.train_Q_network()  # 训练网络

    def train_Q_network(self):
        self.time_step += 1
        # 从经验回放池中随机抽选一个minibatch
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # 计算每组经验对应的y值
        y_batch = []
        # 将next_state_batch送入神经网络，得出对应的Q_value_batch。 需要这个来计算y值
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})
        for i in range(0, BATCH_SIZE):  # 对于batch中的每组经验
            done = minibatch[i][4]  # 根据是否为结尾状态
            if done:
                y_batch.append(reward_batch[i])  # 更新y值
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
            self.y_input: y_batch,  # y_batch用来计算loss
            self.action_input: action_batch,  # action_batch用来计算Q_action
            self.state_input: state_batch  # state_input用来计算Q_value
        })

    def egreedy_action(self, state):  # ε贪婪法
        Q_value = self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0]  # 将state输入神经网络，获得每个action对应Q_value
        if random.random() <= self.epsilon:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000  # ε收敛
            return random.randint(0, self.action_dim - 1)  # 随便返回一个action
        else:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return np.argmax(Q_value)  # 返回Q_value最大值动作

    def action(self, state):  # 完全贪婪法
        return np.argmax(self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0])  # 直接返回Q_value最大的动作

    def weight_variable(self, shape):  # 根据shape创建并初始化权重
        initial = tf.truncated_normal(shape)  # tf.truncated_normal生成shape形状的随机数数组
        return tf.Variable(initial)  # 根据initial生成变量

    def bias_variable(self, shape):  # 根据shape创建并初始化偏置
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)


# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 3000  # 最大循环次数
STEP = 300  # 每次循环最大步数
TEST = 10  # 每100次循环后测试次数


def main():
    # 初始化环境以及DQN
    env = gym.make(ENV_NAME)
    agent = DQN(env)

    for episode in range(EPISODE):
        # 初始化环境 并获得初始状态
        state = env.reset()
        # 训练环节
        for step in range(STEP):
            action = agent.egreedy_action(state)  # ε贪婪法选择动作
            next_state, reward, done, _ = env.step(action)  # 执行动作
            # 只要不死 每一帧加0.1reward
            reward = -1 if done else 0.1
            agent.perceive(state, action, reward, next_state, done)  # 加入经验回放池并训练
            state = next_state  # 下一个状态
            if done:
                break
        # 每100次循环进行一次test
        if episode % 100 == 0:
            total_reward = 0  # 初始化总reward
            for i in range(TEST):  # 测试TEST次
                state = env.reset()  # 初始化状态
                for j in range(STEP):
                    env.render()  # 渲染下一帧环境
                    action = agent.action(state)  # 贪婪法选择动作
                    state, reward, done, _ = env.step(action)  # 执行动作
                    total_reward += reward  # 累加reward
                    if done:
                        break
            ave_reward = total_reward / TEST  # 计算平均reward
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)


if __name__ == '__main__':
    main()
