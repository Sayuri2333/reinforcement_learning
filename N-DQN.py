import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque

# Hyper Parameters for DQN
GAMMA = 0.9  # 计算y值时γ的值
INITIAL_EPSILON = 0.5  # 初始ε值
FINAL_EPSILON = 0.01  # 最终ε值
REPLAY_SIZE = 10000  # 经验回放池大小
BATCH_SIZE = 32  # minibatch大小
REPLACE_TARGET_FREQ = 10  # 更新目标Q网络参数的频率


class DQN():
    # DQN Agent
    def __init__(self, env):
        # 初始化经验回放池
        self.replay_buffer = deque()
        # 初始化参数
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.create_Q_network()
        self.create_training_method()

        # 初始化session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def create_Q_network(self):
        # 状态输入
        self.state_input = tf.placeholder("float", [None, self.state_dim])
        # network weights
        with tf.variable_scope('current_net'):  # 当前Q网络的scope
            W1 = self.weight_variable([self.state_dim, 20])  # 里层20个神经元
            b1 = self.bias_variable([20])
            W2 = self.weight_variable([20, self.action_dim])  # 输出action对应的Q_value
            b2 = self.bias_variable([self.action_dim])

            # 隐藏层
            h_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
            # 输出层
            self.Q_value = tf.matmul(h_layer, W2) + b2

        with tf.variable_scope('target_net'):  # 目标Q网络的scope
            W1t = self.weight_variable([self.state_dim, 20])
            b1t = self.bias_variable([20])
            W2t = self.weight_variable([20, self.action_dim])
            b2t = self.bias_variable([self.action_dim])

            # 隐藏层
            h_layer_t = tf.nn.relu(tf.matmul(self.state_input, W1t) + b1t)
            # 输出层
            self.target_Q_value = tf.matmul(h_layer_t, W2t) + b2t  # 使用目标网络时self.target_Q_value
        # tf.get_collection（key, scope='')用来获取指定集合key中的所有元素并返回这个列表 指定scope后会放入key集合中位于给定scope的元素
        # tf.GraphKeys.GLOBAL_VARIABLES 默认包括所有Variable对象
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='current_net')

        with tf.variable_scope('soft_replacement'):
            # tf.assign(a, b) 将b的值赋给a
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]  # 更新目标t网络的params

    def create_training_method(self):  # 同DQN
        self.action_input = tf.placeholder("float", [None, self.action_dim])
        self.y_input = tf.placeholder("float", [None])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def perceive(self, state, action, reward, next_state, done):  # 同DQN
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def train_Q_network(self):
        self.time_step += 1
        # 从经验回放池中随机抽取minibatch
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # 计算y值
        y_batch = []
        Q_value_batch = self.target_Q_value.eval(feed_dict={self.state_input: next_state_batch})  # 使用target_Q_value计算
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
            self.y_input: y_batch,  # 给y值用来计算loss
            self.action_input: action_batch,  # 给action确定具体Q值
            self.state_input: state_batch  # 给state用来feed神经网络产生Q值列表
        })

    def egreedy_action(self, state):  # 同DQN
        Q_value = self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0]
        if random.random() <= self.epsilon:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return random.randint(0, self.action_dim - 1)
        else:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return np.argmax(Q_value)

    def action(self, state):  # 同DQN
        return np.argmax(self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0])

    def update_target_q_network(self, episode):
        # 更新目标Q网络
        if episode % REPLACE_TARGET_FREQ == 0:
            self.session.run(self.target_replace_op)  # target_replace_op能够批量更新目标Q网络的参数

    def weight_variable(self, shape):  # 同DQN
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):  # 同DQN
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)


# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 3000
STEP = 300
TEST = 5


def main():
    # 初始化运行环境和DQN
    env = gym.make(ENV_NAME)
    agent = DQN(env)

    for episode in range(EPISODE):
        # 重置状态
        state = env.reset()
        # 训练
        for step in range(STEP):
            action = agent.egreedy_action(state)  # ε贪婪法选择action
            next_state, reward, done, _ = env.step(action)
            # 改一下reward
            reward = -1 if done else 0.1
            # 存入经验回放池并进行训练
            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        # 每100循环进行测试
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    env.render()
                    action = agent.action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
        agent.update_target_q_network(episode)


if __name__ == '__main__':
    main()
