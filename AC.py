import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque

GAMMA = 0.95  # 衰减因子
LEARNING_RATE = 0.01  # 学习率


class Actor():  # Actor网络 与PolicyGradient网络相似
    def __init__(self, env, sess):
        self.time_step = 0
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.create_softmax_network()

        self.session = sess
        self.session.run(tf.global_variables_initializer())

    def create_softmax_network(self):
        W1 = self.weight_variable([self.state_dim, 20])
        b1 = self.bias_variable([20])
        W2 = self.weight_variable([20, self.action_dim])
        b2 = self.bias_variable([self.action_dim])

        self.state_input = tf.placeholder("float", [None, self.state_dim])
        self.tf_acts = tf.placeholder(tf.int32, [None, 2], name="actions_num")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error
        # 隐藏层
        h_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
        # 输出层
        self.softmax_input = tf.matmul(h_layer, W2) + b2
        # 输出层进行softmax处理
        self.all_act_prob = tf.nn.softmax(self.softmax_input, name='act_prob')
        # 结合tf_acts计算交叉熵
        self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=self.softmax_input, labels=self.tf_acts)
        # 结合td_error计算actor网络的loss
        self.exp = tf.reduce_mean(self.neg_log_prob * self.td_error)

        # 这里需要最大化当前策略的价值，因此需要最大化self.exp,即最小化-self.exp
        # 上面的注释有问题 原本softmax_cross_entropy_with_logits（）方法就是带负号的  至于为什么不要负号参照知乎
        self.train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.exp)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def choose_action(self, observation):
        prob_weights = self.session.run(self.all_act_prob, feed_dict={self.state_input: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def learn(self, state, action, td_error):  # 每一步learn一次
        s = state[np.newaxis, :]  # 需要升维
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        a = one_hot_action[np.newaxis, :]  # 将action转为one-hot形式
        # 训练
        self.session.run(self.train_op, feed_dict={
            self.state_input: s,
            self.tf_acts: a,
            self.td_error: td_error,
        })


# critic网络参数
EPSILON = 0.01  # ε最终值
REPLAY_SIZE = 10000  # 经验回放池大小
BATCH_SIZE = 32  # minibatch大小


class Critic():
    def __init__(self, env, sess):
        self.time_step = 0
        self.epsilon = EPSILON
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.create_Q_network()
        self.create_training_method()

        # Init session
        self.session = sess
        self.session.run(tf.global_variables_initializer())

    def create_Q_network(self):
        # 网络权重
        W1q = self.weight_variable([self.state_dim, 20])
        b1q = self.bias_variable([20])
        W2q = self.weight_variable([20, 1])
        b2q = self.bias_variable([1])  # 注意输出只有一维
        self.state_input = tf.placeholder(tf.float32, [1, self.state_dim], "state")  # 接收一个状态向量作为输入
        # 隐藏层
        h_layerq = tf.nn.relu(tf.matmul(self.state_input, W1q) + b1q)
        # 输出层
        self.Q_value = tf.matmul(h_layerq, W2q) + b2q

    def create_training_method(self):
        self.next_value = tf.placeholder(tf.float32, [1, 1], "v_next")  # 接收下一个状态对应价值
        self.reward = tf.placeholder(tf.float32, None, 'reward')  # 接收当前状态即时价值

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.reward + GAMMA * self.next_value - self.Q_value  # 计算td-error
            self.loss = tf.square(self.td_error)  # 计算critic网络的loss
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.epsilon).minimize(self.loss)  # 优化loss

    def train_Q_network(self, state, reward, next_state):
        s, s_ = state[np.newaxis, :], next_state[np.newaxis, :]  # 将state升维
        v_ = self.session.run(self.Q_value, {self.state_input: s_})  # 计算下一个状态对应价值
        td_error, _ = self.session.run([self.td_error, self.train_op], {self.state_input: s, self.next_value: v_, self.reward: reward})  # 计算td-error并更新critic网络参数
        return td_error  # 返回td-error

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)


# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 3000
STEP = 3000
TEST = 10


def main():
    sess = tf.InteractiveSession()
    env = gym.make(ENV_NAME)
    actor = Actor(env, sess)
    critic = Critic(env, sess)

    for episode in range(EPISODE):
        state = env.reset()
        # 训练 先训练critic网络 接收td-error 再训练actor网络
        for step in range(STEP):
            action = actor.choose_action(state)  # 根据actor网络得出的动作概率随机选取动作
            next_state, reward, done, _ = env.step(action)  # 执行动作
            td_error = critic.train_Q_network(state, reward, next_state)  # 训练critic网络：grad[(r + gamma * V(s_) - V(s)) ^ 2]；并返回td误差
            actor.learn(state, action, td_error)  # true_gradient = grad[logPi(s,a) * td_error]  # 训练actor网络
            state = next_state  # 更新当前状态
            if done:
                break

        # 测试
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    env.render()
                    action = actor.choose_action(state)
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)


if __name__ == '__main__':
    main()
