import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque

# 超参数
GAMMA = 0.95  # 衰减因子
LEARNING_RATE = 0.01  # 学习率


class Policy_Gradient():
    def __init__(self, env):
        # 初始化超参数
        self.time_step = 0
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self.create_softmax_network()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def create_softmax_network(self):
        # 与DQN一样的网络
        W1 = self.weight_variable([self.state_dim, 20])
        b1 = self.bias_variable([20])
        W2 = self.weight_variable([20, self.action_dim])
        b2 = self.bias_variable([self.action_dim])
        # 输入状态、动作、价值  每组batches由一个完整的蒙特卡洛序列构成
        self.state_input = tf.placeholder("float", [None, self.state_dim])  # 存储每步状态
        self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")  # 存储每步真实的动作
        self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")  # 存储每步价值
        # 隐藏层
        h_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
        # 输出层 输出后进行softmax处理
        self.softmax_input = tf.matmul(h_layer, W2) + b2  # softmax_input的shape是[n, action_dim] n为batchsize（总步数）
        # 使用softmax函数进行归一化  all_act_prob的shape也是[n, action_dim]
        # 独立出来计算acc_act_prob是为了根据动作概率选择动作
        self.all_act_prob = tf.nn.softmax(self.softmax_input, name='act_prob')
        # sparse_softmax_cross_entropy_with_logits函数接收神经网络输出，进行softmax化并计算交叉熵
        # logits：接收神经网络输出 labels：为长度为batch_size的一维向量，存储了真实的动作
        # 函数的输出shape为[n, 1] 每行为每个batch（每步）的交叉熵
        self.neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.softmax_input, labels=self.tf_acts)
        self.loss = tf.reduce_mean(self.neg_log_prob * self.tf_vt)  # 乘上vt得出loss：logπθ(st,at)vt

        self.train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)  # 使用optimizer最小化loss

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def choose_action(self, observation):  # 根据观察到的state随机选取动作
        # 将state输入网络，生成对应动作概率（使用np.newaxis将一维的state转换成二维的state以符合数据输入格式）
        prob_weights = self.session.run(self.all_act_prob, feed_dict={self.state_input: observation[np.newaxis, :]})
        # 从action_dim中随机选取一个动作，使用p指定选取各个动作的概率（ravel将二维的数组拉成一维）
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def store_transition(self, s, a, r):  # 将蒙特卡洛学列每一组数据存入数组
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):  # 模型训练
        # 根据一条完整的蒙特卡洛序列各步的即时奖励计算累计奖励
        discounted_ep_rs = np.zeros_like(self.ep_rs)  # 生成累计奖励数组
        running_add = 0  # 从后向前统计累计奖励
        for t in reversed(range(0, len(self.ep_rs))):  # 从后向前
            running_add = running_add * GAMMA + self.ep_rs[t]  # 累计奖励 = 上一步累计奖励 * γ + 当前奖励
            discounted_ep_rs[t] = running_add  # 存入累计奖励数组

        # 标准化累计奖励 使其平均数为0 标准差为1
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)

        # 训练
        self.session.run(self.train_op, feed_dict={
             self.state_input: np.vstack(self.ep_obs),
             self.tf_acts: np.array(self.ep_as),
             self.tf_vt: discounted_ep_rs,
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # 重置蒙特卡洛序列的存储

ENV_NAME = 'CartPole-v0'
EPISODE = 3000
STEP = 3000
TEST = 10

def main():
  env = gym.make(ENV_NAME)
  agent = Policy_Gradient(env)

  for episode in range(EPISODE):

    state = env.reset()

    for step in range(STEP):
      action = agent.choose_action(state)  # 按照动作概率随机选取一个动作
      next_state,reward,done,_ = env.step(action)
      agent.store_transition(state, action, reward)
      state = next_state
      if done:  # 如果有了一个完整的序列
        agent.learn()  # 训练
        break

    # 测试
    if episode % 100 == 0:
      total_reward = 0
      for i in range(TEST):
        state = env.reset()
        for j in range(STEP):
          env.render()
          action = agent.choose_action(state)  # 同样是按照动作概率选取一个动作
          state,reward,done,_ = env.step(action)
          total_reward += reward
          if done:
            break
      ave_reward = total_reward/TEST
      print ('episode: ',episode,'Evaluation Average Reward:',ave_reward)

if __name__ == '__main__':
  main()