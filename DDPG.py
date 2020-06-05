import tensorflow as tf
import numpy as np
import gym
import time


# 超参数

MAX_EPISODES = 2000
MAX_EP_STEPS = 200
LR_A = 0.001    # actor网络的学习率
LR_C = 0.002    # critic网络的学习率
GAMMA = 0.9     # γ值
TAU = 0.01      # 参数更新幅度
MEMORY_CAPACITY = 10000  # 经验回放池大小
BATCH_SIZE = 32  # minibatch大小

RENDER = False
ENV_NAME = 'Pendulum-v0'


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)  # 使用numpy数组作经验回放池
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,  # a_bound表示动作的取值范围
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')  # 接收当前状态
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')  # 接收下一个状态
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')  # 接收当前即时奖励

        with tf.variable_scope('Actor'):  # self.a 为当前actor网络输出；a_为目标actor网络输出
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):  # q 为当前critic网络输出； q_为目标critic网络输出
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)  # 注意当前critic网络使用当前actor网络的输入
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)  # 注意目标critic网络使用目标actor网络的输入

        # 提取变量到列表 eval为当前网络 target为目标网络
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # 更新目标网络的参数
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_  # 计算y值
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)  # 计算当前critic网络的均方差损失函数
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)  # 训练当前critic网络 使用Var_list指定需要训练的变量列表

        a_loss = - tf.reduce_mean(q)    # 计算当前actor网络的损失函数
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)  # 训练当前actor网络

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]  # 根据当前actor网络选择动作 因为输入状态为一维变量所以要升维

    def learn(self):
        # 更新目标网络
        self.sess.run(self.soft_replace)

        # 随机选取minibatch
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        # 分割出当前状态、下一个状态、奖励值以及动作
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]
        # 训练actor网络以及critic网络
        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):  # 将经验存入memory中
        transition = np.hstack((s, a, [r], s_))  # 需要将r变成数组才能使用hstack进行连接
        index = self.pointer % MEMORY_CAPACITY  # pointer是单调递增的 需要取余
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):  # 造actor网络 s:状态placeholder； scope：变量scope；trainable：是否能够训练  可以支持batch
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)  # dense添加全连接层，30为指定输出维度
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)  # tanh函数值域为[-1,1]
            return tf.multiply(a, self.a_bound, name='scaled_a')  # tanh值乘上动作范围表示具体动作输出值  输出值格式为[n, action_dim]

    def _build_c(self, s, a, scope, trainable):  # 造critic网络 参数同上
        # 这个网络与之前不同之处在于这个网络接收(s,a)作为输入，输出一个Q值
        with tf.variable_scope(scope):
            n_l1 = 30  # 隐藏层输出个数
            w1_s = tf.get_variable('w1_s', [self.s_dim,  n_l1], trainable=trainable)  # 与状态对应的权重
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)  # 与动作对应的权重
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)  # 偏置
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)  # net的shape为[n, n_l1]
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)  输出一个数值 shape为[n, 1]

# 训练

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high  # 获得动作范围

ddpg = DDPG(a_dim, s_dim, a_bound)

var = 3  # 控制动作选择时的噪声方差
t1 = time.time()
for episode in range(MAX_EPISODES):
    s = env.reset()  # 获得初始状态s的特征向量
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        a = ddpg.choose_action(s)  # 根据当前actor网络获得动作
        a = np.clip(np.random.normal(a, var), -2, 2)  # random.normal()生成一个以a为均值 var为标准差的正态分布并取其中一个数字； clip()将这个数字的值限制在-2至2以内
        s_, r, done, info = env.step(a)

        ddpg.store_transition(s, a, r / 10, s_)  # 存储经验

        if ddpg.pointer > MEMORY_CAPACITY:  # 一旦经验回放池满了
            var *= .9995    # 方差缩小
            ddpg.learn()  # 训练当前actor以及critic网络

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            print('Episode:', episode, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            # if ep_reward > -300:RENDER = True
            break
    if episode % 100 == 0:
      total_reward = 0
      for i in range(10):
        state = env.reset()
        for j in range(MAX_EP_STEPS):
          env.render()
          action = ddpg.choose_action(state)  # 直接选择动作
          state,reward,done,_ = env.step(action)
          total_reward += reward
          if done:
            break
      ave_reward = total_reward/300
      print ('episode: ',episode,'Evaluation Average Reward:',ave_reward)
print('Running time: ', time.time() - t1)