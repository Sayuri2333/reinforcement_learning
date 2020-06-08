import os
import gym
import numpy as np
import pandas as pd
import tensorflow as tf


class PPO:
    def __init__(self, ep, batch, t='ppo2'):
        self.t = t  # 算法种类
        self.ep = ep  # 最大循环数
        self.batch = batch  # batch大小
        self.log = 'model/{}_log'.format(t)

        self.env = gym.make('Pendulum-v0')
        self.bound = self.env.action_space.high[0]  # 动作空间范围

        self.gamma = 0.9  # γ值
        self.A_LR = 0.0001  # actor网络学习率
        self.C_LR = 0.0002  # critic网络学习率
        self.A_UPDATE_STEPS = 10  #
        self.C_UPDATE_STEPS = 10

        # KL penalty, d_target、β for ppo1
        self.kl_target = 0.01  # 用来作为参照的KL散度
        self.lam = 0.5  # 使用KL散度计算loss时β值
        # ε for ppo2
        self.epsilon = 0.2  # PPO2算法中控制clip大小的ε值

        self.sess = tf.Session()  # 开启session
        self.build_model()  # 初始化模型

    def _build_critic(self):  # 建立critic网络 既能计算v值也能计算优势函数值
        with tf.variable_scope('critic'):
            x = tf.layers.dense(self.states, 100, tf.nn.relu)  # 输入state为输入层

            self.v = tf.layers.dense(x, 1)  # 输出对应state的v值
            self.advantage = self.dr - self.v  # 计算优势函数

    def _build_actor(self, name, trainable):  # 建立actor网络  trainable指定是否支持训练
        with tf.variable_scope(name):
            x = tf.layers.dense(self.states, 100, tf.nn.relu, trainable=trainable)  # 输入state作为输入层

            mu = self.bound * tf.layers.dense(x, 1, tf.nn.tanh, trainable=trainable)  # 使用tanh函数输出μ值 值域在[-bound, bound]
            sigma = tf.layers.dense(x, 1, tf.nn.softplus, trainable=trainable)  # 使用softplus（relu的平滑版本）输出σ值

            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)  # 使用normal方法根据μ以及σ生成正态分布

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)  # 获得当前网络的所有参数

        return norm_dist, params

    def build_model(self):
        # inputs
        self.states = tf.placeholder(tf.float32, [None, 3], 'states')  # 状态输入
        self.action = tf.placeholder(tf.float32, [None, 1], 'action')  # 动作输入
        self.adv = tf.placeholder(tf.float32, [None, 1], 'advantage')  # 存储优势函数
        self.dr = tf.placeholder(tf.float32, [None, 1], 'discounted_r')  # 存储衰减价值

        # 建立critic网络以及两个actor网络
        self._build_critic()
        nd, pi_params = self._build_actor('actor', trainable=True)
        old_nd, oldpi_params = self._build_actor('old_actor', trainable=False)

        # 定义PPO loss
        with tf.variable_scope('loss'):
            # critic loss
            self.closs = tf.reduce_mean(tf.square(self.advantage))  # 使用优势函数的均方误差作为critic网络的loss

            # actor loss
            with tf.variable_scope('surrogate'):
                # log_prob计算出对应动作出现的概率的log值
                ratio = tf.exp(nd.log_prob(self.action) - old_nd.log_prob(self.action))  # 计算出前后动作的概率比值
                surr = ratio * self.adv  # 乘上优势函数

            if self.t == 'ppo1':  # 在PPO1的情况下
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')  # 接收β值输入（不直接指定β值是因为β会根据KL散度的值发生变化）
                kl = tf.distributions.kl_divergence(old_nd, nd)  # 计算前后动作概率分布的KL散度
                self.kl_mean = tf.reduce_mean(kl)  # 计算所有动作下KL散度的期望
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))  # 构造actor网络的loss
            else:  # 在PPO2的情况下
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.- self.epsilon, 1.+ self.epsilon) * self.adv))  # 构造actor网络的loss

        # 定义优化器
        with tf.variable_scope('optimize'):
            self.ctrain_op = tf.train.AdamOptimizer(self.C_LR).minimize(self.closs)
            self.atrain_op = tf.train.AdamOptimizer(self.A_LR).minimize(self.aloss)

        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(nd.sample(1), axis=0)  # 随机sample一个动作（先不clip）

        # update old actor
        with tf.variable_scope('update_old_actor'):
            self.update_old_actor = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]  # 将旧actor网络中的参数更新为当前actor网络中的参数

        tf.summary.FileWriter(self.log, self.sess.graph)

        self.sess.run(tf.global_variables_initializer())  # 初始化所有向量

    def choose_action(self, state):  # 正态分布中随机抽取一个动作后clip到范围内
        state = state[np.newaxis, :]
        action = self.sess.run(self.sample_op, {self.states: state})[0]

        return np.clip(action, -self.bound, self.bound)

    def get_value(self, state):
        if state.ndim < 2: state = state[np.newaxis, :]  # 如果state不是二维的 要升维

        return self.sess.run(self.v, {self.states: state})  # 使用critic网络计算对应状态v值

    def discount_reward(self, states, rewards, next_observation):  # 计算衰减reward next_observation这个batch最后一个动作执行后的状态
        s = np.vstack([states, next_observation.reshape(-1, 3)])  # 把next_observation添加到states数组底部
        q_values = self.get_value(s).flatten()  # 获得所有states的价值

        targets = rewards + self.gamma * q_values[1:]  # 计算每个状态的衰减reward
        targets = targets.reshape(-1, 1)  # 修改格式

        return targets

# not work.
#    def neglogp(self, mean, std, x):
#        """Gaussian likelihood
#        """
#        return 0.5 * tf.reduce_sum(tf.square((x - mean) / std), axis=-1) \
#               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
#               + tf.reduce_sum(tf.log(std), axis=-1)

    def update(self, states, action, dr):  # 更新模型 dr为衰减reward
        """update model.

        Arguments:
            states: states.
            action: action of states.
            dr: discount reward of action.
        """
        self.sess.run(self.update_old_actor)  # 更新旧actor网络中的参数

        adv = self.sess.run(self.advantage,  # 使用critic网络计算adv值
                            {self.states: states,
                             self.dr: dr})

        # 更新当前actor网络中的参数
        if self.t == 'ppo1':
            # 在PPO1的前提下更新参数
            for _ in range(self.A_UPDATE_STEPS):  # 使用一组数据update指定次数
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.states: states,
                     self.action: action,
                     self.adv: adv,
                     self.tflam: self.lam})

            if kl < self.kl_target / 1.5:  # 根据KL散度调整β大小
                self.lam /= 2
            elif kl > self.kl_target * 1.5:
                self.lam *= 2
        else:
            # 在PPO2的前提下调整参数
            for _ in range(self.A_UPDATE_STEPS):
                self.sess.run(self.atrain_op,
                              {self.states: states,
                               self.action: action,
                               self.adv: adv})

        # 更新critic网络
        for _ in range(self.C_UPDATE_STEPS):
            self.sess.run(self.ctrain_op,
                          {self.states: states,
                           self.dr: dr})

    def train(self):
        """train method.
        """
        tf.reset_default_graph()  # 初始化计算图

        history = {'episode': [], 'Episode_reward': []}

        for i in range(self.ep):  # 每次循环
            observation = self.env.reset()  # 重设环境

            states, actions, rewards = [], [], []
            episode_reward = 0
            j = 0

            while True:  # 运行并接收states action reward
                a = self.choose_action(observation)
                next_observation, reward, done, _ = self.env.step(a)
                states.append(observation)
                actions.append(a)

                episode_reward += reward
                rewards.append((reward + 8) / 8)

                observation = next_observation

                if (j + 1) % self.batch == 0:  # 如果batch够了
                    states = np.array(states)
                    actions = np.array(actions)
                    rewards = np.array(rewards)
                    d_reward = self.discount_reward(states, rewards, next_observation)  # 计算衰减reward

                    self.update(states, actions, d_reward)  # 更新三个网络的参数

                    states, actions, rewards = [], [], []

                if done:  # 如果游戏结束则停止循环
                    break
                j += 1

            history['episode'].append(i)  # 记录episode以及对应总reward
            history['Episode_reward'].append(episode_reward)
            print('Episode: {} | Episode reward: {:.2f}'.format(i, episode_reward))

        return history

    def save_history(self, history, name):  # 存储history
        name = os.path.join('history', name)

        df = pd.DataFrame.from_dict(history)
        df.to_csv(name, index=False, encoding='utf-8')


if __name__ == '__main__':
    model = PPO(1000, 32, 'ppo2')
    history = model.train()
    model.save_history(history, 'ppo1.csv')