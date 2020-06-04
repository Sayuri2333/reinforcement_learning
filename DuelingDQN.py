import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque

# 超参数
GAMMA = 0.9
INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.01
REPLAY_SIZE = 10000
BATCH_SIZE = 128
REPLACE_TARGET_FREQ = 10


class DQN():
  # DQN Agent
  def __init__(self, env):
    # 初始化经验回放池
    self.replay_buffer = deque()

    self.time_step = 0
    self.epsilon = INITIAL_EPSILON
    self.state_dim = env.observation_space.shape[0]
    self.action_dim = env.action_space.n

    self.create_Q_network()
    self.create_training_method()

    self.session = tf.InteractiveSession()
    self.session.run(tf.global_variables_initializer())

  def create_Q_network(self):
    # 状态输入
    self.state_input = tf.placeholder("float", [None, self.state_dim])
    # 网络权重 基于Nature DQN的构造 分成当前网络以及目标网络
    with tf.variable_scope('current_net'):
        W1 = self.weight_variable([self.state_dim,20])
        b1 = self.bias_variable([20])

        # 第二层
        h_layer_1 = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)

        # 第三层价值函数部分  输出为只有一个值的行向量
        with tf.variable_scope('Value'):
          W21= self.weight_variable([20,1])
          b21 = self.bias_variable([1])
          self.V = tf.matmul(h_layer_1, W21) + b21

        # 第三层优势函数部分  输出为action_dim维的行向量
        with tf.variable_scope('Advantage'):
          W22 = self.weight_variable([20,self.action_dim])
          b22 = self.bias_variable([self.action_dim])
          self.A = tf.matmul(h_layer_1, W22) + b22

          # 输出Q值 为action_dim维的行向量 一维行向量与高维行向量相加 = 一维行向量内的数与高维行向量相加
          self.Q_value = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))

    with tf.variable_scope('target_net'):
        W1t = self.weight_variable([self.state_dim,20])
        b1t = self.bias_variable([20])

        h_layer_1t = tf.nn.relu(tf.matmul(self.state_input,W1t) + b1t)

        with tf.variable_scope('Value'):
          W2v = self.weight_variable([20,1])
          b2v = self.bias_variable([1])
          self.VT = tf.matmul(h_layer_1t, W2v) + b2v

        with tf.variable_scope('Advantage'):
          W2a = self.weight_variable([20,self.action_dim])
          b2a = self.bias_variable([self.action_dim])
          self.AT = tf.matmul(h_layer_1t, W2a) + b2a

          self.target_Q_value = self.VT + (self.AT - tf.reduce_mean(self.AT, axis=1, keep_dims=True))

    t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
    e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='current_net')

    with tf.variable_scope('soft_replacement'):
        self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

  def create_training_method(self):
    self.action_input = tf.placeholder("float",[None,self.action_dim]) # one hot presentation
    self.y_input = tf.placeholder("float",[None])
    Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),reduction_indices = 1)
    self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
    self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

  def perceive(self,state,action,reward,next_state,done):
    one_hot_action = np.zeros(self.action_dim)
    one_hot_action[action] = 1
    self.replay_buffer.append((state,one_hot_action,reward,next_state,done))
    if len(self.replay_buffer) > REPLAY_SIZE:
      self.replay_buffer.popleft()

    if len(self.replay_buffer) > BATCH_SIZE:
      self.train_Q_network()

  def train_Q_network(self):
    self.time_step += 1
    # Step 1: obtain random minibatch from replay memory
    minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
    state_batch = [data[0] for data in minibatch]
    action_batch = [data[1] for data in minibatch]
    reward_batch = [data[2] for data in minibatch]
    next_state_batch = [data[3] for data in minibatch]

    # Step 2: calculate y
    y_batch = []
    Q_value_batch = self.target_Q_value.eval(feed_dict={self.state_input:next_state_batch})
    for i in range(0,BATCH_SIZE):
      done = minibatch[i][4]
      if done:
        y_batch.append(reward_batch[i])
      else :
        y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

    self.optimizer.run(feed_dict={
      self.y_input:y_batch,
      self.action_input:action_batch,
      self.state_input:state_batch
      })

  def egreedy_action(self,state):
    Q_value = self.Q_value.eval(feed_dict = {
      self.state_input:[state]
      })[0]
    if random.random() <= self.epsilon:
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
        return random.randint(0,self.action_dim - 1)
    else:
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
        return np.argmax(Q_value)

  def action(self,state):
    return np.argmax(self.Q_value.eval(feed_dict = {
      self.state_input:[state]
      })[0])

  def update_target_q_network(self, episode):
    # update target Q netowrk
    if episode % REPLACE_TARGET_FREQ == 0:
        self.session.run(self.target_replace_op)


  def weight_variable(self,shape):
    initial = tf.truncated_normal(shape)
    return tf.Variable(initial)

  def bias_variable(self,shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)
# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 3000 # Episode limitation
STEP = 300 # Step limitation in an episode
TEST = 5 # The number of experiment test every 100 episode

def main():
  # initialize OpenAI Gym env and dqn agent
  env = gym.make(ENV_NAME)
  agent = DQN(env)

  for episode in range(EPISODE):
    # initialize task
    state = env.reset()
    # Train
    for step in range(STEP):
      action = agent.egreedy_action(state) # e-greedy action for train
      next_state,reward,done,_ = env.step(action)
      # Define reward for agent
      reward = -1 if done else 0.1
      agent.perceive(state,action,reward,next_state,done)
      state = next_state
      if done:
        break
    # Test every 100 episodes
    if episode % 100 == 0:
      total_reward = 0
      for i in range(TEST):
        state = env.reset()
        for j in range(STEP):
          env.render()
          action = agent.action(state) # direct action for test
          state,reward,done,_ = env.step(action)
          total_reward += reward
          if done:
            break
      ave_reward = total_reward/TEST
      print ('episode: ',episode,'Evaluation Average Reward:',ave_reward)
    agent.update_target_q_network(episode)

if __name__ == '__main__':
  main()