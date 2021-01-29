# coding:utf-8

import os
import gym
import random
import numpy as np
import tensorflow as tf

from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

GAME = 'BreakoutDeterministic-v4'
FRAME_SIZE = 80

NUM_EXPLORING = 5000
NUM_TRAINING = 100000
NUM_TESTING = 25000

FRAME_STACKING = 4
GAMMA = 0.99
FIRST_EPSILON = 1
FINAL_EPSILON = 0.1
LEARNING_RATE = 0.00025
REPLAY_MEMORY = 50000
BATCH_SIZE = 32
NUM_PLOT_EPISODE = 10
NUM_UPDATE = NUM_TRAINING / 100

IS_TRAIN = True
LOAD_PATH = './DQN_NETWORK'
SAVE_PATH = './DQN_NETWORK'


class DQN:
    def __init__(self):
        self.env = gym.make(GAME)

        self.num_actions = self.env.action_space.n
        self.epsilon = FIRST_EPSILON
        self.progress = ''

        self.step = 1
        self.score = 0
        self.episode = 0

        self.replay_memory = []

        self.Num_Exploration = NUM_EXPLORING
        self.Num_Training = NUM_TRAINING
        self.Num_Testing = NUM_TESTING

        self.loss = 0
        self.maxQ = 0
        self.score_board = 0
        self.maxQ_board = 0
        self.loss_board = 0
        self.step_old = 0

        self.s, self.Q, self.model = self.network()
        self.st, self.Qt, self.target_model = self.network()
        network_weights = self.model.trainable_weights
        target_network_weights = self.target_model.trainable_weights

        self.update_target_network = [target_network_weights[i].assign(network_weights[i]) for i in
                                      range(len(target_network_weights))]
        self.sess, self.saver, self.summary_placeholders, self.update_ops, self.summary_op, self.summary_writer = self.init_sess()

    def main(self):
        observation = self.env.reset()
        state = self.process_image(observation)
        state_set = np.stack(tuple([state] * FRAME_STACKING), axis=2)
        # state_set = state_set.reshape(1, state_set.shape[0], state_set.shape[1], state_set.shape[2])

        while True:
            self.progress = self.get_progress()
            action = self.select_action(state_set)

            next_observation, reward, terminal, _ = self.env.step(np.argmax(action))
            next_state = self.process_image(next_observation)
            next_state = next_state.reshape(next_state.shape[0], next_state.shape[1], 1)
            # print(next_state.shape)
            # print('state_set shape: ' + str(state_set.shape))
            next_state_set = np.append(next_state, state_set[:, :, :3], axis=2)

            self.experience_replay(state_set, action, reward, next_state_set, terminal)

            if self.progress == "Training":
                if self.step % NUM_UPDATE == 0:
                    self.sess.run(self.update_target_network)

                self.train(self.replay_memory, self.model, self.target_model)

                self.save_model()

            state_set = next_state_set
            self.score += reward
            self.step += 1

            self.plotting(terminal)

            if terminal:
                state_set = self.if_terminal()

            if self.progress == 'Finished':
                print("Finished!")
                break

    def network(self):
        model = Sequential()
        model.add(Conv2D(32, 8, (4, 4), activation='relu', padding='same',
                         input_shape=(FRAME_SIZE, FRAME_SIZE, FRAME_STACKING),
                         kernel_initializer=tf.keras.initializers.glorot_uniform(),
                         bias_initializer=tf.keras.initializers.glorot_uniform()))
        model.add(Conv2D(64, 4, (2, 2), activation='relu', padding='same',
                         kernel_initializer=tf.keras.initializers.glorot_uniform(),
                         bias_initializer=tf.keras.initializers.glorot_uniform()))
        model.add(Conv2D(64, 3, (1, 1), activation='relu', padding='same',
                         kernel_initializer=tf.keras.initializers.glorot_uniform(),
                         bias_initializer=tf.keras.initializers.glorot_uniform()))
        model.add(Flatten())
        model.add(Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.glorot_uniform(),
                        bias_initializer=tf.keras.initializers.glorot_uniform()))
        model.add(Dense(self.num_actions, kernel_initializer=tf.keras.initializers.glorot_uniform(),
                        bias_initializer=tf.keras.initializers.glorot_uniform()))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE, epsilon=1e-02))

        state = tf.placeholder(tf.float32, [None, FRAME_SIZE, FRAME_SIZE, FRAME_STACKING])
        # state = (state - (255.0 / 2)) / (255.0 / 2)
        q_values = model(state)

        return state, q_values, model

    def init_sess(self):
        # 初始化
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        sess = tf.InteractiveSession(config=config)

        # 新建用于存储数据的文件夹
        os.makedirs(SAVE_PATH)

        # Summary for tensorboard
        summary_placeholders, update_ops, summary_op = self.setup_summary()
        summary_writer = tf.summary.FileWriter(SAVE_PATH, sess.graph)

        init = tf.global_variables_initializer()
        sess.run(init)  # 初始化变量

        # 如果有文件的话读档
        saver = tf.train.Saver()

        if not IS_TRAIN:
            # IS_TRAIN为False时
            saver.restore(sess, LOAD_PATH + "/model.ckpt")  # 读取模型
            print("Model restored.")
            self.Num_Exploration = 0
            self.Num_Training = 0  # 探索和训练需要的步数都设置为0

        return sess, saver, summary_placeholders, update_ops, summary_op, summary_writer

    def setup_summary(self):
        episode_score = tf.Variable(0.)
        episode_maxQ = tf.Variable(0.)
        episode_loss = tf.Variable(0.)

        tf.summary.scalar('Average Score/' + str(NUM_PLOT_EPISODE) + ' episodes', episode_score)
        tf.summary.scalar('Average MaxQ/' + str(NUM_PLOT_EPISODE) + ' episodes', episode_maxQ)
        tf.summary.scalar('Average Loss/' + str(NUM_PLOT_EPISODE) + ' episodes', episode_loss)  # 新建需要追踪的标量

        summary_vars = [episode_score, episode_maxQ, episode_loss]

        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]  # 新建placeholder用于获得数值
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]  # 更新操作
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def process_image(self, observation):
        observation = rgb2gray(observation)
        observation = resize(observation, (FRAME_SIZE, FRAME_SIZE))
        observation = rescale_intensity(observation, out_range=(0, 255))
        return observation

    def get_progress(self):
        progress = ''
        if self.step <= self.Num_Exploration:
            progress = 'Exploring'
        elif self.step <= self.Num_Exploration + self.Num_Training:
            progress = 'Training'
        elif self.step <= self.Num_Exploration + self.Num_Training + self.Num_Testing:
            progress = 'Testing'
        else:
            progress = 'Finished'

        return progress

    def select_action(self, state_set):  # 根据给定state以及当前阶段选择动作
        action = np.zeros([self.num_actions])

        # 探索阶段随机选择
        if self.progress == 'Exploring':
            # 随机选
            action_index = random.randint(0, self.num_actions - 1)
            action[action_index] = 1

        # training阶段使用ε-greedy方法选择
        elif self.progress == 'Training':
            if random.random() < self.epsilon:
                # 随机选
                action_index = random.randint(0, self.num_actions - 1)
                action[action_index] = 1
            else:
                # 最优选
                Q_value = self.Q.eval(feed_dict={self.s: [state_set]})
                action_index = np.argmax(Q_value)
                action[action_index] = 1
                self.maxQ = np.max(Q_value)

            # ε的值随着每次select递减
            if self.epsilon > FINAL_EPSILON:
                self.epsilon -= FIRST_EPSILON / NUM_TRAINING

        elif self.progress == 'Testing':
            # 测试阶段直接选择最优动作
            Q_value = self.Q.eval(feed_dict={self.s: [state_set]})
            action_index = np.argmax(Q_value)
            action[action_index] = 1
            self.maxQ = np.max(Q_value)

            self.epsilon = 0

        return action

    def experience_replay(self, state, action, reward, next_state, terminal):
        if len(self.replay_memory) >= REPLAY_MEMORY:
            del self.replay_memory[0]
        self.replay_memory.append([state, action, reward, next_state, terminal])

    def train(self, replay_memory, model, target_model):
        minibatch = random.sample(replay_memory, BATCH_SIZE)

        state_batch = np.array([batch[0] for batch in minibatch])
        action_one_hot_batch = np.array([batch[1] for batch in minibatch])
        action_batch = np.argmax(action_one_hot_batch, axis=1)
        reward_batch = np.array([batch[2] for batch in minibatch])
        # print("reward: " + str(reward_batch[0]))
        next_state_batch = np.array([batch[3] for batch in minibatch])
        terminal_batch = np.array([batch[4] for batch in minibatch])

        Q_batch = model.predict(state_batch)
        # print('Q batch: ' + str(Q_batch[0]))
        target_Q_batch = target_model.predict(next_state_batch)
        Q_batch[range(BATCH_SIZE), action_batch] = reward_batch + GAMMA * (1 - terminal_batch) * np.max(
            target_Q_batch, axis=1)
        # print('modified Q batch: ' + str(Q_batch[0]))
        self.loss = model.train_on_batch(state_batch, Q_batch)
        # print("loss: " + str(self.loss))

    def save_model(self):
        if self.step == self.Num_Exploration + self.Num_Training:
            save_path = self.saver.save(self.sess, SAVE_PATH + "/model.ckpt")
            print("Model Saved!")

    def plotting(self, terminal):
        if self.progress != 'Exploring':
            if terminal:
                self.score_board += self.score  # score_board maxQ_board以及loss_board是用来统计每轮画图时总的Q值以及loss值的 scord_board只在terminal状态时更新

            self.maxQ_board += self.maxQ  # Q以及loss在每个step都要更新
            self.loss_board += self.loss

            if self.episode % NUM_PLOT_EPISODE == 0 and self.episode != 0 and terminal:
                diff_step = self.step - self.step_old
                tensorboard_info = [self.score_board / NUM_PLOT_EPISODE, self.maxQ_board / diff_step,
                                    self.loss_board / diff_step]

                for i in range(len(tensorboard_info)):
                    self.sess.run(self.update_ops[i],
                                  feed_dict={self.summary_placeholders[i]: float(tensorboard_info[i])})
                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, self.step)

                self.score_board = 0
                self.maxQ_board = 0
                self.loss_board = 0
                self.step_old = self.step
        else:
            self.step_old = self.step

    def if_terminal(self):
        print('Step: ' + str(self.step) + ' / ' +
              'Episode: ' + str(self.episode) + ' / ' +
              'Progress: ' + self.progress + ' / ' +
              'Epsilon: ' + str(self.epsilon) + ' / ' +
              'Score: ' + str(self.score))

        if self.progress != 'Exploring':
            self.episode += 1
        self.score = 0

        observation = self.env.reset()
        state = self.process_image(observation)
        state_set = np.stack(tuple([state] * FRAME_STACKING), axis=2)
        # state_set = state_set.reshape(1, state_set.shape[0], state_set.shape[1], state_set.shape[2])

        return state_set


if __name__ == '__main__':
    agent = DQN()
    agent.main()
