import numpy as np
import random
import datetime
import time
import tensorflow as tf
from collections import deque
from mlagents.envs import UnityEnvironment

state_size = 10*4
action_size = 3

load_model = False
train_mode = True

batch_size = 32
mem_maxlen = 50000
discount_factor = 0.99
learning_rate = 0.00025

run_episode = 10000
test_episode = 200

start_train_episode = 250

target_update_step = 10000
print_interval = 10
save_interval = 5000

epsilon_init = 1.0
epsilon_min = 0.1

date_time = str(datetime.date.today()) + '_' + \
            str(datetime.datetime.now().hour) + '_' + \
            str(datetime.datetime.now().minute) + '_' + \
            str(datetime.datetime.now().second)

game = "Pong"
env_name = "./env/Pong"

save_path = "./saved_models/" + game + "/" + date_time + "_DQN"
load_path = "./saved_models/" + game + "/2019-07-08_14_51_24_DQN/model/model"

class Model():
    def __init__(self, model_name):
        self.input = tf.placeholder(shape=[None, state_size], dtype=tf.float32)

        with tf.variable_scope(name_or_scope=model_name):
            self.fc1 = tf.layers.dense(self.input,512,activation=tf.nn.relu)
            self.fc2 = tf.layers.dense(self.fc1,512,activation=tf.nn.relu)
            self.fc3 = tf.layers.dense(self.fc2,512,activation=tf.nn.relu)
            self.Q_Out = tf.layers.dense(self.fc3,action_size,activation=None)
        self.predict = tf.argmax(self.Q_Out, 1)

        self.target_Q = tf.placeholder(shape=[None, action_size], dtype=tf.float32)

        self.loss = tf.losses.huber_loss(self.target_Q, self.Q_Out)
        self.UpdateModel = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model_name)

class DQNAgent():
    def __init__(self):
        self.model = Model("Q")
        self.target_model = Model("target")

        self.memory = deque(maxlen=mem_maxlen)

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        self.sess = tf.InteractiveSession(config=self.config)
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        self.epsilon = epsilon_init

        self.Saver = tf.train.Saver()

        self.Summary, self.Merge = self.make_Summary()

        self.update_target()

        if load_model == True:
            self.Saver.restore(self.sess, load_path)

    def get_action(self, state, train_mode=True):
        if train_mode == True and self.epsilon > np.random.rand():
            return np.random.randint(0, action_size)
        else:
            predict = self.sess.run(self.model.predict, feed_dict={self.model.input: [state]})
            return np.asscalar(predict)

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def save_model(self):
        self.Saver.save(self.sess, save_path + "/model/model")

    def train_model(self, done):
        if done:
            if self.epsilon > epsilon_min:
                self.epsilon -= 1 / (run_episode - start_train_episode)

        mini_batch = random.sample(self.memory, batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for i in range(batch_size):
            states.append(mini_batch[i][0])
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states.append(mini_batch[i][3])
            dones.append(mini_batch[i][4])

        target = self.sess.run(self.model.Q_Out, feed_dict={self.model.input: states})
        target_val = self.sess.run(self.target_model.Q_Out, 
                                   feed_dict={self.target_model.input: next_states})

        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + discount_factor * np.amax(target_val[i])

        _, loss = self.sess.run([self.model.UpdateModel, self.model.loss],
                                feed_dict={self.model.input: states, 
                                           self.model.target_Q: target})
        return loss

    def update_target(self):
        for i in range(len(self.model.trainable_var)):
            self.sess.run(self.target_model.trainable_var[i].assign(self.model.trainable_var[i]))

    def make_Summary(self):
        self.summary_loss = tf.placeholder(dtype=tf.float32)
        self.summary_reward = tf.placeholder(dtype=tf.float32)
        tf.summary.scalar("loss", self.summary_loss)
        tf.summary.scalar("reward", self.summary_reward)
        return tf.summary.FileWriter(logdir=save_path, graph=self.sess.graph), tf.summary.merge_all()

    def Write_Summray(self, reward, loss, episode):
        self.Summary.add_summary(
            self.sess.run(self.Merge, feed_dict={self.summary_loss: loss, 
                                                 self.summary_reward: reward}), episode)

if __name__ == '__main__':

    env = UnityEnvironment(file_name=env_name)

    default_brain = env.brain_names[0]
    brain = env.brains[default_brain]

    agent = DQNAgent()

    step = 0
    rewards = []
    losses = []

    env_info = env.reset(train_mode=train_mode)[default_brain]

    for episode in range(run_episode + test_episode):
        if episode == run_episode:
            train_mode = False
            env_info = env.reset(train_mode=train_mode)[default_brain]
        
        state = env_info.vector_observations[0]
        episode_rewards = 0
        done = False

        while not done:
            step += 1

            action = agent.get_action(state, train_mode)
            env_info = env.step(action)[default_brain]

            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            episode_rewards += reward
            done = env_info.local_done[0]

            if train_mode:
                agent.append_sample(state, action, reward, next_state, done)
            else:
                time.sleep(0.01)
                agent.epsilon = 0.0

            state = next_state

            if episode > start_train_episode and train_mode:
                loss = agent.train_model(done)
                losses.append(loss)

                if step % (target_update_step) == 0:
                    agent.update_target()

        rewards.append(episode_rewards)

        if episode % print_interval == 0 and episode != 0:
            print("step: {} / episode: {} / reward: {:.2f} / loss: {:.4f} / epsilon: {:.3f}".format
                  (step, episode, np.mean(rewards), np.mean(losses), agent.epsilon))
            agent.Write_Summray(np.mean(rewards), np.mean(losses), episode)
            rewards = []
            losses = []

        if episode % save_interval == 0 and episode != 0:
            agent.save_model()
            print("Save Model {}".format(episode))

    env.close()