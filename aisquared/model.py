import tensorflow as tf
import datetime
import numpy as np
from aisquared.reinforce import Reinforce
from aisquared.model_manager import ModelManager

class AISquared(object):
    """docstring for AISquared."""
    def __init__(self, dataset,
                max_layers=3,
                learning_rate=0.99,
                decay_steps=500,
                decay_rate=0.96):

        self.max_layers = max_layers

        self.log("Setting up tf session.")
        self.sess = tf.Session()
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step,
                                               decay_steps, decay_rate, staircase=True)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

        self.log("Instantiate the reinforce object.")
        self.reinforce = Reinforce(self.sess, optimizer, self.policy_network, max_layers, self.global_step)

        self.log("Instantiate the NetManager object.")
        self.net_manager = ModelManager(num_input=784,
                                 num_classes=10,
                                 learning_rate=0.001,
                                 mnist=dataset,
                                 bathc_size=100,
                                 max_step_per_action=5000)

    def train(self, max_episodes=2):
        self.log("Training AISquared model.")
        self.step = 0

        # Generate random setup for the child AI
        state = np.array([[10.0, 128.0, 1.0, 1.0]*self.max_layers], dtype=np.float32)

        # Set 0 as the previous accuracy
        pre_acc = 0.0
        total_rewards = 0

        for i_episode in range(max_episodes):
            self.log(f"Starting episode: {i_episode+1}")
            action = self.reinforce.get_action(state)
            self.log("Choosen model configuration:", action)
            if all(ai > 0 for ai in action[0][0]):
                reward, pre_acc = self.net_manager.get_reward(action, self.step, pre_acc)
                self.log("=====>", reward, pre_acc)
            else:
                reward = -1.0
            total_rewards += reward

            # In our sample action is equal state
            state = action[0]
            self.reinforce.storeRollout(state, reward)

            self.step += 1
            ls = self.reinforce.train_step(1)
            log_str = "current time:  "+str(datetime.datetime.now().time())+" episode:  "+str(i_episode)+" loss:  "+str(ls)+" last_state:  "+str(state)+" last_reward:  "+str(reward)+"\n"
            log = open("lg3.txt", "a+")
            log.write(log_str)
            log.close()

    def policy_network(self, state, max_layers):
        self.log("Creating policy network")
        num_outputs = 4*max_layers
        with tf.name_scope("policy_network"):
            self.log("Creating NASCells")
            nas_cell = tf.contrib.rnn.NASCell(num_outputs)
            self.log("Creating dynamic RNN")
            outputs, state = tf.nn.dynamic_rnn(
                nas_cell,
                tf.expand_dims(state, -1),
                dtype=tf.float32
            )
            bias = tf.Variable([0.05]* num_outputs)
            outputs = tf.nn.bias_add(outputs, bias)

            return outputs[:, -1:, :]

    def log(self, log):
        print(f"AISquared: {log}")
