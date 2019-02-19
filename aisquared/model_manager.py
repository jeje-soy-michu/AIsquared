import tensorflow as tf
from aisquared.model_generator import ModelGenerator

class ModelManager():
    def __init__(self, num_input, num_classes, learning_rate, mnist,
                 max_step_per_action=5500*3,
                 bathc_size=100,
                 dropout_rate=0.85):
        self.log("Setting up Net Manager.")
        self.num_input = num_input
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.mnist = mnist

        self.max_step_per_action = max_step_per_action
        self.bathc_size = bathc_size
        self.dropout_rate = dropout_rate

    def get_reward(self, action, step, pre_acc):
        self.log("Calculating child reward.")
        self.log("Formatting model data.")
        action = [action[0][0][x:x+4] for x in range(0, len(action[0][0]), 4)]
        self.log("Formatting drop rate.")
        cnn_drop_rate = [c[3] for c in action]
        self.log("Loading graph.")
        with tf.Graph().as_default() as g:
            self.log(f"Creating container for the step: {step}")
            with g.container('experiment'+str(step)):
                model = ModelGenerator(self.num_input, self.num_classes, action)
                self.log("Create loss function for the child model.")
                loss_op = tf.reduce_mean(model.loss)
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                self.log("Create train step function for the child model.")
                train_op = optimizer.minimize(loss_op)

                self.log("Create a session.")
                with tf.Session() as train_sess:
                    self.log("Initialize tf variables.")
                    init = tf.global_variables_initializer()
                    train_sess.run(init)

                    self.log("Train the model.")
                    for step in range(self.max_step_per_action):
                        batch_x, batch_y = self.mnist.train.next_batch(self.bathc_size)
                        feed = {model.X: batch_x,
                                model.Y: batch_y,
                                model.dropout_keep_prob: self.dropout_rate,
                                model.cnn_dropout_rates: cnn_drop_rate}
                        _ = train_sess.run(train_op, feed_dict=feed)

                        if step % 100 == 0:
                            # Calculate batch loss and accuracy
                            loss, acc = train_sess.run(
                                [loss_op, model.accuracy],
                                feed_dict={model.X: batch_x,
                                           model.Y: batch_y,
                                           model.dropout_keep_prob: 1.0,
                                           model.cnn_dropout_rates: [1.0]*len(cnn_drop_rate)})
                            self.log("Step " + str(step) +
                                  ", Minibatch Loss= " + "{:.4f}".format(loss) +
                                  ", Current accuracy= " + "{:.3f}".format(acc))
                    batch_x, batch_y = self.mnist.test.next_batch(10000)
                    loss, acc = train_sess.run(
                                [loss_op, model.accuracy],
                                feed_dict={model.X: batch_x,
                                           model.Y: batch_y,
                                           model.dropout_keep_prob: 1.0,
                                           model.cnn_dropout_rates: [1.0]*len(cnn_drop_rate)})
                    self.log("!!!!!!acc:", acc, pre_acc)
                    if acc - pre_acc <= 0.01:
                        return acc, acc
                    else:
                        return 0.01, acc

    def log(self, log):
        print(f"ModelManager: {log}")
