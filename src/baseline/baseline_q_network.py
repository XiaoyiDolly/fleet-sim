import os
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense
from . import settings
from .settings import FLAGS, MAIN_DEPTH, MAIN_LENGTH, AUX_DEPTH, AUX_LENGTH
from config.settings import MAP_WIDTH, MAP_HEIGHT, GLOBAL_STATE_UPDATE_CYCLE,\
    DESTINATION_PROFILE_TEMPORAL_AGGREGATION, DESTINATION_PROFILE_SPATIAL_AGGREGATION


class DeepQNetwork(object):
    def __init__(self, network_path=None):``
        # self.sa_input, self.q_values, self.model = self.build_q_network()
        self.main_input, self.aux_input, self.q_values, self.model = self.build_q_network()

        if not os.path.exists(FLAGS.save_network_dir):
            os.makedirs(FLAGS.save_network_dir)
        self.saver = tf.train.Saver(self.model.trainable_weights)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        if network_path:
            self.load_network(network_path)

    def build_q_network(self):
        # sa_input = Input(shape=(settings.NUM_FEATURES, ), dtype='float32')
        # x = Dense(100, activation='relu', name='dense_1')(sa_input)
        # x = Dense(100, activation='relu', name='dense_2')(x)
        # q_value = Dense(1, name='q_value')(x)
        # model = Model(inputs=sa_input, outputs=q_value)
        # return sa_input, q_value, model

        main_input = Input(shape=(MAIN_DEPTH, MAIN_LENGTH, MAIN_LENGTH), dtype='float32')
        aux_input = Input(shape=(AUX_DEPTH, AUX_LENGTH, AUX_LENGTH), dtype='float32')

        c = OUTPUT_LENGTH / 2
        sliced_input = Lambda(lambda x: x[:, :-1, :, :])(main_input)
        ave = AveragePooling2D(pool_size=(OUTPUT_LENGTH, OUTPUT_LENGTH), strides=(1, 1))(sliced_input)
        ave1 = Cropping2D(cropping=((c, c), (c, c)))(ave)
        ave2 = AveragePooling2D(pool_size=(OUTPUT_LENGTH, OUTPUT_LENGTH), strides=(1, 1))(ave)
        gra = Cropping2D(cropping=((c * 2, c * 2), (c * 2, c * 2)))(main_input)

        merge1 = merge([gra, ave1, ave2], mode='concat', concat_axis=1)
        x = Convolution2D(16, 5, 5, activation='relu', name='main/conv_1')(merge1)
        x = Convolution2D(32, 3, 3, activation='relu', name='main/conv_2')(x)
        main_output = Convolution2D(64, 3, 3, activation='relu', name='main/conv_3')(x)
        aux_output = Convolution2D(16, 1, 1, activation='relu', name='ayx/conv')(aux_input)
        merge2 = merge([main_output, aux_output], mode='concat', concat_axis=1)
        x = Convolution2D(128, 1, 1, activation='relu', name='merge/conv')(merge2)
        x = Convolution2D(1, 1, 1, name='main/q_value')(x)
        z = Flatten()(x)
        legal = Flatten()(Lambda(lambda x: x[:, -1:, :, :])(aux_input))
        q_values = merge([z, legal], mode='mul')

        model = Model(input=[main_input, aux_input], output=q_values)

        return main_input, aux_input, q_values, model


    def load_network(self, network_path):
        self.saver.restore(self.sess, network_path)
        print('Successfully loaded: ' + network_path)
        # checkpoint = tf.train.get_checkpoint_state(FLAGS.save_network_dir)
        # if checkpoint and checkpoint.model_checkpoint_path:
        #     self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
        #     print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        # else:
        #     print('Loading failed')


    def compute_q_values(self, s):
        main_features = s[:NUM_SUPPLY_DEMAND_MAPS]
        aux_features = s[NUM_SUPPLY_DEMAND_MAPS:]
        # q = self.q_values.eval(
        #     feed_dict={
        #         self.sa_input: np.array([s_feature + a_feature for a_feature in a_features], dtype=np.float32)
        #     })[:, 0]
        q = self.q_values.eval(
            feed_dict={
                self.main_input: np.array([main_features], dtype=np.float32),
                self.aux_input: np.array([aux_features], dtype=np.float32)
            })[:, 0]
        return q

    def get_action(self, q_values, amax):
        if FLAGS.alpha > 0:
            exp_q = np.exp((q_values - q_values[amax]) / FLAGS.alpha)
            p = exp_q / exp_q.sum()
            return np.random.choice(len(p), p=p)
        else:
            return amax


class FittingDeepQNetwork(DeepQNetwork):

    def __init__(self, network_path=None):
        super().__init__(network_path)

        model_weights = self.model.trainable_weights
        # Create target network
        self.target_main_input, self.target_aux_input, self.target_q_values, self.target_model = self.build_q_network()
        target_model_weights = self.target_model.trainable_weights

        # Define target network update operation
        self.update_target_network = [target_model_weights[i].assign(model_weights[i]) for i in
                                      range(len(target_model_weights))]

        # Define loss and gradient update operation
        self.y, self.loss, self.grad_update = self.build_training_op(model_weights)
        self.sess.run(tf.global_variables_initializer())

        # if load_network:
        #     self.load_network()
        # Initialize target network
        self.sess.run(self.update_target_network)

        self.n_steps = 0
        self.epsilon = settings.INITIAL_EPSILON
        self.epsilon_step = (settings.FINAL_EPSILON - settings.INITIAL_EPSILON) / settings.EXPLORATION_STEPS


        for var in model_weights:
            tf.summary.histogram(var.name, var)
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(FLAGS.save_summary_dir, self.sess.graph)


    def get_action(self, q_values, amax):
        # e-greedy exploration
        if self.epsilon > np.random.random():
            return np.random.randint(len(q_values))
        else:
            return super().get_action(q_values, amax)

    def get_fingerprint(self):
        return self.n_steps, self.epsilon

    def compute_target_q_values(self, s):
        main_features = s[:NUM_SUPPLY_DEMAND_MAPS]
        aux_features = s[NUM_SUPPLY_DEMAND_MAPS:]
        q = self.target_q_values.eval(
            feed_dict={
                self.target_main_input: np.array([main_features], dtype=np.float32),
                self.target_aux_input: np.array([aux_features], dtype=np.float32)
            })[:, 0]
        return q

    def compute_target_value(self, s):
        Q = self.compute_target_q_values(s)
        amax = np.argmax(self.compute_q_values(s))
        V = Q[amax]
        if FLAGS.alpha > 0:
            V += FLAGS.alpha * np.log(np.exp((Q - Q.max()) / FLAGS.alpha).sum())
        return V


    def fit(self, main_feature_batch, aux_feature_batch, y_batch):
        loss, _ = self.sess.run([self.loss, self.grad_update], feed_dict={
            self.main_input: np.array(main_feature_batch, dtype=np.float32),
            self.aux_input: np.array(aux_feature_batch, dtype=np.float32),
            self.y: np.array(y_batch, dtype=np.float32)
        })
        return loss

    def run_cyclic_updates(self):
        self.n_steps += 1
        # Update target network
        if self.n_steps % settings.TARGET_UPDATE_INTERVAL == 0:
            self.sess.run(self.update_target_network)
            print("Update target network")

        # Save network
        if self.n_steps % settings.SAVE_INTERVAL == 0:
            save_path = self.saver.save(self.sess, os.path.join(FLAGS.save_network_dir, "model"), global_step=(self.n_steps))
            print('Successfully saved: ' + save_path)

        # Anneal epsilon linearly over time
        if self.n_steps < settings.EXPLORATION_STEPS:
            self.epsilon += self.epsilon_step


    def build_training_op(self, q_network_weights):
        # y = tf.placeholder(tf.float32, shape=(None))
        # q_value = tf.reduce_sum(self.q_values, reduction_indices=1)
        # loss = tf.losses.huber_loss(y, q_value)
        # optimizer = tf.train.RMSPropOptimizer(settings.LEARNING_RATE, momentum=settings.MOMENTUM, epsilon=settings.MIN_GRAD)
        # grad_update = optimizer.minimize(loss, var_list=q_network_weights)
        #
        # return y, loss, grad_update
        
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector
        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)
        q_value = tf.reduce_sum(tf.mul(self.q_values, a_one_hot), reduction_indices=1)

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error = tf.abs(y - q_value)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD)
        grad_update = optimizer.minimize(loss, var_list=q_network_weights)

        return a, y, loss, grad_update

    def setup_summary(self):
        avg_max_q = tf.Variable(0.)
        tf.summary.scalar('Average_Max_Q', avg_max_q)
        avg_loss = tf.Variable(0.)
        tf.summary.scalar('Average_Loss', avg_loss)
        summary_vars = [avg_max_q, avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def write_summary(self, avg_loss, avg_q_max):
        stats = [avg_q_max, avg_loss]
        for i in range(len(stats)):
            self.sess.run(self.update_ops[i], feed_dict={
                self.summary_placeholders[i]: float(stats[i])
            })
        summary_str = self.sess.run(self.summary_op)
        self.summary_writer.add_summary(summary_str, self.n_steps)
