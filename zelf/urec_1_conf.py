import tensorflow as tf
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

class UAutoRec1conf():
    def __init__(self, sess, num_user, num_item, learning_rate=0.001, reg_rate=0.1, epoch=20, batch_size=200,
                 verbose=False, T=1, display_step=1000):
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.reg_rate = reg_rate
        self.sess = sess
        self.num_user = num_user
        self.num_item = num_item
        self.verbose = verbose
        self.T = T
        self.display_step = display_step
        self.train_loss_history = []
        self.test_rmse_history = []
        print("UAutoRec with Confounder.")

    def build_network(self, hidden_neuron=500):
        self.rating_matrix = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.num_item, None])
        self.rating_matrix_mask = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.num_item, None])
        self.confounder_matrix = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.num_item, None])

        # Rating path
        V_R = tf.Variable(tf.random.normal([hidden_neuron, self.num_item], stddev=0.01))
        mu_R = tf.Variable(tf.random.normal([hidden_neuron], stddev=0.01))
        layer_1_R = tf.sigmoid(tf.expand_dims(mu_R, 1) + tf.matmul(V_R, self.rating_matrix))
        
        # Confounder path
        V_C = tf.Variable(tf.random.normal([hidden_neuron, self.num_item], stddev=0.01))
        mu_C = tf.Variable(tf.random.normal([hidden_neuron], stddev=0.01))
        layer_1_C = tf.sigmoid(tf.expand_dims(mu_C, 1) + tf.matmul(V_C, self.confounder_matrix))
        
        # Combine paths
        layer_1 = layer_1_R + layer_1_C
        
        # Output layer
        W = tf.Variable(tf.random.normal([self.num_item, hidden_neuron], stddev=0.01))
        b = tf.Variable(tf.random.normal([self.num_item], stddev=0.01))
        self.layer_2 = tf.matmul(W, layer_1) + tf.expand_dims(b, 1)
        self.loss = tf.reduce_mean(tf.square(
            tf.norm(tf.multiply((self.rating_matrix - self.layer_2), self.rating_matrix_mask)))) + self.reg_rate * (
        tf.square(tf.norm(W)) + tf.square(tf.norm(V_R)) + tf.square(tf.norm(V_C)))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self, train_data, confounder_data):
        self.num_training = self.num_user
        total_batch = int(self.num_training / self.batch_size)
        idxs = np.random.permutation(self.num_training)  # shuffled ordering

        total_loss = 0
        for i in range(total_batch):
            start_time = time.time()
            if i == total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size:]
            elif i < total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size: (i + 1) * self.batch_size]

            try:
                _, loss = self.sess.run([self.optimizer, self.loss],
                                        feed_dict={self.rating_matrix: self.train_data[:, batch_set_idx],
                                                   self.rating_matrix_mask: self.train_data_mask[:, batch_set_idx],
                                                   self.confounder_matrix: confounder_data[:, batch_set_idx]})
                total_loss += loss
            except IndexError as e:
                print(f"IndexError: {e}")
                print(f"Max index in batch_set_idx: {max(batch_set_idx)}")
                print(f"Train data shape: {self.train_data.shape}")
                print(f"Confounder data shape: {confounder_data.shape}")
                raise
        avg_loss = total_loss / total_batch
        self.train_loss_history.append(avg_loss)
        return avg_loss


    def test(self, test_data, confounder_data):
        self.reconstruction = self.sess.run(self.layer_2, feed_dict={self.rating_matrix: self.train_data,
                                                                     self.rating_matrix_mask: self.train_data_mask,
                                                                     self.confounder_matrix: confounder_data})
        error = 0
        error_mae = 0
        test_set = list(test_data.keys())
        for (u, i) in test_set:
            pred_rating_test = self.predict(u, i)
            error += (float(test_data.get((u, i))) - pred_rating_test) ** 2
            error_mae += (np.abs(float(test_data.get((u, i))) - pred_rating_test))
        rmse = RMSE(error, len(test_set))
        mae = MAE(error_mae, len(test_set))
        self.test_rmse_history.append(rmse)
        return rmse, mae

    def execute(self, train_data, test_data, confounder_data):
        self.train_data = self._data_process(train_data.transpose())
        self.train_data_mask = np.sign(self.train_data)
        print(f"Train data processed shape: {self.train_data.shape}")
        print(f"Confounder data shape: {confounder_data.shape}")
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

        with tqdm(total=self.epochs, desc="Training", unit="epoch") as pbar:
            for epoch in range(self.epochs):
                avg_loss = self.train(train_data, confounder_data)
                if (epoch) % self.T == 0:
                    rmse, mae = self.test(test_data, confounder_data)
                    pbar.set_postfix({"Loss": avg_loss, "RMSE": rmse, "MAE": mae})
                pbar.update(1)

    def save(self, path):
        saver = tf.compat.v1.train.Saver()
        saver.save(self.sess, path)

    def predict(self, user_id, item_id):
        if user_id >= self.num_user or item_id >= self.num_item:
            raise IndexError("user_id or item_id out of bounds")
        return self.reconstruction[item_id, user_id]

    def _data_process(self, data):
        output = np.zeros((self.num_item, self.num_user))
        for u in range(self.num_user):
            for i in range(self.num_item):
                output[i, u] = data.get((i, u), 0)  # Use .get() with a default value of 0
        return output

def RMSE(error, num):
    return np.sqrt(error / num)

def MAE(error_mae, num):
    return (error_mae / num)