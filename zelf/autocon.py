#!/usr/bin/env python
"""Implementation of User based AutoRec with confounders.
"""

import tensorflow as tf
import time
import numpy as np
import scipy
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from tqdm import tqdm

class UAutoRec():
    def __init__(self, sess, num_user, num_item, learning_rate=0.001, reg_rate=0.1, epoch=500, batch_size=200,
                 verbose=False, T=3, display_step=1000):
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
        print("UAutoRec.")

    def build_network(self, hidden_neuron=500):
        self.rating_matrix = tf.placeholder(dtype=tf.float32, shape=[self.num_item, None])
        self.rating_matrix_mask = tf.placeholder(dtype=tf.float32, shape=[self.num_item, None])

        V = tf.Variable(tf.random_normal([hidden_neuron, self.num_item], stddev=0.01))
        W = tf.Variable(tf.random_normal([self.num_item, hidden_neuron], stddev=0.01))

        mu = tf.Variable(tf.random_normal([hidden_neuron], stddev=0.01))
        b = tf.Variable(tf.random_normal([self.num_item], stddev=0.01))
        layer_1 = tf.sigmoid(tf.expand_dims(mu, 1) + tf.matmul(V, self.rating_matrix))
        self.layer_2 = tf.matmul(W, layer_1) + tf.expand_dims(b, 1)
        self.loss = tf.reduce_mean(tf.square(
            tf.norm(tf.multiply((self.rating_matrix - self.layer_2), self.rating_matrix_mask)))) + self.reg_rate * (
        tf.square(tf.norm(W)) + tf.square(tf.norm(V)))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self, train_data):
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

            _, loss = self.sess.run([self.optimizer, self.loss],
                                    feed_dict={self.rating_matrix: self.train_data[:, batch_set_idx],
                                               self.rating_matrix_mask: self.train_data_mask[:, batch_set_idx]
                                               })
            total_loss += loss

        return total_loss / total_batch

    def test(self, test_data):
        self.reconstruction = self.sess.run(self.layer_2, feed_dict={self.rating_matrix: self.train_data,
                                                                     self.rating_matrix_mask:
                                                                         self.train_data_mask})
        error = 0
        error_mae = 0
        test_set = list(test_data.keys())
        for (u, i) in test_set:
            pred_rating_test = self.predict(u, i)
            error += (float(test_data.get((u, i))) - pred_rating_test) ** 2
            error_mae += (np.abs(float(test_data.get((u, i))) - pred_rating_test))
        rmse = RMSE(error, len(test_set))
        mae = MAE(error_mae, len(test_set))
        return rmse, mae

    def execute(self, train_data, test_data):
        self.train_data = self._data_process(train_data.transpose())
        self.train_data_mask = scipy.sign(self.train_data)
        init = tf.global_variables_initializer()
        self.sess.run(init)

        with tqdm(total=self.epochs, desc="Training", unit="epoch") as pbar:
            for epoch in range(self.epochs):
                avg_loss = self.train(train_data)
                if (epoch) % self.T == 0:
                    rmse, mae = self.test(test_data)
                    pbar.set_postfix({"Loss": avg_loss, "RMSE": rmse, "MAE": mae})
                pbar.update(1)

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def predict(self, user_id, item_id):
        return self.reconstruction[item_id, user_id]

    def _data_process(self, data):
        output = np.zeros((self.num_item, self.num_user))
        for u in range(self.num_user):
            for i in range(self.num_item):
                output[i, u] = data.get((i, u))
        return output

class UAutoRecWithConfounders(UAutoRec):
    def __init__(self, sess, num_user, num_item, confounders, learning_rate=0.001, reg_rate=0.1, epoch=500, batch_size=200,
                 verbose=False, T=3, display_step=1000):
        super().__init__(sess, num_user, num_item, learning_rate, reg_rate, epoch, batch_size, verbose, T, display_step)
        self.confounders = confounders
        print("UAutoRec with Confounders.")

    def build_network(self, hidden_neuron=500):
        self.rating_matrix = tf.placeholder(dtype=tf.float32, shape=[self.num_item, None])
        self.rating_matrix_mask = tf.placeholder(dtype=tf.float32, shape=[self.num_item, None])
        self.confounder_matrix = tf.placeholder(dtype=tf.float32, shape=[self.num_item, None])  # Confounder input

        V = tf.Variable(tf.random_normal([hidden_neuron, self.num_item], stddev=0.01))
        W = tf.Variable(tf.random_normal([self.num_item, hidden_neuron], stddev=0.01))

        mu = tf.Variable(tf.random_normal([hidden_neuron], stddev=0.01))
        b = tf.Variable(tf.random_normal([self.num_item], stddev=0.01))
        layer_1 = tf.sigmoid(tf.expand_dims(mu, 1) + tf.matmul(V, self.rating_matrix + self.confounder_matrix))
        self.layer_2 = tf.matmul(W, layer_1) + tf.expand_dims(b, 1)
        self.loss = tf.reduce_mean(tf.square(
            tf.norm(tf.multiply((self.rating_matrix - self.layer_2), self.rating_matrix_mask)))) + self.reg_rate * (
        tf.square(tf.norm(W)) + tf.square(tf.norm(V)))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self, train_data):
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

            batch_confounders = self.confounders[:, batch_set_idx]
            _, loss = self.sess.run([self.optimizer, self.loss],
                                    feed_dict={self.rating_matrix: self.train_data[:, batch_set_idx],
                                               self.rating_matrix_mask: self.train_data_mask[:, batch_set_idx],
                                               self.confounder_matrix: batch_confounders
                                               })
            total_loss += loss

        return total_loss / total_batch

    def test(self, test_data):
        reconstruction = self.sess.run(self.layer_2, feed_dict={self.rating_matrix: self.train_data,
                                                                     self.rating_matrix_mask: self.train_data_mask,
                                                                     self.confounder_matrix: self.confounders})
        error = 0
        error_mae = 0
        test_set = list(test_data.keys())
        for (u, i) in test_set:
            pred_rating_test = self.predict(u, i)
            error += (float(test_data.get((u, i))) - pred_rating_test) ** 2
            error_mae += (np.abs(float(test_data.get((u, i))) - pred_rating_test))
        rmse = np.sqrt(error / len(test_set))
        mae = error_mae / len(test_set)
        return rmse, mae

def RMSE(error, num):
    return np.sqrt(error / num)

def MAE(error_mae, num):
    return (error_mae / num)

def load_data_with_confounders(rating_path, confounder_path, columns=[0, 1, 2], test_size=0.1, sep="\t"):
    df = pd.read_csv(rating_path, sep=sep, header=None, names=['userId', 'songId', 'rating'], usecols=columns, engine="python")

    print(df.head())

    n_users = df['userId'].unique().shape[0]
    n_items = df['songId'].unique().shape[0]

    print('Number of users:', n_users)
    print('Number of items:', n_items)
    train_data, test_data = train_test_split(df, test_size=test_size)

    train_row = []
    train_col = []
    train_rating = []

    max_user_id = df['userId'].max()
    max_item_id = df['songId'].max()

    print(f"Max userId: {max_user_id}, Max songId: {max_item_id}")

    for line in train_data.itertuples():
        u = line[1]
        i = line[2]
        train_row.append(u)
        train_col.append(i)
        train_rating.append(line[3])

    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(max_user_id + 1, max_item_id + 1))

    test_row = []
    test_col = []
    test_rating = []
    for line in test_data.itertuples():
        u = line[1]
        i = line[2]
        test_row.append(u)
        test_col.append(i)
        test_rating.append(line[3])

    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(max_user_id + 1, max_item_id + 1))

    print("Load data finished. Number of users:", n_users, "Number of items:", n_items)

    U = np.loadtxt(confounder_path + '_U.csv')
    B = np.loadtxt(confounder_path + '_V.csv')
    U = (np.atleast_2d(U.T).T)
    B = (np.atleast_2d(B.T).T)
    confounder_matrix = U.dot(B.T)

    print(f"Confounder matrix shape: {confounder_matrix.shape}")

    return train_matrix.todok(), test_matrix.todok(), max_user_id + 1, max_item_id + 1, confounder_matrix

rating_path = 'C:/Users/Sten Stokroos/Desktop/zelf/neural_collaborative_filtering/Data/ml-1m.train.rating'
confounder_path = 'C:/Users/Sten Stokroos/Desktop/Thesis2.0/zelf/dat/proc/ml_wg/train_full'
train, test, user, item, confounders = load_data_with_confounders(rating_path, confounder_path, columns=[0, 1, 2], test_size=0.1, sep="\t")

# Set TensorFlow session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    model = UAutoRecWithConfounders(sess, user, item, confounders, learning_rate=0.001, reg_rate=0.1, epoch=20, batch_size=500, verbose=True)
    model.build_network()
    model.execute(train, test)
