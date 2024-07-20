#!/usr/bin/env python
"""Implementation of Item based AutoRec and user based AutoRec.
Reference: Sedhain, Suvash, et al. "Autorec: Autoencoders meet collaborative filtering." Proceedings of the 24th International Conference on World Wide Web. ACM, 2015.
"""

import tensorflow as tf
import time
import numpy as np
import scipy
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from tqdm import tqdm
import os


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
        self.rating_matrix = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.num_item, None])
        self.rating_matrix_mask = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.num_item, None])

        V = tf.Variable(tf.random.normal([hidden_neuron, self.num_item], stddev=0.01))
        W = tf.Variable(tf.random.normal([self.num_item, hidden_neuron], stddev=0.01))

        mu = tf.Variable(tf.random.normal([hidden_neuron], stddev=0.01))
        b = tf.Variable(tf.random.normal([self.num_item], stddev=0.01))
        layer_1 = tf.sigmoid(tf.expand_dims(mu, 1) + tf.matmul(V, self.rating_matrix))
        self.layer_2 = tf.matmul(W, layer_1) + tf.expand_dims(b, 1)
        self.loss = tf.reduce_mean(tf.square(
            tf.norm(tf.multiply((self.rating_matrix - self.layer_2), self.rating_matrix_mask)))) + self.reg_rate * (
        tf.square(tf.norm(W)) + tf.square(tf.norm(V)))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

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
        self.train_data_mask = np.sign(self.train_data)
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

        with tqdm(total=self.epochs, desc="Training", unit="epoch") as pbar:
            for epoch in range(self.epochs):
                avg_loss = self.train(train_data)
                if (epoch) % self.T == 0:
                    rmse, mae = self.test(test_data)
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

