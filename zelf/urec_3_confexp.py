import tensorflow as tf
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

class UAutoRec3confexp():
    def __init__(self, sess, num_user, num_item, hidden_neuron=500, learning_rate=0.001, reg_rate=0.1, epoch=500, batch_size=200,
                 verbose=False, T=3, display_step=1000):
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.reg_rate = reg_rate
        self.sess = sess
        self.num_user = num_user
        self.num_item = num_item
        self.hidden_neuron = hidden_neuron
        self.verbose = verbose
        self.T = T
        self.display_step = display_step
        self.train_loss_history = []
        self.test_rmse_history = []
        print("UAutoRec with Confounder and Exposure.")

    def build_network(self, hidden_neuron=500):
        self.rating_matrix = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.num_item, None])
        self.rating_matrix_mask = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.num_item, None])
        self.confounder_matrix = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.num_item, None])
        self.exposure_matrix = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.num_item, None])

        # Rating path
        V_R = tf.Variable(tf.random.normal([hidden_neuron, self.num_item], stddev=0.01))
        mu_R = tf.Variable(tf.random.normal([hidden_neuron], stddev=0.01))
        
        # Nonlinear combination with confounder and exposure
        combined_input = self.rating_matrix + self.confounder_matrix + self.exposure_matrix
        layer_1 = tf.sigmoid(tf.expand_dims(mu_R, 1) + tf.matmul(V_R, combined_input))
        
        # Output layer
        W = tf.Variable(tf.random.normal([self.num_item, hidden_neuron], stddev=0.01))
        b = tf.Variable(tf.random.normal([self.num_item], stddev=0.01))
        self.layer_2 = tf.matmul(W, layer_1) + tf.expand_dims(b, 1)
        
        # Loss function
        self.loss = tf.reduce_mean(tf.square(
            tf.norm(tf.multiply((self.rating_matrix - self.layer_2), self.rating_matrix_mask)))) + self.reg_rate * (
            tf.square(tf.norm(W)) + tf.square(tf.norm(V_R)))
        
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self, train_data, confounder_data, exposure_data):
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
            
            # print(f"Train data shape: {self.train_data.shape}")
            # print(f"Confounder data shape: {confounder_data.shape}")

            try:
                _, loss = self.sess.run([self.optimizer, self.loss],
                                        feed_dict={self.rating_matrix: self.train_data[:, batch_set_idx],
                                                   self.rating_matrix_mask: self.train_data_mask[:, batch_set_idx],
                                                   self.confounder_matrix: confounder_data[:, batch_set_idx],
                                                   self.exposure_matrix: exposure_data[:, batch_set_idx]})
                
                total_loss += loss
            except IndexError as e:
                print(f"IndexError: {e}")
                print(f"Batch set idx: {batch_set_idx}")
                print(f"Max index in batch_set_idx: {max(batch_set_idx)}")
                print(f"Train data shape: {self.train_data.shape}")
                print(f"Confounder data shape: {confounder_data.shape}")
                raise

        avg_loss = total_loss / total_batch
        self.train_loss_history.append(avg_loss)
        return avg_loss

    def test(self, test_data, confounder_data, exposure_data):
        self.reconstruction = self.sess.run(self.layer_2, feed_dict={self.rating_matrix: self.train_data,
                                                                     self.rating_matrix_mask: self.train_data_mask,
                                                                     self.confounder_matrix: confounder_data,
                                                                     self.exposure_matrix: exposure_data})
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

    def execute(self, train_data, test_data, confounder_data, exposure_data): #, patience = 10):
        self.train_data = self._data_process(train_data.transpose())
        self.train_data_mask = np.sign(self.train_data)
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

        # best_rmse = float('inf')
        # epochs_no_improve = 0


        with tqdm(total=self.epochs, desc="Training", unit="epoch") as pbar:
            for epoch in range(self.epochs):
                avg_loss = self.train(train_data, confounder_data, exposure_data)
                if (epoch) % self.T == 0:
                    rmse, mae = self.test(test_data, confounder_data, exposure_data)
                    pbar.set_postfix({"Loss": avg_loss, "RMSE": rmse, "MAE": mae})

                    # # Check for improvement
                    # if rmse < best_rmse:
                    #     best_rmse = rmse
                    #     epochs_no_improve = 0
                    # else:
                    #     epochs_no_improve += 1

                    # # Early stopping
                    # if epochs_no_improve >= patience:
                    #     print(f"Early stopping at epoch {epoch}. Best RMSE: {best_rmse}")
                    #     break

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