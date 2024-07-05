import tensorflow as tf
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from tqdm import tqdm
import os
import matplotlib.pyplot as plt



dir_r3 = 'C:/Users/Sten Stokroos/Desktop/NEW/zelf/Data/out'
dir_ml = 'C:/Users/Sten Stokroos/Desktop/NEW/zelf/Data/out'
randseed = 42

def choose_data(dat, test_size=0.1):
    if dat == 'r3':
        train = pd.read_csv(os.path.join(dir_r3, 'r3_train.csv'), sep="\t", header=None, names=['userId', 'songId', 'rating'], usecols=[0, 1, 2], engine="python")
        test = pd.read_csv(os.path.join(dir_r3, 'r3_test.csv'), sep="\t", header=None, names=['userId', 'songId', 'rating'], usecols=[0, 1, 2], engine="python")
        
        # Combine train and test to create the full dataset
        r3_full = pd.concat([train, test]).sort_values(by=['userId', 'songId']).reset_index(drop=True)
        
        return r3_full, train, test
    elif dat == 'ml':
        ml_full = pd.read_csv(os.path.join(dir_ml, 'ml-1m_full.csv'), sep="\t", header=None, names=['userId', 'songId', 'rating'], usecols=[0, 1, 2], engine="python")
        train, test = train_test_split(ml_full, test_size=test_size, random_state=randseed)
        return ml_full, train, test
    else:
        print('Wrong data input')
        return None, None, None
    
def load_data_rating(dat, columns=[0, 1, 2], sep="\t"):
    full, train, test = choose_data(dat, test_size=0.1)
    
    train, vad = train_test_split(train, test_size=0.1, random_state=42)  # Create a validation set
    
    n_users = max(train['userId'].max(), test['userId'].max()) + 1
    n_items = max(train['songId'].max(), test['songId'].max()) + 1

    def create_matrix(df):
        rows = []
        cols = []
        ratings = []
        for line in df.itertuples():
            u = line[1]
            i = line[2]
            rows.append(u)
            cols.append(i)
            ratings.append(line[3])
        return csr_matrix((ratings, (rows, cols)), shape=(n_users, n_items))

    train_matrix = create_matrix(train)
    vad_matrix = create_matrix(vad)
    test_matrix = create_matrix(test)

    print("Load data finished. Number of users:", n_users, "Number of items:", n_items)
    return train_matrix.todok(), vad_matrix.todok(), test_matrix.todok(), n_users, n_items


class IAutoRec():
    def __init__(self, sess, num_user, num_item, learning_rate=0.001, reg_rate=0.1, epoch=500, batch_size=500,
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
        self.train_rmse_list = []
        self.val_rmse_list = []
        self.test_rmse_list = []
        print("IAutoRec with Confounder.")

    def build_network(self, hidden_neuron=500):
        self.rating_matrix = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.num_user, None])
        self.rating_matrix_mask = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.num_user, None])
        self.confounder_matrix = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.num_user, None])
        self.keep_rate_net = tf.compat.v1.placeholder(tf.float32)
        
        # Rating path
        V_R = tf.Variable(tf.random.normal([hidden_neuron, self.num_user], stddev=0.01))
        mu_R = tf.Variable(tf.random.normal([hidden_neuron], stddev=0.01))
        layer_1_R = tf.sigmoid(tf.expand_dims(mu_R, 1) + tf.matmul(V_R, self.rating_matrix))
        
        # Confounder path
        V_C = tf.Variable(tf.random.normal([hidden_neuron, self.num_user], stddev=0.01))
        mu_C = tf.Variable(tf.random.normal([hidden_neuron], stddev=0.01))
        layer_1_C = tf.sigmoid(tf.expand_dims(mu_C, 1) + tf.matmul(V_C, self.confounder_matrix))
        
        # Combine paths
        layer_1 = layer_1_R + layer_1_C
        
        # Apply dropout
        layer_1 = tf.nn.dropout(layer_1, rate=1 - self.keep_rate_net)
        
        # Output layer
        W = tf.Variable(tf.random.normal([self.num_user, hidden_neuron], stddev=0.01))
        b = tf.Variable(tf.random.normal([self.num_user], stddev=0.01))
        self.layer_2 = tf.matmul(W, layer_1) + tf.expand_dims(b, 1)
        self.loss = tf.reduce_mean(tf.square(
            tf.norm(tf.multiply((self.rating_matrix - self.layer_2), self.rating_matrix_mask)))) + self.reg_rate * (
        tf.square(tf.norm(W)) + tf.square(tf.norm(V_R)) + tf.square(tf.norm(V_C)))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self, train_data, confounder_data):
        self.num_training = self.num_item
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
                                                   self.confounder_matrix: confounder_data[:, batch_set_idx],
                                                   self.keep_rate_net: 0.95})
                total_loss += loss
            except IndexError as e:
                print(f"IndexError: {e}")
                print(f"Max index in batch_set_idx: {max(batch_set_idx)}")
                print(f"Train data shape: {self.train_data.shape}")
                print(f"Confounder data shape: {confounder_data.shape}")
                raise

        return total_loss / total_batch

    def calculate_rmse(self, data):
        reconstruction = self.sess.run(self.layer_2, feed_dict={self.rating_matrix: self.train_data,
                                                                self.rating_matrix_mask: self.train_data_mask,
                                                                self.confounder_matrix: confounder_data,
                                                                self.keep_rate_net: 1.0})
        error = 0
        data_set = list(data.keys())
        for (u, i) in data_set:
            pred_rating = reconstruction[u, i]
            error += (float(data.get((u, i))) - pred_rating) ** 2
        rmse = RMSE(error, len(data_set))
        return rmse

    # def validate(self, val_data):
    #     reconstruction = self.sess.run(self.layer_2, feed_dict={self.rating_matrix: self.train_data,
    #                                                             self.rating_matrix_mask: self.train_data_mask,
    #                                                             self.confounder_matrix: self.confounder_data,
    #                                                             self.keep_rate_net: 1.0})
    #     error = 0
    #     val_set = list(val_data.keys())
    #     for (u, i) in val_set:
    #         pred_rating = reconstruction[u, i]
    #         error += (float(val_data.get((u, i))) - pred_rating) ** 2
    #     rmse = RMSE(error, len(val_set))
    #     return rmse

    def test(self, test_data, confounder_data):
        self.reconstruction = self.sess.run(self.layer_2, feed_dict={self.rating_matrix: self.train_data,
                                                                     self.rating_matrix_mask: self.train_data_mask,
                                                                     self.confounder_matrix: confounder_data,
                                                                     self.keep_rate_net: 1.0})
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

    def execute(self, train_data, val_data, test_data, confounder_data):
        self.train_data = self._data_process(train_data)
        self.train_data_mask = np.sign(self.train_data)
        # self.confounder_data = self._data_process(confounder_data)
        print(f"Train data processed shape: {self.train_data.shape}")
        print(f"Confounder data shape: {confounder_data.shape}")
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

        with tqdm(total=self.epochs, desc="Training", unit="epoch") as pbar:
            for epoch in range(self.epochs):
                avg_loss = self.train(train_data, confounder_data)
                train_rmse = self.calculate_rmse(train_data)
                # val_rmse = self.validate(val_data)
                self.train_rmse_list.append(train_rmse)
                # self.val_rmse_list.append(val_rmse)
                if (epoch) % self.T == 0:
                    test_rmse, _ = self.test(test_data, confounder_data)
                    self.test_rmse_list.append(test_rmse)
                    pbar.set_postfix({"Train RMSE": train_rmse, "Val RMSE: NONE RIGHT NOW": 3 , "Test RMSE": test_rmse})
                pbar.update(1)

    def save(self, path):
        saver = tf.compat.v1.train.Saver()
        saver.save(self.sess, path)

    def predict(self, user_id, item_id):
        if user_id >= self.num_user or item_id >= self.num_item:
            raise IndexError("user_id or item_id out of bounds")
        return self.reconstruction[user_id, item_id]

    def _data_process(self, data):
        output = np.zeros((self.num_user, self.num_item))
        for u in range(self.num_user):
            for i in range(self.num_item):
                output[u, i] = data.get((u, i), 0)  # Use .get() with a default value of 0
        return output

    def plot_rmse(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_rmse_list, label='Training RMSE')
        plt.plot(self.val_rmse_list, label='Validation RMSE', color='green')
        plt.plot(range(0, self.epochs, self.T), self.test_rmse_list, label='Test RMSE', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('RMSE')
        plt.legend()
        plt.title('Training, Validation, and Test RMSE over Epochs')
        plt.show()

# Helper functions
def RMSE(error, num):
    return np.sqrt(error / num)

def MAE(error_mae, num):
    return (error_mae / num)

# Loading Data
train, test, val, user, item = load_data_rating('ml', columns=[0, 1, 2], sep="\t")

CAUSEFIT_DIR = 'C:/Users/Sten Stokroos/Desktop/NEW/zelf/Data/exposure_output/exp_ml_0.1_k30.csv'
conf_df = pd.read_csv(CAUSEFIT_DIR, header=None)
confounder_data = conf_df.to_numpy()

# Set TensorFlow session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

with tf.compat.v1.Session(config=config) as sess:
    model = IAutoRec(sess, user, item, learning_rate=0.001, reg_rate=0.1, epoch=100, batch_size=500, verbose=True)
    model.build_network()
    model.execute(train, val, test, confounder_data)
    model.plot_rmse()
