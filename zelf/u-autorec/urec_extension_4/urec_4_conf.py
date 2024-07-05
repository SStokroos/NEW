import tensorflow as tf
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from tqdm import tqdm
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import os



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
        print("UAutoRec with Confounder as Bias.")

    def build_network(self, hidden_neuron=500):
        self.rating_matrix = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.num_item, None])
        self.rating_matrix_mask = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.num_item, None])
        self.confounder_matrix = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.num_item, None])

        V = tf.Variable(tf.random.normal([hidden_neuron, self.num_item], stddev=0.01))
        W = tf.Variable(tf.random.normal([self.num_item, hidden_neuron], stddev=0.01))

        # Use the confounder matrix as the bias term
        layer_1 = tf.sigmoid(self.confounder_matrix + tf.matmul(V, self.rating_matrix))
        self.layer_2 = tf.matmul(W, layer_1)
        
        self.loss = tf.reduce_mean(tf.square(
            tf.norm(tf.multiply((self.rating_matrix - self.layer_2), self.rating_matrix_mask)))) + self.reg_rate * (
        tf.square(tf.norm(W)) + tf.square(tf.norm(V)))
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

            _, loss = self.sess.run([self.optimizer, self.loss],
                                    feed_dict={self.rating_matrix: self.train_data[:, batch_set_idx],
                                               self.rating_matrix_mask: self.train_data_mask[:, batch_set_idx],
                                               self.confounder_matrix: confounder_data[:, batch_set_idx]
                                               })
            total_loss += loss

        return total_loss / total_batch

    def test(self, test_data, confounder_data):
        self.reconstruction = self.sess.run(self.layer_2, feed_dict={self.rating_matrix: self.train_data,
                                                                     self.rating_matrix_mask:
                                                                         self.train_data_mask,
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
        return rmse, mae

    def execute(self, train_data, test_data, confounder_data):
        self.train_data = self._data_process(train_data.transpose())
        self.train_data_mask = np.sign(self.train_data)
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

def load_data_rating(dat, columns=[0, 1, 2], sep="\t"):
    full, train, test = choose_data(dat, test_size= 0.1)
    
    n_users = max(train['userId'].max(), test['userId'].max()) + 1
    n_items = max(train['songId'].max(), test['songId'].max()) + 1

    train_row = []
    train_col = []
    train_rating = []

    for line in train.itertuples():
        u = line[1]
        i = line[2]
        train_row.append(u)
        train_col.append(i)
        train_rating.append(line[3])

    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))

    test_row = []
    test_col = []
    test_rating = []
    for line in test.itertuples():
        u = line[1]
        i = line[2]
        test_row.append(u)
        test_col.append(i)
        test_rating.append(line[3])

    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))

    print("Load data finished. Number of users:", n_users, "Number of items:", n_items)
    return train_matrix.todok(), test_matrix.todok(), n_users, n_items

# Assuming confounder data is loaded from somewhere
CAUSEFIT_DIR = 'C:/Users/Sten Stokroos/Desktop/NEW/zelf/Data/exposure_output/exp_ml_0.1_k30.csv'
conf_df = pd.read_csv(CAUSEFIT_DIR, header=None)
confounder_data = conf_df.to_numpy()
confounder_data = confounder_data.T

train, test, user, item = load_data_rating('ml', columns=[0, 1, 2], sep="\t")

# Set TensorFlow session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

with tf.compat.v1.Session(config=config) as sess:
    model = UAutoRec(sess, user, item, learning_rate=0.001, reg_rate=0.1, epoch=50, batch_size=500, verbose=True)
    model.build_network()
    model.execute(train, test, confounder_data)
    
    final_rmse, final_mae = model.test(test, confounder_data)
    print(f"Final RMSE: {final_rmse}, Final MAE: {final_mae}")
