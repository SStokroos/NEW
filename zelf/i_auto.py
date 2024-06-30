import tensorflow as tf
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from tqdm import tqdm

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
        print("IAutoRec with Confounder and Exposure.")

    def build_network(self, hidden_neuron=500):
        self.rating_matrix = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.num_user, None])
        self.rating_matrix_mask = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.num_user, None])
        self.confounder_matrix = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.num_user, None])
        self.exposure_matrix = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.num_user, None])
        self.keep_rate_net = tf.compat.v1.placeholder(tf.float32)
        
        # Rating path
        V_R = tf.Variable(tf.random.normal([hidden_neuron, self.num_user], stddev=0.01))
        mu_R = tf.Variable(tf.random.normal([hidden_neuron], stddev=0.01))
        layer_1_R = tf.sigmoid(tf.expand_dims(mu_R, 1) + tf.matmul(V_R, self.rating_matrix))
        
        # Confounder path
        V_C = tf.Variable(tf.random.normal([hidden_neuron, self.num_user], stddev=0.01))
        mu_C = tf.Variable(tf.random.normal([hidden_neuron], stddev=0.01))
        layer_1_C = tf.sigmoid(tf.expand_dims(mu_C, 1) + tf.matmul(V_C, self.confounder_matrix))
        
        # Exposure path
        V_E = tf.Variable(tf.random.normal([hidden_neuron, self.num_user], stddev=0.01))
        mu_E = tf.Variable(tf.random.normal([hidden_neuron], stddev=0.01))
        layer_1_E = tf.sigmoid(tf.expand_dims(mu_E, 1) + tf.matmul(V_E, self.exposure_matrix))
        
        # Combine paths
        layer_1 = layer_1_R + layer_1_C + layer_1_E
        
        # Apply dropout
        layer_1 = tf.nn.dropout(layer_1, rate=1 - self.keep_rate_net)
        
        # Output layer
        W = tf.Variable(tf.random.normal([self.num_user, hidden_neuron], stddev=0.01))
        b = tf.Variable(tf.random.normal([self.num_user], stddev=0.01))
        self.layer_2 = tf.matmul(W, layer_1) + tf.expand_dims(b, 1)
        self.loss = tf.reduce_mean(tf.square(
            tf.norm(tf.multiply((self.rating_matrix - self.layer_2), self.rating_matrix_mask)))) + self.reg_rate * (
        tf.square(tf.norm(W)) + tf.square(tf.norm(V_R)) + tf.square(tf.norm(V_C)) + tf.square(tf.norm(V_E)))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self, train_data, confounder_data, exposure_data):
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
                                                   self.exposure_matrix: exposure_data[:, batch_set_idx],
                                                   self.keep_rate_net: 0.95})
                total_loss += loss
            except IndexError as e:
                print(f"IndexError: {e}")
                print(f"Max index in batch_set_idx: {max(batch_set_idx)}")
                print(f"Train data shape: {self.train_data.shape}")
                print(f"Confounder data shape: {confounder_data.shape}")
                raise

        return total_loss / total_batch

    def test(self, test_data, confounder_data, exposure_data):
        self.reconstruction = self.sess.run(self.layer_2, feed_dict={self.rating_matrix: self.train_data,
                                                                     self.rating_matrix_mask: self.train_data_mask,
                                                                     self.confounder_matrix: confounder_data,
                                                                     self.exposure_matrix: exposure_data,
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

    def execute(self, train_data, test_data, confounder_data, exposure_data):
        self.train_data = self._data_process(train_data)
        self.train_data_mask = np.sign(self.train_data)
        print(f"Train data processed shape: {self.train_data.shape}")
        print(f"Confounder data shape: {confounder_data.shape}")
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

        with tqdm(total=self.epochs, desc="Training", unit="epoch") as pbar:
            for epoch in range(self.epochs):
                avg_loss = self.train(train_data, confounder_data, exposure_data)
                if (epoch) % self.T == 0:
                    rmse, mae = self.test(test_data, confounder_data, exposure_data)
                    pbar.set_postfix({"Loss": avg_loss, "RMSE": rmse, "MAE": mae})
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

# Helper functions
def RMSE(error, num):
    return np.sqrt(error / num)

def MAE(error_mae, num):
    return (error_mae / num)

# Data loading function
def load_data_rating(train_file, test_file, columns=[0, 1, 2], sep="\t"):
    tr_vd_dat = pd.read_csv(train_file, sep=sep, header=None, names=['userId', 'itemId', 'rating'], usecols=columns, engine="python")
    train_data, vd_data = train_test_split(tr_vd_dat, test_size=0.2, random_state=42)
    test_data = pd.read_csv(test_file, sep=sep, header=None, names=['userId', 'itemId', 'rating'], usecols=columns, engine="python")

    n_users = max(train_data['userId'].max(), test_data['userId'].max()) + 1
    n_items = max(train_data['itemId'].max(), test_data['itemId'].max()) + 1

    train_row = []
    train_col = []
    train_rating = []

    for line in train_data.itertuples():
        u = line[1]
        i = line[2]
        train_row.append(u)
        train_col.append(i)
        train_rating.append(line[3])

    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))

    test_row = []
    test_col = []
    test_rating = []
    for line in test_data.itertuples():
        u = line[1]
        i = line[2]
        test_row.append(u)
        test_col.append(i)
        test_rating.append(line[3])

    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))

    vd_row = []
    vd_col = []
    vd_rating = []
    for line in vd_data.itertuples():
        u = line[1]
        i = line[2]
        vd_row.append(u)
        vd_col.append(i)
        vd_rating.append(line[3])

    vd_matrix = csr_matrix((vd_rating, (vd_row, vd_col)), shape=(n_users, n_items))

    print("Load data finished. Number of users:", n_users, "Number of items:", n_items)
    return train_matrix.todok(), test_matrix.todok(), vd_matrix.todok(), n_users, n_items

# Example usage
file1 = 'C:/Users/Sten Stokroos/Desktop/Thesis2.0/zelf/neural_collaborative_filtering/Data/ml-1m.train.rating'  # Replace with the actual file path
file2 = 'C:/Users/Sten Stokroos/Desktop/Thesis2.0/zelf/neural_collaborative_filtering/Data/ml-1m.test.rating' # Replace with the actual file path

train, test, vad, user, item = load_data_rating(file1, file2, columns=[0, 1, 2], sep="\t")

CAUSEFIT_DIR = 'C:/Users/Sten Stokroos/Desktop/Thesis2.0/zelf/xposure_matrix.csv'

conf_df = pd.read_csv(CAUSEFIT_DIR, header=None)

# Convert the DataFrame to a NumPy array
confounder_data = conf_df.to_numpy()
confounder_data = confounder_data

# Create the exposure data matrix
exposure_data = (train > 0).astype(np.float32).todense()

# Set TensorFlow session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

with tf.compat.v1.Session(config=config) as sess:
    model = IAutoRec(sess, user, item, learning_rate=0.001, reg_rate=0.1, epoch=20, batch_size=500, verbose=True)
    model.build_network()
    model.execute(train, test, confounder_data, exposure_data)
