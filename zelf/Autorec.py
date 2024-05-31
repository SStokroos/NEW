import tensorflow as tf
import time
import numpy as np
import scipy
from tqdm import tqdm
import pandas as pd
import os

def prepare_data(num_user, num_item, train_file, test_file, confounder_file):
    # Load train and test data
    train_df = pd.read_csv(train_file, sep="\t", header=None, names=['userId', 'songId', 'rating'], usecols=[0, 1, 2], engine="python")
    test_df = pd.read_csv(test_file, sep="\t", header=None, names=['userId', 'songId', 'rating'], usecols=[0, 1, 2], engine="python")
    
    # Load confounder data
    U = np.loadtxt(confounder_file + '_U.csv')
    B = np.loadtxt(confounder_file + '_V.csv')
    U = (np.atleast_2d(U.T).T)
    B = (np.atleast_2d(B.T).T)
    confounder_matrix = U.dot(B.T)
    
    # Adjust the confounder matrix to match the number of items
    if confounder_matrix.shape[1] < num_item:
        # Pad the confounder matrix with zeros to match the number of items
        padding = np.zeros((confounder_matrix.shape[0], num_item - confounder_matrix.shape[1]))
        confounder_matrix = np.hstack((confounder_matrix, padding))
    elif confounder_matrix.shape[1] > num_item:
        # Truncate the confounder matrix to match the number of items
        confounder_matrix = confounder_matrix[:, :num_item]
    
    # Initialize matrices
    train_matrix = np.zeros((num_user, num_item), dtype=np.float32)
    test_matrix = np.zeros((num_user, num_item), dtype=np.float32)
    train_mask = np.zeros((num_user, num_item), dtype=np.float32)
    test_mask = np.zeros((num_user, num_item), dtype=np.float32)
    confounder_matrix = confounder_matrix.astype(np.float32)

    # Populate train matrix and mask
    for _, row in train_df.iterrows():
        user, item, rating = int(row['userId']), int(row['songId']), row['rating']
        train_matrix[user, item] = rating
        train_mask[user, item] = 1.0

    # Populate test matrix and mask
    for _, row in test_df.iterrows():
        user, item, rating = int(row['userId']), int(row['songId']), row['rating']
        test_matrix[user, item] = rating
        test_mask[user, item] = 1.0

    # Check consistency
    mismatch_count = 0
    for user in range(num_user):
        for item in range(num_item):
            if train_mask[user, item] > 0 and confounder_matrix.shape[1] > item:
                if train_matrix[user, item] != 0 and confounder_matrix[user, item] == 0:
                    mismatch_count += 1
                    print(f"Mismatch in confounder_matrix at user: {user}, item: {item}")
            if test_mask[user, item] > 0 and confounder_matrix.shape[1] > item:
                if test_matrix[user, item] != 0 and confounder_matrix[user, item] == 0:
                    mismatch_count += 1
                    print(f"Mismatch in confounder_matrix at user: {user}, item: {item}")

    print(f"Total mismatches found: {mismatch_count}")

    return train_matrix, test_matrix, train_mask, test_mask, confounder_matrix

class IAutoRec(tf.keras.Model):
    def __init__(self, num_user, num_item, hidden_neuron=500, learning_rate=0.001, reg_rate=0.1, epoch=100, batch_size=500, verbose=1, T=3, display_step=1000):
        super(IAutoRec, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.hidden_neuron = hidden_neuron
        self.learning_rate = learning_rate
        self.reg_rate = reg_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.verbose = verbose
        self.T = T
        self.display_step = display_step

        self.V = tf.Variable(tf.random.normal([hidden_neuron, num_item], stddev=0.01, dtype=tf.float32))
        self.W = tf.Variable(tf.random.normal([num_item, hidden_neuron], stddev=0.01, dtype=tf.float32))
        self.mu = tf.Variable(tf.random.normal([hidden_neuron, 1], stddev=0.01, dtype=tf.float32))  # Changed shape to [hidden_neuron, 1]
        self.b = tf.Variable(tf.random.normal([num_item, 1], stddev=0.01, dtype=tf.float32))  # Changed shape to [num_item, 1]

        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)

    def call(self, inputs, training=False):
        rating_matrix, rating_matrix_mask = inputs
        layer_1 = tf.nn.dropout(tf.sigmoid(self.mu + tf.matmul(self.V, tf.transpose(rating_matrix))), 0.95 if training else 0.99)
        layer_2 = tf.matmul(self.W, layer_1) + self.b  # This should broadcast correctly now
        return tf.transpose(layer_2)  # Transpose to match the original rating matrix shape

    def loss_fn(self, rating_matrix, rating_matrix_mask):
        reconstruction = self.call((rating_matrix, rating_matrix_mask), training=True)
        loss = tf.reduce_mean(tf.square(tf.multiply((rating_matrix - reconstruction), rating_matrix_mask))) + self.reg_rate * (tf.square(tf.norm(self.W)) + tf.square(tf.norm(self.V)))
        return loss

    def train_step(self, rating_matrix, rating_matrix_mask):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(rating_matrix, rating_matrix_mask)
        gradients = tape.gradient(loss, [self.V, self.W, self.mu, self.b])
        self.optimizer.apply_gradients(zip(gradients, [self.V, self.W, self.mu, self.b]))
        return loss

    def train(self, train_data, train_mask):
        num_training = train_data.shape[0]
        total_batch = int(num_training / self.batch_size)
        idxs = np.random.permutation(num_training)

        for i in range(total_batch):
            start_time = time.time()
            if i == total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size:]
            else:
                batch_set_idx = idxs[i * self.batch_size: (i + 1) * self.batch_size]

            batch_data = train_data[batch_set_idx]
            batch_mask = train_mask[batch_set_idx]

            loss = self.train_step(batch_data, batch_mask)

            if i % self.display_step == 0 and self.verbose:
                print(f"Index: {i + 1:04d}; cost= {loss:.9f}")
                print(f"one iteration: {time.time() - start_time:.2f} seconds.")

    def test(self, test_data, test_mask):
        reconstruction = self.call((test_data, test_mask), training=False)
        error = 0
        error_mae = 0
        num_test_points = np.sum(test_mask)
        
        # Iterate over the non-zero entries in the test_mask
        for u in range(test_data.shape[0]):
            for i in range(test_data.shape[1]):
                if test_mask[u, i] > 0:
                    pred_rating_test = reconstruction[u, i]
                    error += (test_data[u, i] - pred_rating_test) ** 2
                    error_mae += np.abs(test_data[u, i] - pred_rating_test)
        
        print("RMSE:" + str(np.sqrt(error / num_test_points)) + "; MAE:" + str(error_mae / num_test_points))


    def execute(self, train_data, train_mask, test_data, test_mask):
        for epoch in range(self.epochs):
            if self.verbose:
                print(f"Epoch: {epoch:04d};")
            self.train(train_data, train_mask)
            if epoch % self.T == 0:
                print(f"Epoch: {epoch:04d}; ", end='')
                self.test(test_data, test_mask)

    def save(self, path):
        self.save_weights(path)

    def predict(self, user_id, item_id):
        reconstruction = self.call((self.train_data, self.train_data_conf), training=False)
        return reconstruction[user_id, item_id]

    def _data_process(self, data):
        output = np.zeros((self.num_user, self.num_item))
        for u in range(self.num_user):
            for i in range(self.num_item):
                output[u, i] = data.get((u, i), 0)
        return output


class IAutoRecWithConfounders(IAutoRec):
    def __init__(self, num_user, num_item, hidden_neuron=500, learning_rate=0.001, reg_rate=0.1, epoch=100, batch_size=500, verbose=1, T=3, display_step=1000):
        super(IAutoRecWithConfounders, self).__init__(num_user, num_item, hidden_neuron, learning_rate, reg_rate, epoch, batch_size, verbose, T, display_step)

    def call(self, inputs, training=False):
        rating_matrix, rating_matrix_mask, confounder_matrix = inputs

        # Debug print statements to check shapes
        # print(f"rating_matrix shape: {rating_matrix.shape}")
        # print(f"confounder_matrix shape: {confounder_matrix.shape}")

        # Ensure the shapes are aligned before adding
        if rating_matrix.shape != confounder_matrix.shape:
            raise ValueError(f"Shape mismatch: rating_matrix shape {rating_matrix.shape} and confounder_matrix shape {confounder_matrix.shape} must be the same.")
        
        layer_1 = tf.nn.dropout(tf.sigmoid(self.mu + tf.matmul(self.V, tf.transpose(rating_matrix + confounder_matrix))), 0.95 if training else 0.99)
        layer_2 = tf.matmul(self.W, layer_1) + self.b
        return tf.transpose(layer_2)  # Transpose to match the original rating matrix shape




    def loss_fn(self, rating_matrix, rating_matrix_mask, confounder_matrix):
        reconstruction = self.call((rating_matrix, rating_matrix_mask, confounder_matrix), training=True)
        loss = tf.reduce_mean(tf.square(tf.multiply((rating_matrix - reconstruction), rating_matrix_mask))) + self.reg_rate * (tf.square(tf.norm(self.W)) + tf.square(tf.norm(self.V)))
        return loss

    def train_step(self, rating_matrix, rating_matrix_mask, confounder_matrix):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(rating_matrix, rating_matrix_mask, confounder_matrix)
        gradients = tape.gradient(loss, [self.V, self.W, self.mu, self.b])
        self.optimizer.apply_gradients(zip(gradients, [self.V, self.W, self.mu, self.b]))
        return loss

    def train(self, train_data, train_mask, confounder_data):
        num_training = train_data.shape[0]
        total_batch = int(num_training / self.batch_size)
        idxs = np.random.permutation(num_training)

        for i in range(total_batch):
            start_time = time.time()
            if i == total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size:]
            else:
                batch_set_idx = idxs[i * self.batch_size: (i + 1) * self.batch_size]

            batch_data = train_data[batch_set_idx]
            batch_mask = train_mask[batch_set_idx]
            batch_confounder = confounder_data[batch_set_idx]

            loss = self.train_step(batch_data, batch_mask, batch_confounder)

            if i % self.display_step == 0 and self.verbose:
                print(f"Index: {i + 1:04d}; cost= {loss:.9f}")
                print(f"one iteration: {time.time() - start_time:.2f} seconds.")

    def test(self, test_data, test_mask, test_confounder):
        reconstruction = self.call((test_data, test_mask, test_confounder), training=False)
        error = 0
        error_mae = 0
        num_test_points = np.sum(test_mask)
        
        # Iterate over the non-zero entries in the test_mask
        for u in range(test_data.shape[0]):
            for i in range(test_data.shape[1]):
                if test_mask[u, i] > 0:
                    pred_rating_test = reconstruction[u, i]
                    error += (test_data[u, i] - pred_rating_test) ** 2
                    error_mae += np.abs(test_data[u, i] - pred_rating_test)
        
        print("RMSE:" + str(np.sqrt(error / num_test_points)) + "; MAE:" + str(error_mae / num_test_points))

    def execute(self, train_data, train_mask, test_data, test_mask, confounder_data):
        for epoch in range(self.epochs):
            if self.verbose:
                print(f"Epoch: {epoch:04d};")
            self.train(train_data, train_mask, confounder_data)
            if epoch % self.T == 0:
                print(f"Epoch: {epoch:04d}; ", end='')
                self.test(test_data, test_mask, confounder_data)


# Prepare data

if __name__ == '__main__':
    num_user = 6040
    num_item = 3706

    CAUSEFIT_DIR = 'C:/Users/Sten Stokroos/Desktop/Thesis2.0/zelf/dat/out/ml_wg'
    DATA_DIR = 'C:/Users/Sten Stokroos/Desktop/zelf/neural_collaborative_filtering/Data'
    dim = 30

    train_file = os.path.join(DATA_DIR, 'ml-1m.train.rating')
    test_file = os.path.join(DATA_DIR, 'ml-1m.test.rating')
    confounder_file = os.path.join(CAUSEFIT_DIR, 'cause_pmf_k' + str(dim))

    train_data, test_data, train_mask, test_mask, confounder_data = prepare_data(num_user, num_item, train_file, test_file, confounder_file)

    # IAutoRec with Confounders
    confounder_model = IAutoRecWithConfounders(num_user, num_item)
    print("Training IAutoRec with Confounders...")
    confounder_model.execute(train_data, train_mask, test_data, test_mask, confounder_data)

    # Original IAutoRec
    original_model = IAutoRec(num_user, num_item)
    print("Training Original IAutoRec...")
    original_model.execute(train_data, train_mask, test_data, test_mask)


