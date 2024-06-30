import tensorflow as tf
import time
import numpy as np
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
        print("UAutoRec with Confounder and Exposure.")

    def build_network(self, hidden_neuron=500):
        self.rating_matrix = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.num_item, None])
        self.rating_matrix_mask = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.num_item, None])
        self.confounder_matrix = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.num_item, None])
        self.exposure_matrix = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.num_item, None])

        # Rating path
        V_R = tf.Variable(tf.random.normal([hidden_neuron, self.num_item], stddev=0.01))
        mu_R = tf.Variable(tf.random.normal([hidden_neuron], stddev=0.01))
        layer_1_R = tf.sigmoid(tf.expand_dims(mu_R, 1) + tf.matmul(V_R, self.rating_matrix))
        
        # Confounder path
        V_C = tf.Variable(tf.random.normal([hidden_neuron, self.num_item], stddev=0.01))
        mu_C = tf.Variable(tf.random.normal([hidden_neuron], stddev=0.01))
        layer_1_C = tf.sigmoid(tf.expand_dims(mu_C, 1) + tf.matmul(V_C, self.confounder_matrix))
        
        # Exposure path
        V_E = tf.Variable(tf.random.normal([hidden_neuron, self.num_item], stddev=0.01))
        mu_E = tf.Variable(tf.random.normal([hidden_neuron], stddev=0.01))
        layer_1_E = tf.sigmoid(tf.expand_dims(mu_E, 1) + tf.matmul(V_E, self.exposure_matrix))
        
        # Combine paths
        layer_1 = layer_1_R + layer_1_C + layer_1_E
        
        # Output layer
        W = tf.Variable(tf.random.normal([self.num_item, hidden_neuron], stddev=0.01))
        b = tf.Variable(tf.random.normal([self.num_item], stddev=0.01))
        self.layer_2 = tf.matmul(W, layer_1) + tf.expand_dims(b, 1)
        self.loss = tf.reduce_mean(tf.square(
            tf.norm(tf.multiply((self.rating_matrix - self.layer_2), self.rating_matrix_mask)))) + self.reg_rate * (
        tf.square(tf.norm(W)) + tf.square(tf.norm(V_R)) + tf.square(tf.norm(V_C)) + tf.square(tf.norm(V_E)))
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

            try:
                _, loss = self.sess.run([self.optimizer, self.loss],
                                        feed_dict={self.rating_matrix: self.train_data[:, batch_set_idx],
                                                   self.rating_matrix_mask: self.train_data_mask[:, batch_set_idx],
                                                   self.confounder_matrix: confounder_data[:, batch_set_idx],
                                                   self.exposure_matrix: exposure_data[:, batch_set_idx]})
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
        return rmse, mae

    def execute(self, train_data, test_data, confounder_data, exposure_data):
        self.train_data = self._data_process(train_data.transpose())
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

def load_data_rating(df, file1, file2, columns=[0, 1, 2], test_size=0.1, sep="\t"):
    print(df.head())

    n_users = df['userId'].unique().shape[0]
    n_items = df['itemId'].unique().shape[0]

    print('Number of users:', n_users)
    print('Number of items:', n_items)
    # train_data, test_data = train_test_split(df, test_size=test_size, random_state= 0)
    train_data = pd.read_csv(file1, sep=sep, header=None, names=['userId', 'itemId', 'rating'], usecols=columns, engine="python")
    test_data = pd.read_csv(file2, sep=sep, header=None, names=['userId', 'itemId', 'rating'], usecols=columns, engine="python")

    print(train_data.shape)
    print(test_data.shape)


    train_row = []
    train_col = []
    train_rating = []

    max_user_id = df['userId'].max()
    max_item_id = df['itemId'].max()

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
    return train_matrix.todok(), test_matrix.todok(), max_user_id + 1, max_item_id + 1  # Adjust dimensions by adding 1

def load_and_combine_data(file1, file2, columns=[0, 1, 2], sep="\t"):
    # Load the first file
    df1 = pd.read_csv(file1, sep=sep, header=None, names=['userId', 'itemId', 'rating'], usecols=columns, engine="python")
    
    # Load the second file
    df2 = pd.read_csv(file2, sep=sep, header=None, names=['userId', 'itemId', 'rating'], usecols=columns, engine="python")

    # Sort the dataframes by userId
    df1 = df1.sort_values(by=['userId'])
    df2 = df2.sort_values(by=['userId'])
    
    # Initialize an empty list to hold the combined rows
    combined_rows = []

    # Iterate over each user in the first dataframe
    for user in df1['userId'].unique():
        # Get all rows for the current user in the first dataframe
        user_df1 = df1[df1['userId'] == user]
        combined_rows.append(user_df1)
        
        # Get all rows for the current user in the second dataframe
        user_df2 = df2[df2['userId'] == user]
        if not user_df2.empty:
            combined_rows.append(user_df2)
    
    # Concatenate all the collected rows into a single dataframe
    combined_df = pd.concat(combined_rows, ignore_index=True)
    
    return combined_df

# Example usage
file1 = 'C:/Users/Sten Stokroos/Desktop/zelf/neural_collaborative_filtering/Data/ml-1m.train.rating'  # Replace with the actual file path
file2 = 'C:/Users/Sten Stokroos/Desktop/zelf/neural_collaborative_filtering/Data/ml-1m.test.rating' # Replace with the actual file path
combined_df = load_and_combine_data(file1, file2, columns=[0, 1, 2], sep="\t")

combined_df.to_csv('C:/Users/Sten Stokroos/Desktop/Thesis2.0/zelf/neural_collaborative_filtering/Data/ml-1m.csv', index = False)

train, test, user, item = load_data_rating(combined_df, file1, file2, columns=[0, 1, 2], test_size=0.2, sep="\t")



CAUSEFIT_DIR = 'C:/Users/Sten Stokroos/Desktop/Thesis2.0/zelf/xposure_matrix.csv'
    
dim = 30 
# U = np.loadtxt(CAUSEFIT_DIR)# + '/cause_pmf_k'+str(dim)+'_U.csv')
# B = np.loadtxt(CAUSEFIT_DIR)# + '/cause_pmf_k'+str(dim)+'_V.csv')
# U = np.atleast_2d(U.T).T
# B = np.atleast_2d(B.T).T
# confounder_data = (U.dot(B.T)).T

conf_df = pd.read_csv(CAUSEFIT_DIR, header=None)

# Convert the DataFrame to a NumPy array
confounder_data = conf_df.to_numpy()
confounder_data = confounder_data.T

# #MLP OPtion
# df = pd.read_csv('C:/Users/Sten Stokroos/Desktop/Thesis2.0/zelf/neural_collaborative_filtering/Data/predicted_scores_copy.csv')
# confounder_data = df.to_numpy()
# # Transpose the array if needed
# confounder_data = confounder_data.T



# Create the exposure data matrix
exposure_data = (train > 0).astype(np.float32).todense().T

# Set TensorFlow session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

with tf.compat.v1.Session(config=config) as sess:
    model = UAutoRec(sess, user, item, learning_rate=0.001, reg_rate=0.1, epoch=20, batch_size=500, verbose=True)
    model.build_network()
    model.execute(train, test, confounder_data, exposure_data)



# 20/20 [01:15<00:00,  3.79s/epoch, Loss=5.98e+4, RMSE=0.91, MAE=0.721] YES BABYYYYY MLP

# 3.16s/epoch, Loss=6.76e+4, RMSE=0.976, MAE=0.778] HPF and test.rating everywehre


