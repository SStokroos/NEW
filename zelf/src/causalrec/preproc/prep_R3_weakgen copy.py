
# coding: utf-8

# In[1]:


# get_ipython().magic(u'matplotlib inline')
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy.random as npr
from sklearn.cluster import KMeans
from scipy.stats import invgamma

from scipy import sparse, stats
from sklearn.model_selection import train_test_split
# plt.style.use('ggplot')


# In[2]:


# import seaborn as sns
# sns.set_style("white")
# sns.set_context("paper")

# color_names = ["red",
#                "windows blue",
#                "medium green",
#                "dusty purple",
#                "orange",
#                "amber",
#                "clay",
#                "pink",
#                "greyish",
#                "light cyan",
#                "steel blue",
#                "forest green",
#                "pastel purple",
#                "mint",
#                "salmon",
#                "dark brown"]
# colors = sns.xkcd_palette(color_names)


# In[3]:


# DATA_DIR = 'C:/Users/Sten Stokroos/Desktop/zelf/neural_collaborative_filtering/Data'
DATA_DIR = 'C:/Users/Sten Stokroos/Desktop/zelf/neural_collaborative_filtering/Data'
OUT_DATA_DIR = 'C:/Users/Sten Stokroos/Desktop/Thesis2.0/zelf/dat/proc/ml_wg2'

# Now you can use these paths to read your data
# tr_vd_data = pd.read_csv(os.path.join(DATA_DIR, 'ydata-ymusic-rating-study-v1_0-train.txt'), sep="\t", header=None, names=['userId', 'songId', 'rating'], engine="python")
# test_data = pd.read_csv(os.path.join(DATA_DIR, 'ydata-ymusic-rating-study-v1_0-test.txt'), sep="\t", header=None, names=['userId', 'songId', 'rating'], engine="python")

def load_and_combine_data(df1, df2):
    # Load the first file
    # df1 = pd.read_csv(file1, sep=sep, header=None, names=['userId', 'itemId', 'rating'], usecols=columns, engine="python")
    
    # Load the second file
    # df2 = pd.read_csv(file2, sep=sep, header=None, names=['userId', 'itemId', 'rating'], usecols=columns, engine="python")

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

def load_data_rating(df, columns=[0, 1, 2], test_size=0.1, sep="\t"):

    train_data, test_data = train_test_split(df, test_size=test_size, random_state= 0)

    return train_data, test_data  


file1 = pd.read_csv(os.path.join(DATA_DIR, 'ml-1m.train.rating'), sep="\t", header=None, names=['userId', 'songId', 'rating'], usecols=[0, 1, 2], engine="python")
file2 = pd.read_csv(os.path.join(DATA_DIR, 'ml-1m.test.rating'), sep="\t", header=None, names=['userId', 'songId', 'rating'], usecols=[0, 1, 2], engine="python")

df = load_and_combine_data(file1, file2)

tr_vd_data, test_data = load_data_rating(df)
# In[6]:


tr_vd_data.head(), tr_vd_data.shape


# In[7]:


test_data.head(), test_data.shape

# In[8]:
#2037	1805	3	974668726

def split_train_test_proportion(data, uid, test_prop=0.5, random_seed=0):
    data_grouped_by_user = data.groupby(uid)
    tr_list, te_list = list(), list()

    np.random.seed(random_seed)

    for u, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)

        if u % 5000 == 0:
            print("%d users sampled" % u)
            sys.stdout.flush()

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)
    
    return data_tr, data_te


# In[9]:


def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


# In[10]:


user_activity = get_count(tr_vd_data, 'userId')
item_popularity = get_count(tr_vd_data, 'songId')
print(item_popularity)
print(user_activity)
# In[11]:


unique_uid = user_activity.index
unique_sid = item_popularity.index

print(unique_uid)

# In[12]:


n_users = len(unique_uid)
n_items = len(unique_sid)
print(n_users)
print(n_items)

# In[13]:


n_users, n_items


# In[14]:
# After you have unique_uid and unique_sid populated
user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))
song2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))

user2id



# In[15]:


# for the test set, only keep the users/items from the training set

test_data = test_data.loc[test_data['userId'].isin(unique_uid)]
test_data = test_data.loc[test_data['songId'].isin(unique_sid)]
print(test_data)

# In[16]:
if not os.path.exists(OUT_DATA_DIR):
    os.makedirs(OUT_DATA_DIR)

with open(os.path.join(OUT_DATA_DIR, 'unique_uid.txt'), 'w') as f:
    for uid in unique_uid:
        f.write('%s\n' % uid)

with open(os.path.join(OUT_DATA_DIR, 'unique_sid.txt'), 'w') as f:
    for sid in unique_sid:
        f.write('%s\n' % sid)


# # Turn userId and songId to 0-based index

# In[17]:


# def numerize(tp):
#     uid = list(map(lambda x: user2id[x], tp['userId']))
#     sid = list(map(lambda x: song2id[x], tp['songId']))
#     tp.loc[:, 'uid'] = uid
#     tp.loc[:, 'sid'] = sid
#     return tp[['uid', 'sid', 'rating']]


def numerize(tp):
    # Adjust for 1-based userID and songID
    tp['uid'] = tp['userId'].apply(lambda x: user2id.get(x))  # Subtract 1 for 1-based userId
    tp['sid'] = tp['songId'].apply(lambda x: song2id.get(x))  # Subtract 1 for 1-based songId
    
    # Drop any rows that have NaN after the operation (i.e., were not found in the dictionary)
    # tp = tp.dropna(subset=['userId', 'songId'])
    
    return tp[['uid', 'sid', 'rating']]


# In[18]:


tr_vd_data = numerize(tr_vd_data)
test_data = numerize(test_data)

# Ensure data types are integers
# tr_vd_data['uid'] = tr_vd_data['uid'].astype('int')
# tr_vd_data['sid'] = tr_vd_data['sid'].astype('int')

# Example of checking for NaNs in each column
print(tr_vd_data['uid'].isnull().sum(), "NaNs in 'uid'")
print(tr_vd_data['sid'].isnull().sum(), "NaNs in 'sid'")
print(tr_vd_data['rating'].isnull().sum(), "NaNs in 'rating'")


# tr_vd_data = tr_vd_data.dropna(subset=['sid'])

print(tr_vd_data)

# In[19]:


train_data, vad_data = split_train_test_proportion(tr_vd_data, 'uid', test_prop=0.6, random_seed=12345)
obs_test_data, vad_data = split_train_test_proportion(vad_data, 'uid', test_prop=0.5, random_seed=12345)


# In[20]:


print("There are total of %d unique users in the training set and %d unique users in the entire dataset" % (len(pd.unique(train_data['uid'])), len(unique_uid)))


# In[21]:


print("There are total of %d unique items in the training set and %d unique items in the entire dataset" % (len(pd.unique(train_data['sid'])), len(unique_sid)))


# In[22]:
# def move_to_fill(part_data_1, part_data_2, unique_id, key):
#     # move the data from part_data_2 to part_data_1 so that part_data_1 has the same number of unique "key" as unique_id
#     part_id = set(pd.unique(part_data_1[key]))
    
#     left_id = list()
#     for i, _id in enumerate(unique_id):
#         if _id not in part_id:
#             left_id.append(_id)
            
#     move_idx = part_data_2[key].isin(left_id)
#     part_data_1 = pd.concat([part_data_1, part_data_2[move_idx]])
#     part_data_2 = part_data_2[~move_idx]
#     return part_data_1, part_data_2

def move_to_fill(part_data_1, part_data_2, unique_id, key):
    # move the data from part_data_2 to part_data_1 so that part_data_1 has the same number of unique "key" as unique_id
    part_id = set(pd.unique(part_data_1[key]))
    
    left_id = list()
    for i, _id in enumerate(unique_id):
        if _id not in part_id:
            left_id.append(_id)
            
    move_idx = part_data_2[key].isin(left_id)
    
    # Here we concatenate along the rows, which is the same as using .append()
    part_data_1 = pd.concat([part_data_1, part_data_2.loc[move_idx]], ignore_index=True)
    part_data_2 = part_data_2.loc[~move_idx]
    
    return part_data_1, part_data_2



# def move_to_fill(part_data_1, part_data_2, unique_id, key):
#     # move the data from part_data_2 to part_data_1 so that part_data_1 has the same number of unique "key" as unique_id
#     part_id = set(pd.unique(part_data_1[key]))
    
#     left_id = list()
#     for i, _id in enumerate(unique_id):
#         if _id not in part_id:
#             left_id.append(_id)
            
#     move_idx = part_data_2[key].isin(left_id)
#     print(type(part_data_1))

#     part_data_1 = part_data_1.append(part_data_2[move_idx])
#     part_data_2 = part_data_2[~move_idx]
#     return part_data_1, part_data_2


# In[23]:


train_data, vad_data = move_to_fill(train_data, vad_data, np.arange(n_items), 'sid')
train_data, obs_test_data = move_to_fill(train_data, obs_test_data, np.arange(n_items), 'sid')


# In[24]:




print("There are total of %d unique items in the training set and %d unique items in the entire dataset" % (len(pd.unique(train_data['sid'])), len(unique_sid)))


# In[25]:


train_data.to_csv(os.path.join(OUT_DATA_DIR, 'train.csv'), index=False)
vad_data.to_csv(os.path.join(OUT_DATA_DIR, 'validation.csv'), index=False)
tr_vd_data.to_csv(os.path.join(OUT_DATA_DIR, 'train_full.csv'), index=False)


# In[26]:


obs_test_data.to_csv(os.path.join(OUT_DATA_DIR, 'obs_test_full.csv'), index=False)
test_data.to_csv(os.path.join(OUT_DATA_DIR, 'test_full.csv'), index=False)

print('HERE ARE SIZES')
print(train_data.shape)
print(vad_data.shape)
print(obs_test_data.shape)
print(test_data.shape)

# # Load the data

# In[27]:


unique_uid = list()
with open(os.path.join(OUT_DATA_DIR, 'unique_uid.txt'), 'r') as f:
    for line in f:
        unique_uid.append(line.strip())
    
unique_sid = list()
with open(os.path.join(OUT_DATA_DIR, 'unique_sid.txt'), 'r') as f:
    for line in f:
        unique_sid.append(line.strip())


# In[28]:


n_items = len(unique_sid)
n_users = len(unique_uid)

print(n_users, n_items)


# In[29]:


def load_data(csv_file, shape=(n_users, n_items)):
    tp = pd.read_csv(csv_file)
    rows, cols, vals = np.array(tp['uid']), np.array(tp['sid']), np.array(tp['rating']) 
    data = sparse.csr_matrix((vals, (rows, cols)), dtype=np.float32, shape=shape)
    return data


# In[30]:


def binarize_rating(data, cutoff=3, eps=1e-6):
    data.data[data.data < cutoff] = eps   # small value so that it will not be treated as 0 in sparse matrix 
    data.data[data.data >= cutoff] = 1
    return data


# In[31]:


def exp_to_imp(data, cutoff=0.5):
    # turn data (explicit feedback) to implict with cutoff
    data_imp = data.copy()
    data_imp.data[data_imp.data < cutoff] = 0
    data_imp.data[data_imp.data >= cutoff] = 1
    data_imp.data = data_imp.data.astype('int32')
    data_imp.eliminate_zeros()
    return data_imp


# In[32]:


def binarize_spmat(spmat):
    spmat_binary = spmat.copy()
    spmat_binary.data = np.ones_like(spmat_binary.data)
    return spmat_binary


# In[33]:


def subsample_negatives(data, full_data=None, random_state=0, verbose=False):
    # roughly subsample the same number of negative as the positive in `data` for each user
    # `full_data` is all the positives we *are supposed to* know
    n_users, n_items = data.shape
    
    if full_data is None:
        full_data = data

    rows_neg, cols_neg = [], []

    np.random.seed(random_state)

    for u in range(n_users):
        p = np.ones(n_items, dtype='float32')
        p[full_data[u].nonzero()[1]] = 0
        p /= p.sum()

        neg_items = np.random.choice(n_items, size=data[u].nnz, replace=False, p=p)

        rows_neg.append([u] * data[u].nnz)
        cols_neg.append(neg_items)

        if verbose and u % 5000 == 0:
            print("%d users sampled" % u)
            sys.stdout.flush()

    rows_neg = np.hstack(rows_neg)
    cols_neg = np.hstack(cols_neg)

    return rows_neg, cols_neg


# In[34]:


train_data = load_data(os.path.join(OUT_DATA_DIR, 'train_full.csv'))


# In[35]:


# bins = np.histogram(train_data.data, bins=5)[0]
# plt.bar(np.arange(1, 6), bins)
# pass


# In[36]:


test_data = load_data(os.path.join(OUT_DATA_DIR, 'test_full.csv'))
vad_data = load_data(os.path.join(OUT_DATA_DIR, 'validation.csv'))


# In[37]:
# import matplotlib as plt

# bins = np.histogram(test_data.data, bins=5)[0]
# plt.bar(np.arange(1, 6), bins)
# pass


# In[38]:


# bins = np.histogram(vad_data.data, bins=5)[0]
# plt.bar(np.arange(1, 6), bins)
# pass




