import scipy.sparse as sp
import numpy as np
import pandas as pd

class Dataset(object):
    def __init__(self, train_df, test_df, neg_path, num_users, num_items):
        self.num_users = num_users
        self.num_items = num_items
        self.trainMatrix = self.load_rating_df_as_matrix(train_df)
        self.testRatings = self.load_rating_df_as_list(test_df)
        self.testNegatives = self.load_negative_file(neg_path)
        print(f'Number of test ratings: {len(self.testRatings)}')
        print(f'Number of test negatives: {len(self.testNegatives)}')
        assert len(self.testRatings) == len(self.testNegatives)
        
    def load_rating_df_as_list(self, df):
        ratingList = []
        for _, row in df.iterrows():
            ratingList.append([int(row['userId']), int(row['songId'])])
        return ratingList
    
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split("\t")[1:]
                negatives = [int(x) for x in parts]
                negativeList.append(negatives)
        return negativeList
    
    def load_rating_df_as_matrix(self, df):
        mat = sp.dok_matrix((self.num_users + 1, self.num_items + 1), dtype=np.float32)
        for _, row in df.iterrows():
            user, item, rating = int(row['userId']), int(row['songId']), float(row['rating'])
            if rating > 0:
                mat[user, item] = 1.0
        return mat
