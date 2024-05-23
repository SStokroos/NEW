'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)

@author: hexiangnan
'''
import math
import heapq # for retrieval topK
import multiprocessing
import numpy as np
from time import time
from tqdm import tqdm
#from numba import jit, autojit

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None

def evaluate_model(model, testRatings, testNegatives, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K

    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K


        
    hits, ndcgs = [],[]
    if(num_thread > 1): # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    # Single thread
    for idx in tqdm(range(len(_testRatings))):
        (hr,ndcg) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)      
    return (hits, ndcgs)


# def evaluate_model(model, testRatings, testNegatives, K, num_thread, substitute_values):
#     global _model
#     global _testRatings
#     global _testNegatives
#     global _K
#     global _substitute_values
#     _model = model
#     _testRatings = testRatings
#     _testNegatives = testNegatives
#     _K = K
#     _substitute_values = substitute_values  # New Addition
        
#     hits, ndcgs = [],[]
    
#     # Validate and filter data here if necessary
#     valid_indices = [(idx, rating) for idx, rating in enumerate(_testRatings) if rating[0] < substitute_values.shape[0] and all(item < substitute_values.shape[1] for item in _testNegatives[idx] + [rating[1]])]

    

#     if(num_thread > 1):  # Multi-thread
#         pool = multiprocessing.Pool(processes=num_thread)
#         res = pool.map(eval_one_rating, [idx for idx, _ in valid_indices])
#         pool.close()
#         pool.join()
#     else:  # Single thread
#         for idx, _ in tqdm(valid_indices):
#             (hr, ndcg) = eval_one_rating(idx)
#             hits.append(hr)
#             ndcgs.append(ndcg)      
    
#     return (hits, ndcgs)

# def eval_one_rating(idx):
#     rating = _testRatings[idx]
#     items = _testNegatives[idx].copy()
#     u = rating[0]
#     gtItem = rating[1]
#     items.append(gtItem)
#     users = np.full(len(items), u, dtype='int32')

#     confounders = _substitute_values[u, items]  # This is now safe to access
#     predictions = _model.predict([users, np.array(items), confounders], batch_size=100, verbose=0)
#     map_item_score = {item: predictions[i][0] for i, item in enumerate(items)}

#     items.pop()  # Remove the ground truth item added earlier
    
#     ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
#     hr = getHitRatio(ranklist, gtItem)
#     ndcg = getNDCG(ranklist, gtItem)
#     return (hr, ndcg)



def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype = 'int32')
    predictions = _model.predict([users, np.array(items)], 
                                 batch_size=100, verbose=0)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()
    
    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
