import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time


@cython.boundscheck(False)
@cython.wraparound(False)
def get_train_instances(np.ndarray train, np.ndarray confounders, int num_negatives):
    cdef int num_users = train.shape[0]
    cdef int num_items = train.shape[1]
    cdef int max_item_index = confounders.shape[1] - 1

    user_input = []
    item_input = []
    confounder_input = []
    labels = []

    for u in range(num_users):
        for i in range(num_items):
            if train[u, i] > 0:
                if i >= confounders.shape[1]:
                    i = max_item_index
                user_input.append(u)
                item_input.append(i)
                confounder_input.append(confounders[u, i])
                labels.append(1)

                for _ in range(num_negatives):
                    j = rand() % num_items
                    while train[u, j] > 0 or j >= confounders.shape[1]:
                        j = rand() % num_items
                    user_input.append(u)
                    item_input.append(j)
                    confounder_input.append(confounders[u, j])
                    labels.append(0)

    return np.array(user_input), np.array(item_input), np.array(confounder_input), np.array(labels)

@cython.boundscheck(False)
@cython.wraparound(False)
def eval_one_rating(np.ndarray testRatings, np.ndarray testNegatives, np.ndarray confounders, int idx, model, int K):
    cdef int u = testRatings[idx, 0]
    cdef int gtItem = testRatings[idx, 1]
    cdef int num_items = confounders.shape[1]
    cdef int max_item_index = num_items - 1

    items = testNegatives[idx]
    items.append(gtItem)

    valid_items = [item if item < num_items else max_item_index for item in items]
    gtItem = gtItem if gtItem < num_items else max_item_index

    map_item_score = {}
    users = np.full(len(valid_items), u, dtype=np.int32)
    confounder_values = [confounders[u, item] for item in valid_items]

    predictions = model.predict([users, np.array(valid_items), np.array(confounder_values)], batch_size=100, verbose=0)
    for i in range(len(valid_items)):
        item = valid_items[i]
        map_item_score[item] = predictions[i]

    ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return hr, ndcg
