'''
Created on Aug 9, 2016
Keras Implementation of Multi-Layer Perceptron (GMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.  

@author: Xiangnan He (xiangnanhe@gmail.com)
'''

import numpy as np
from time import time
import os
from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, Adagrad, RMSprop, SGD
from tensorflow.keras.initializers import RandomNormal
import numpy as np

# from cython_helpers import get_train_instances, eval_one_rating

from Dataset import Dataset
from evaluate import evaluate_model
import argparse

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='C:/Users/Sten Stokroos/Desktop/Thesis2.0/zelf/neural_collaborative_filtering/Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each layer")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()

def get_model(num_users, num_items, layers, reg_layers):
    assert len(layers) == len(reg_layers), "Each layer should have a corresponding regularization parameter"

    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')
    confounder_input = Input(shape=(1,), dtype='float32', name='confounder_input')

    embedding_initializer = RandomNormal(mean=0.0, stddev=0.01)
    user_embedding = Embedding(input_dim=num_users, output_dim=layers[0] // 2,
                               embeddings_initializer=embedding_initializer,
                               embeddings_regularizer=l2(reg_layers[0]),
                               name='user_embedding')(user_input)
    item_embedding = Embedding(input_dim=num_items, output_dim=layers[0] // 2,
                               embeddings_initializer=embedding_initializer,
                               embeddings_regularizer=l2(reg_layers[0]),
                               name='item_embedding')(item_input)

    user_latent = Flatten()(user_embedding)
    item_latent = Flatten()(item_embedding)


    # TRY WITH INTERACITN LAYER FIRST, End. Best Iteration 0:  HR = 0.5745, NDCG = 0.3254.
    # interaction_layer = Dense(16, activation='relu')(Concatenate()([user_latent, item_latent, confounder_input]))  
    # vector = Dense(layers[1], activation='relu')(interaction_layer)


    vector = Concatenate()([user_latent, item_latent, confounder_input])

    for idx, layer_size in enumerate(layers[1:], start=1):
        vector = Dense(layer_size, activation='relu',
                       kernel_regularizer=l2(reg_layers[idx]),
                       name=f'layer{idx}')(vector)

    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='prediction')(vector)

    model = Model(inputs=[user_input, item_input, confounder_input], outputs=prediction)
    return model



# def get_train_instances(train, confounders, num_negatives):
#     user_input, item_input, confounder_input, labels = [], [], [], []
    
#     num_users = max(key[0] for key in train.keys()) + 1
#     num_items = max(key[1] for key in train.keys()) + 1
#     max_item_index = confounders.shape[1] - 1

#     for (u, i) in train.keys():
#         if i >= confounders.shape[1]:
#             i = max_item_index  # Clip item index
#         user_input.append(u)
#         item_input.append(i)
#         confounder_input.append(confounders[u, i])
#         labels.append(1)

#         for _ in range(num_negatives):
#             j = np.random.randint(num_items)
#             while (u, j) in train or j >= confounders.shape[1]:
#                 j = np.random.randint(num_items)
#             user_input.append(u)
#             item_input.append(j)
#             confounder_input.append(confounders[u, j])
#             labels.append(0)

#     return np.array(user_input), np.array(item_input), np.array(confounder_input), np.array(labels)

def get_train_instances(train, confounders, num_negatives):
    user_input, item_input, confounder_input, labels = [], [], [], []
    
    num_users = train.shape[0]
    num_items = train.shape[1]
    max_item_index = confounders.shape[1] - 1

    for (u, i) in train.keys():
        rating = train[u, i]
        if rating > 3:  # Consider only ratings > 3 as positive instances
            user_input.append(u)
            item_input.append(i)
            confounder_input.append(confounders[u, i])
            labels.append(1)
        else:
            # Use ratings <= 3 as negative instances but still observed
            user_input.append(u)
            item_input.append(i)
            confounder_input.append(confounders[u, i])
            labels.append(0)

        # Generate negative instances
        for _ in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train.keys() or j >= confounders.shape[1]:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            confounder_input.append(confounders[u, j])
            labels.append(0)

    return np.array(user_input), np.array(item_input), np.array(confounder_input), np.array(labels)



CAUSEFIT_DIR = 'C:/Users/Sten Stokroos/Desktop/Thesis2.0/zelf/dat/out/ml_wg'

dim = 30
U = np.loadtxt(CAUSEFIT_DIR + '/cause_pmf_k' + str(dim) + '_U.csv')
V = np.loadtxt(CAUSEFIT_DIR + '/cause_pmf_k' + str(dim) + '_V.csv')
U = (np.atleast_2d(U.T).T)
V = (np.atleast_2d(V.T).T)
confounders = U.dot(V.T)

print(f"Confounders matrix shape: {confounders.shape}")


if __name__ == '__main__':
    args = parse_args()
    path = args.path
    dataset = args.dataset
    layers = eval(args.layers)
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose
    
    topK = 10
    evaluation_threads = 1
    print("MLP arguments: %s " % (args))

    timestamp = int(time())
    filename = f"{args.dataset}_MLP_{args.layers.replace(',', '_').replace(' ', '').replace('[', '').replace(']', '')}_{timestamp}.weights.h5"
    model_out_file = os.path.join('C:/Users/Sten Stokroos/Desktop/Thesis2.0/zelf/neural_collaborative_filtering/Pretrain', filename)

    # Load dataset
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print(f"Loaded dataset: #users={num_users}, #items={num_items}, #train={train.nnz}, #test={len(testRatings)}")

    # Load confounders
    dim = 100
    U = np.loadtxt(CAUSEFIT_DIR + '/cause_pmf_k' + str(dim) + '_U.csv')
    V = np.loadtxt(CAUSEFIT_DIR + '/cause_pmf_k' + str(dim) + '_V.csv')
    U = (np.atleast_2d(U.T).T)
    V = (np.atleast_2d(V.T).T)
    confounders = U.dot(V.T)

    print(f"Confounders matrix shape: {confounders.shape}")

    # Check for max indices in datasets
    max_user_index_train = max(train.keys(), key=lambda x: x[0])[0]
    max_item_index_train = max(train.keys(), key=lambda x: x[1])[1]
    max_user_index_test = max(testRatings, key=lambda x: x[0])[0]
    max_item_index_test = max(testRatings, key=lambda x: x[1])[1]

    print(f"Max user index in train data: {max_user_index_train}")
    print(f"Max item index in train data: {max_item_index_train}")
    print(f"Max user index in test data: {max_user_index_test}")
    print(f"Max item index in test data: {max_item_index_test}")

    if max_item_index_train >= confounders.shape[1]:
        print(f"Adjusting max item index from {max_item_index_train} to {confounders.shape[1] - 1}")
        max_item_index_train = confounders.shape[1] - 1

    if max_item_index_test >= confounders.shape[1]:
        print(f"Adjusting max item index from {max_item_index_test} to {confounders.shape[1] - 1}")
        max_item_index_test = confounders.shape[1] - 1

    # Verify and adjust train data
    def adjust_indices(data, max_user_idx, max_item_idx):
        adjusted_data = {}
        for key in data.keys():
            user, item = key
            if user <= max_user_idx and item <= max_item_idx:
                adjusted_data[key] = data[key]
        return adjusted_data

    train = adjust_indices(train, max_user_index_train, max_item_index_train)
    testRatings = [(u, i) if i <= max_item_index_test else (u, max_item_index_test) for (u, i) in testRatings]
    testNegatives = [[j if j <= max_item_index_test else max_item_index_test for j in neg_list] for neg_list in testNegatives]

    # Proceed with model setup and training if dimensions are correct
    model = get_model(num_users, num_items, layers, reg_layers)
    if learner.lower() == "adagrad":
        model.compile(optimizer=Adagrad(learning_rate=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(learning_rate=learning_rate), loss='binary_crossentropy')

    t1 = time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, confounders, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f]' % (hr, ndcg, time() - t1))

    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in tqdm(range(epochs)):
        t1 = time()
        user_input, item_input, confounder_input, labels = get_train_instances(train, confounders, num_negatives)
        hist = model.fit([user_input, item_input, confounder_input],
                         labels, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)

        t2 = time()
        if epoch % verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, confounders, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best MLP model is saved to %s" % (model_out_file))


# if __name__ == '__main__':
#     args = parse_args()
#     path = args.path
#     dataset = args.dataset
#     layers = eval(args.layers)
#     reg_layers = eval(args.reg_layers)
#     num_negatives = args.num_neg
#     learner = args.learner
#     learning_rate = args.lr
#     batch_size = args.batch_size
#     epochs = args.epochs
#     verbose = args.verbose

#     topK = 10
#     evaluation_threads = 1  # Adjust this if you use multi-threading
#     print("MLP arguments: %s " % (args))

#     timestamp = int(time())
#     filename = f"{args.dataset}_MLP_{args.layers.replace(',', '_').replace(' ', '').replace('[', '').replace(']', '')}_{timestamp}.weights.h5"
#     model_out_file = os.path.join('C:/Users/Sten Stokroos/Desktop/Thesis2.0/zelf/neural_collaborative_filtering/Pretrain', filename)

#     # Load dataset
#     t1 = time()
#     dataset = Dataset(args.path + args.dataset)
#     train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
#     num_users, num_items = train.shape
#     print(f"Loaded dataset: #users={num_users}, #items={num_items}, #train={train.nnz}, #test={len(testRatings)}")

#     # Load confounders
#     dim = 1
#     U = np.loadtxt(CAUSEFIT_DIR + '/cause_pmf_k' + str(dim) + '_U.csv')
#     V = np.loadtxt(CAUSEFIT_DIR + '/cause_pmf_k' + str(dim) + '_V.csv')
#     U = (np.atleast_2d(U.T).T)
#     V = (np.atleast_2d(V.T).T)
#     confounders = U.dot(V.T)

#     print(f"Confounders matrix shape: {confounders.shape}")

#     # Check for max indices in datasets
#     max_user_index_train = max(train.keys(), key=lambda x: x[0])[0]
#     max_item_index_train = max(train.keys(), key=lambda x: x[1])[1]
#     max_user_index_test = max(testRatings, key=lambda x: x[0])[0]
#     max_item_index_test = max(testRatings, key=lambda x: x[1])[1]

#     print(f"Max user index in train data: {max_user_index_train}")
#     print(f"Max item index in train data: {max_item_index_train}")
#     print(f"Max user index in test data: {max_user_index_test}")
#     print(f"Max item index in test data: {max_item_index_test}")

#     if max_item_index_train >= confounders.shape[1]:
#         print(f"Adjusting max item index from {max_item_index_train} to {confounders.shape[1] - 1}")
#         max_item_index_train = confounders.shape[1] - 1

#     if max_item_index_test >= confounders.shape[1]:
#         print(f"Adjusting max item index from {max_item_index_test} to {confounders.shape[1] - 1}")
#         max_item_index_test = confounders.shape[1] - 1

#     # Verify and adjust train data
#     def adjust_indices(data, max_user_idx, max_item_idx):
#         adjusted_data = {}
#         for key in data.keys():
#             user, item = key
#             if user <= max_user_idx and item <= max_item_idx:
#                 adjusted_data[key] = data[key]
#         return adjusted_data

#     train = adjust_indices(train, max_user_index_train, max_item_index_train)
#     testRatings = [(u, i) if i <= max_item_index_test else (u, max_item_index_test) for (u, i) in testRatings]
#     testNegatives = [[j if j <= max_item_index_test else max_item_index_test for j in neg_list] for neg_list in testNegatives]

#     # Proceed with model setup and training if dimensions are correct
#     model = get_model(num_users, num_items, layers, reg_layers)
#     if learner.lower() == "adagrad":
#         model.compile(optimizer=Adagrad(learning_rate=learning_rate), loss='binary_crossentropy')
#     elif learner.lower() == "rmsprop":
#         model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss='binary_crossentropy')
#     elif learner.lower() == "adam":
#         model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy')
#     else:
#         model.compile(optimizer=SGD(learning_rate=learning_rate), loss='binary_crossentropy')

#     t1 = time()
#     (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, confounders, topK, evaluation_threads)
#     hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
#     print('Init: HR = %.4f, NDCG = %.4f [%.1f]' % (hr, ndcg, time() - t1))

#     best_hr, best_ndcg, best_iter = hr, ndcg, -1
#     for epoch in tqdm(range(epochs)):
#         t1 = time()
#         user_input, item_input, confounder_input, labels = get_train_instances(train, confounders, num_negatives)
#         hist = model.fit([user_input, item_input, confounder_input],
#                          labels, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)

#         t2 = time()
#         if epoch % verbose == 0:
#             (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, confounders, topK, evaluation_threads)
#             hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
#             print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
#                   % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
#             if hr > best_hr:
#                 best_hr, best_ndcg, best_iter = hr, ndcg, epoch
#                 if args.out > 0:
#                     model.save_weights(model_out_file, overwrite=True)

#     print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
#     if args.out > 0:
#         print("The best MLP model is saved to %s" % (model_out_file))


