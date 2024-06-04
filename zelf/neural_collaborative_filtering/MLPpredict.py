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

from Dataset import Dataset
from evaluate import evaluate_model
import argparse
import pandas as pd  # Added for CSV export

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='C:/Users/Sten Stokroos/Desktop/zelf/neural_collaborative_filtering/Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=10,
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
    parser.add_argument('--pred_out', type=str, default='C:/Users/Sten Stokroos/Desktop/zelf/neural_collaborative_filtering/Data/predicted_scores.csv',
                        help='Output CSV file for predicted scores.')
    return parser.parse_args()

def get_model(num_users, num_items, layers, reg_layers):
    assert len(layers) == len(reg_layers), "Each layer should have a corresponding regularization parameter"
    
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    # Using TensorFlow Keras initializers and regularizers
    embedding_initializer = RandomNormal(mean=0.0, stddev=0.01)
    user_embedding = Embedding(input_dim=num_users, output_dim=layers[0]//2,
                               embeddings_initializer=embedding_initializer,
                               embeddings_regularizer=l2(reg_layers[0]),
                               name='user_embedding')(user_input)
    item_embedding = Embedding(input_dim=num_items, output_dim=layers[0]//2,
                               embeddings_initializer=embedding_initializer,
                               embeddings_regularizer=l2(reg_layers[0]),
                               name='item_embedding')(item_input)
    
    print(layers[0]/2)

    # Flatten the embedding output to remove the sequence dimension
    user_latent = Flatten()(user_embedding)
    item_latent = Flatten()(item_embedding)

    # Concatenate the user and item embeddings
    vector = Concatenate()([user_latent, item_latent])

    # Add MLP layers as specified in the layers list
    for idx, layer_size in enumerate(layers[1:], start=1):  # Start from 1 since layer[0] is for embeddings
        vector = Dense(layer_size, activation='relu',
                       kernel_regularizer=l2(reg_layers[idx]),
                       name=f'layer{idx}')(vector)

    # Prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='prediction')(vector)

    model = Model(inputs=[user_input, item_input], outputs=prediction)
    return model



def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [], [], []
    num_users = train.shape[0]  # Assuming this is correct based on your dataset's structure

    # Iterate over each user-item pair in the train dataset
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)

        # print(f"Processing positive instance for user {u}")
        # negative instances
        for _ in range(num_negatives):
            j = np.random.randint(num_items)
            # Ensure the negative example is not a positive example
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u) 
            item_input.append(j)
            labels.append(0)
        # print(f"Processed {num_negatives} negative instances for user {u}")
    return np.array(user_input), np.array(item_input), np.array(labels)

def generate_predictions(model, num_users, num_items):
    user_input = np.repeat(np.arange(num_users), num_items)
    item_input = np.tile(np.arange(num_items), num_users)
    predictions = model.predict([user_input, item_input], batch_size=256, verbose=1)
    return user_input, item_input, predictions


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
    evaluation_threads = 1 #mp.cpu_count()
    print("MLP arguments: %s " %(args))

    timestamp = int(time())
    filename = f"{args.dataset}_MLP_{args.layers.replace(',', '_').replace(' ', '').replace('[', '').replace(']', '')}_{timestamp}.weights.h5"
    model_out_file = os.path.join('C:/Users/Sten Stokroos/Desktop/zelf/neural_collaborative_filtering/Pretrain', filename)
    # model_out_file = 'Pretrain/%s_MLP_%s_%d.weights.h5' % (args.dataset, args.layers, time())

    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    print(train.shape)

    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))
    

    # Build model
    model = get_model(num_users, num_items, layers, reg_layers)
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(learning_rate=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
            model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy')
    else:
        
        model.compile(optimizer=SGD(learning_rate=learning_rate), loss='binary_crossentropy')    
    
    # Check Init performance
    t1 = time()

    CAUSEFIT_DIR = 'C:/Users/Sten Stokroos/Desktop/zelf/dat/out/ml_wg'
    
    dim = 30 
    U = np.loadtxt(CAUSEFIT_DIR + '/cause_pmf_k'+str(dim)+'_U.csv')
    V = np.loadtxt(CAUSEFIT_DIR + '/cause_pmf_k'+str(dim)+'_V.csv')
    U = (np.atleast_2d(U.T).T)
    V = (np.atleast_2d(V.T).T)
    substitute_values = U.dot(V.T)
 
    print("Substitute confounders matrix shape:", substitute_values.shape)
    print("Type of substitute_values:", type(substitute_values))

    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f]' %(hr, ndcg, time()-t1))
    

    # Train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in tqdm(range(epochs)):
        t1 = time()
        user_input, item_input, labels = get_train_instances(train, num_negatives)
        print("Model will be fitted here")
        hist = model.fit([user_input, item_input], labels, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)
        t2 = time()

        # Evaluation
        if epoch % verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best MLP model is saved to %s" %(model_out_file))
    
    # Generate predictions and save to CSV
    user_input, item_input, predictions = generate_predictions(model, num_users, num_items)
    prediction_matrix = np.zeros((num_users, num_items))
    for user, item, prediction in zip(user_input, item_input, predictions):
        prediction_matrix[user, item] = prediction[0]  # Extract scalar from array

    pred_out_file = args.pred_out
    np.savetxt(pred_out_file, prediction_matrix, delimiter=',')
    print(f"Predicted scores are saved to {pred_out_file}")

    
