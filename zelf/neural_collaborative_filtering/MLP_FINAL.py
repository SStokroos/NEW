import numpy as np
from time import time
import os
from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, Adagrad, RMSprop, SGD
from tensorflow.keras.initializers import RandomNormal
import pandas as pd
import argparse

from evaluate import evaluate_model
from Dataset_new import Dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--train_path', nargs='?', default='C:/Users/Sten Stokroos/Desktop/NEW/zelf/Data/out/ml_train.csv',
                        help='Input training data path.')
    parser.add_argument('--test_path', nargs='?', default='C:/Users/Sten Stokroos/Desktop/NEW/zelf/Data/out/ml_test.csv',
                        help='Input testing data path.')
    parser.add_argument('--neg_path', nargs='?', default='C:/Users/Sten Stokroos/Desktop/NEW/zelf/Data/out/ml_negatives.csv',
                        help='Input negative data path.')
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

    embedding_initializer = RandomNormal(mean=0.0, stddev=0.01)
    user_embedding = Embedding(input_dim=num_users, output_dim=layers[0]//2,
                               embeddings_initializer=embedding_initializer,
                               embeddings_regularizer=l2(reg_layers[0]),
                               name='user_embedding')(user_input)
    item_embedding = Embedding(input_dim=num_items, output_dim=layers[0]//2,
                               embeddings_initializer=embedding_initializer,
                               embeddings_regularizer=l2(reg_layers[0]),
                               name='item_embedding')(item_input)
    
    user_latent = Flatten()(user_embedding)
    item_latent = Flatten()(item_embedding)

    vector = Concatenate()([user_latent, item_latent])

    for idx, layer_size in enumerate(layers[1:], start=1):
        vector = Dense(layer_size, activation='relu',
                       kernel_regularizer=l2(reg_layers[idx]),
                       name=f'layer{idx}')(vector)

    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='prediction')(vector)

    model = Model(inputs=[user_input, item_input], outputs=prediction)
    return model

def get_train_instances(train, num_negatives, num_items):
    user_input, item_input, labels = [], [], []
    num_users = train.shape[0]

    for (u, i) in train.keys():
        user_input.append(u)
        item_input.append(i)
        labels.append(1)

        for _ in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return np.array(user_input), np.array(item_input), np.array(labels)

def generate_predictions(model, num_users, num_items):
    user_input = np.repeat(np.arange(num_users), num_items)
    item_input = np.tile(np.arange(num_items), num_users)
    predictions = model.predict([user_input, item_input], batch_size=256, verbose=1)
    return user_input, item_input, predictions

if __name__ == '__main__':
    args = parse_args()
    
    # Load data
    # Load data
    train_df = pd.read_csv(args.train_path, sep=",", header=0)
    test_df = pd.read_csv(args.test_path, sep=",", header=0)

    # Combine train and test dataframes
    combined_df = pd.concat([train_df, test_df])

    # Calculate max user ID and max item ID
    max_user_id = combined_df['userId'].max() + 1
    max_item_id = combined_df['songId'].max() + 1

    print(f'Max User ID: {max_user_id}')
    print(f'Max Item ID: {max_item_id}')
    
    dataset = Dataset(train_df, test_df, args.neg_path, max_user_id, max_item_id)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape

    model = get_model(num_users, num_items, eval(args.layers), eval(args.reg_layers))
    if args.learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(learning_rate=args.lr), loss='binary_crossentropy')
    elif args.learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(learning_rate=args.lr), loss='binary_crossentropy')
    elif args.learner.lower() == "adam":
        model.compile(optimizer=Adam(learning_rate=args.lr), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(learning_rate=args.lr), loss='binary_crossentropy')

    t1 = time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, 10, 1)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f]' %(hr, ndcg, time()-t1))

    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in tqdm(range(args.epochs)):
        t1 = time()
        user_input, item_input, labels = get_train_instances(train, args.num_neg, num_items)
        hist = model.fit([user_input, item_input], labels, batch_size=args.batch_size, epochs=1, verbose=1, shuffle=True)
        t2 = time()

        if epoch % args.verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, 10, 1)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model_out_file = os.path.join('C:/Users/Sten Stokroos/Desktop/zelf/neural_collaborative_filtering/Pretrain', f"{args.dataset}_MLP_{args.layers.replace(',', '_').replace(' ', '').replace('[', '').replace(']', '')}_{int(time())}.weights.h5")
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best MLP model is saved to %s" %(model_out_file))
    
    user_input, item_input, predictions = generate_predictions(model, num_users, num_items)
    prediction_matrix = np.zeros((num_users, num_items))
    for user, item, prediction in zip(user_input, item_input, predictions):
        prediction_matrix[user, item] = prediction[0]

    np.savetxt(args.pred_out, prediction_matrix, delimiter=',')
    print(f"Predicted scores are saved to {args.pred_out}")
