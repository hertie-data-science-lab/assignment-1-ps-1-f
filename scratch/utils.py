import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return np.exp(-x) / ((np.exp(-x) + 1) ** 2)


def softmax(x):
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, axis=0)


def softmax_deriv(x):
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_deriv(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_pred.shape[0]

# ADDED for hyperparameter tuning
def gen_random_search(h1, hratio, lr, n_rand):
    '''Generates randomized size lists and learning rate for networks tuning
       h1, ratio, lr: inputs are tuples for min and max into randomization
       returns sizes (list) and initial learning_rate'''
    
    h1_sizes = np.random.randint(h1[0], h1[1], size=n_rand)
    hratios = np.random.uniform(hratio[0], hratio[1], size=n_rand)
    lrs = np.random.uniform(lr[0], lr[1], size=n_rand) #maybe logarithmic better?

    searches = []
    for h1_size, size_ratio, lr in zip(h1_sizes, hratios, lrs):
        h1_size = int(h1_size)
        h2_size = int(h1_size / size_ratio)
        sizes = [784, h1_size, h2_size, 10] #hard-coded as per problem definition
        lr = round(lr, 2)
        searches.append((sizes, lr))

    return searches
