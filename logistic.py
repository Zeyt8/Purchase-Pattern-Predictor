import numpy as np

def logistic(x):
    return 1. / (1. + np.exp(-x))

def nll(Y, T):
    return -np.sum(T * np.log(Y) + (1 - T) * np.log(1 - Y))

def accuracy(Y, T):
    predictions = (Y > 0.5).astype(int)
    correct_predictions = np.sum(predictions == T)
    return correct_predictions / T.size

def split_dataset(X, T, train=.8):
    N = X.shape[0]
    N_train = int(round(N * train))
    N_test = N - N_train

    X_train, X_test = X[:N_train,:], X[N_train:,:]
    T_train, T_test = T[:N_train], T[N_train:]
    return X_train, T_train, X_test, T_test

def train_logistic(X, T, lr=.01, epochs_no=100):
    (N, D) = X.shape
    X_hat = np.concatenate([X, np.ones((N, 1))], axis=1)
    W = np.random.randn((D+1))

    for epoch in range(epochs_no):
        Y = logistic(np.dot(X_hat,  W))
        W -= lr * np.dot(X_hat.T, Y - T) / N

    return W

def predict_logistic(X, W):
    (N, D) = X.shape
    X_hat = np.concatenate([X, np.ones((N, 1))], axis=1)
    Y = logistic(np.dot(X_hat, W))
    return Y