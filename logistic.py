import numpy as np

def split_dataset(X, T, train=.8):
    N = X.shape[0]
    N_train = int(round(N * train))

    X_train, X_test = X[:N_train,:], X[N_train:,:]
    T_train, T_test = T[:N_train], T[N_train:]
    return X_train, X_test, T_train, T_test

def logistic_accuracy(T, Y):
    predictions = (Y > 0.5).astype(int)
    correct_predictions = np.sum(predictions == T)
    return correct_predictions / T.size

class MyLogisticRegression:

    def fit(self, X, T, lr=.01, epochs_no=100):
        (N, D) = X.shape
        X_hat = np.concatenate([X, np.ones((N, 1))], axis=1)
        self.W = np.random.randn((D+1))

        for _ in range(epochs_no):
            Y = self.__logistic(np.dot(X_hat, self.W))
            self.W -= lr * np.dot(X_hat.T, Y - T) / N
    
    def predict(self, X):
        (N, _) = X.shape
        X_hat = np.concatenate([X, np.ones((N, 1))], axis=1)
        Y = self.__logistic(np.dot(X_hat, self.W))
        return Y
    
    def __logistic(self, x):
        return 1. / (1. + np.exp(-x))