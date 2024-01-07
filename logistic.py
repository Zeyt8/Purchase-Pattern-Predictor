import numpy as np

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
        return Y > .5
    
    def __logistic(self, x):
        return 1. / (1. + np.exp(-x))