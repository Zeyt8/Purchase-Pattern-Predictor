import numpy as np
import numpy.typing as npt

class MyLogisticRegression:

    def fit(self, X: npt.NDArray[np.float32], T: npt.NDArray[np.float32], lr: float = .01, epochs_no: int = 100) -> None:
        (N, D) = X.shape
        X_hat = np.concatenate([X, np.ones((N, 1))], axis=1)
        self.W = np.random.randn((D+1))

        for _ in range(epochs_no):
            Y = self.__logistic(np.dot(X_hat, self.W))
            self.W -= lr * np.dot(X_hat.T, Y - T) / N
    
    def predict(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
        (N, _) = X.shape
        X_hat = np.concatenate([X, np.ones((N, 1))], axis=1)
        Y = self.__logistic(np.dot(X_hat, self.W))
        return Y > .5
    
    def __logistic(self, x: float) -> float:
        return 1. / (1. + np.exp(-x))