import numpy as np
import typing


class MinMaxScaler:
    def fit(self, data: np.ndarray) -> None:

        self.max = np.amax(data, axis=0)
        self.min = np.amin(data, axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
       
        res = np.zeros((data.shape[0], data.shape[1]))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                res[i][j] = (data[i][j] - self.min[j])/(self.max[j] - self.min[j])
        return res


class StandardScaler:
    def fit(self, data: np.ndarray) -> None:
        
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        
        res = np.zeros((data.shape[0], data.shape[1]))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                res[i][j] = (data[i][j] - self.mean[j])/self.std[j]
        return res
