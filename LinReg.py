import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from math import sqrt
import random

class LinReg():
    def __init__(self):
        pass

    def train(self, data, y):
        model  = LinearRegression().fit(data, y)
        return model

    def get_fitness(self, x, y, random_state=0):
        if random_state==0:
            x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.5)
        else:
            x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.5, random_state=random_state)
        model = self.train(x_train, y_train)
        predictions = model.predict(x_test)
        e1 = sqrt(mean_squared_error(predictions, y_test))

        model = self.train(x_test, y_test)
        predictions = model.predict(x_train)
        e2 = sqrt(mean_squared_error(predictions, y_train))
        return  (e1+e2)/2

    def ger(self, x, y, n=10):
        indexes = np.asarray([i for i in range(x.shape[1])])
        removed_indexes = np.random.choice(indexes, n, replace=False)
        arr = np.asarray(x)
        arr = np.delete(arr, removed_indexes, axis=1)
        return self.get_fitness(arr, y)

    def get_columns(self,x,bitstring):
        indexes = []
        for i, s in enumerate(bitstring):
            if s=='0':
                indexes.append(i)
        arr = np.asarray(x)
        arr = np.delete(arr, indexes, axis=1)
        return arr

