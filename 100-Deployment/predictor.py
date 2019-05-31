import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class my_customer_predictor():
    def __init__(self):
        pass

    def deserialize(self):
        with open('knn_model.obj', 'rb') as handle:
            model = pickle.load(handle)
        return model
    
    def predict(self, age, gender, income, spend):
        model = self.deserialize()
        return model.predict(np.array([[age, gender, income, spend]]))
        