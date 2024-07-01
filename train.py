import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)

import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import pandas as pd
from joblib import dump
from sklearn import preprocessing

def train():

    # Load, read and normalize training data
    training = "./training.csv"
    data_train = pd.read_csv(training)
        
    y_train = data_train['Income'].values
    X_train = data_train.drop(columns=['Income'],axis = 1)

    print("Shape of the training data")
    print(X_train.shape)
    print(y_train.shape)
        
    # Data normalization (0,1)
    X_train = preprocessing.normalize(X_train, norm='l2')
    
    # Models training
    
    # Linear Discrimant Analysis (Default parameters)
    model_1 = LinearDiscriminantAnalysis()
    model_1.fit(X_train, y_train)
    
    # Save model
    from joblib import dump
    dump(model_1, 'Inference_lda.joblib')
        
    # Neural Networks multi-layer perceptron (MLP) algorithm
    model_2 = MLPClassifier(solver='adam', activation='relu', alpha=0.0001, hidden_layer_sizes=(500,), random_state=0, max_iter=1000)
    model_2.fit(X_train, y_train)
       
    # Save model
    from joblib import dump
    dump(model_2, 'Inference_NN.joblib')
        
if __name__ == '__main__': train()