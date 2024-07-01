import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)

import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import pandas as pd
from joblib import dump, load
from sklearn import preprocessing

def inference():

    # Load, read and normalize training data
    testing = "./testingg.csv"
    data_test = pd.read_csv(testing)
        
    y_test = data_test['Income'].values
    X_test = data_test.drop(columns=['Income'], axis = 1)
   
    print("Shape of the test data")
    print(X_test.shape)
    print(y_test.shape)
    
    # Data normalization (0,1)
    X_test = preprocessing.normalize(X_test, norm='l2')
    
    # Models training
    
    # Run model
    model_1 = load('Inference_lda.joblib')
    print("LDA score and classification:")
    print(model_1.score(X_test, y_test))
    print(model_1.predict(X_test))
        
    # Run model
    model_2 = load('Inference_NN.joblib')
    print("NN score and classification:")
    print(model_2.score(X_test, y_test))
    print(model_2.predict(X_test))
    
if __name__ == '__main__': inference()
