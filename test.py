from keras.models import load_model
from utils import *
import numpy as np
import pandas as pd 

from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D,MaxPooling1D
from keras.layers.embeddings import Embedding

model = load_model('model.h5')

def predict(val):
    if val >= 0.5:
        print("Male with probability",val)
    else:
        print("Female with probability",1-val)
    

while True:
    name = input('Enter a name:').lower()
    name = conv2vec(name,15)
    name = np.array([name])
    val = model.predict(name)
    predict(val[0])
