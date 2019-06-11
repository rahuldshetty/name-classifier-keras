from utils import *
import numpy as np
import pandas as pd 


from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D,MaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.utils import shuffle

PATH = "data.txt"

data  = read_data(PATH)

no_chars = 0

chars = get_chars()

for item in data:
    name = item[0]
    if len(name) > no_chars: no_chars = len(name)

newData = []

for item in data:
    name = item[0]
    name = conv2vec(name,no_chars)
    newData.append(name + [item[1]])


 
vocab_size = 26 + 2

print("Dimension:",no_chars,"VOCAB_SIZE",vocab_size)

npData = np.array(newData)

np.random.shuffle(npData)


x = npData[0:,0:15]
y = npData[0:,15]


LIM = int(0.7*len(x))

xtrain = x[0:LIM]
ytrain = y[0:LIM]
xtest = x[LIM:]
ytest = y[LIM:]

model = Sequential()
model.add(Embedding(vocab_size,128,input_length=no_chars))
model.add(Conv1D(filters=128,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(filters=32,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPooling1D(2))
model.add(LSTM(32))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
h = model.fit(xtrain,ytrain,epochs=80,batch_size=128,validation_data=(xtest,ytest),verbose=1)
model.save('model.h5')