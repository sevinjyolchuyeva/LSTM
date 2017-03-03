'''Trains a LSTM on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
Notes:

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function

import numpy as np
np.random.seed(7)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from keras.optimizers import RMSprop



#from keras.datasets import imdb

maxlen = 40  # cut texts after this number of words (among top max_features most common words)
step = 5
import string
import sys
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder
import itertools
import time
import os
start_time =time.time()
print ('100 epoch with early stop')
path1='/home/phd-sevinj2/work/word_embedding/data/tr_arrays.txt'
path2='/home/phd-sevinj2/work/word_embedding/data/tr_labels.txt'
path3='/home/phd-sevinj2/work/word_embedding/data/test_x.txt'
path4='/home/phd-sevinj2/work/word_embedding/data/test_y.txt'
path5='/home/phd-sevinj2/work/word_embedding/data/val_x.txt'
path6='/home/phd-sevinj2/work/word_embedding/data/val_y.txt'
path7='/home/phd-sevinj2/work/word_embedding/data/test_dat.txt'

Data=np.loadtxt(path1)
X_train=Data.reshape(25000,100)

Labels=np.loadtxt(path2)
Labels=Labels.reshape(25002,)
y_train= Labels[:25000]

test_x=np.loadtxt(path3)
X_test=test_x.reshape(300,100)

test_y=np.loadtxt(path4)
test_y=test_y.reshape(306,)
y_test=test_y[:300]

val_x=np.loadtxt(path5)
X_val=val_x.reshape(1000,100)

val_y=np.loadtxt(path6)
val_y=val_y.reshape(1008,)
y_val= val_y[:1000]

X_E=np.loadtxt(path7)
X_E=X_E.reshape(50,100)


print( 'X_train:',X_train.shape)
print( 'X_test:',y_train.shape)
print( 'y_train:',X_test.shape)
print( 'y_test:',y_test.shape)
print ('valx:', X_val.shape)
print('valy:', y_val.shape)
print('example_test:', X_E.shape)

embedding_vector_length=128
top_words=20000

# input shape has 3 items. this function changes data shape
def reshape_dataset(train):
    trainX = np.reshape(train, (train.shape[0], 1, train.shape[1]))
    return np.array(trainX)

XX_train = reshape_dataset(X_train)
XX_val= reshape_dataset(X_val)
XX_test= reshape_dataset(X_test)
XX_E = reshape_dataset(X_E)

model = Sequential()
model.add(LSTM(128,input_shape=(1,100)))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

#early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
model.fit(XX_train, y_train, validation_data=(XX_val, y_val), nb_epoch=30, batch_size=64)
scores= model.evaluate(XX_test, y_test,batch_size=64)
print ("Accuracy: %.2f%%" % (scores[1]*100))
                            
'''
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

'''

#class1----book1 and class0---- book2

class_zero=0
class_one=0
predictions = model.predict_classes(XX_E,batch_size=64,verbose=0) 
for x in predictions:
    if x == 0:
        class_zero=class_zero+1
    else:
        class_one=class_one+1

if class_zero < class_one:
    print('Example is included Book1')
elif class_zero==class_one:
    print('Example is included both Book1 and Book2 ')
else:
    print('Example is included Book2')

print('number_of_class_book2', class_zero)
print('number_of_class_book1', class_one)

print("--- %s seconds ---" % (time.time() - start_time))
