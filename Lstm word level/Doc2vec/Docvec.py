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
np.random.seed(2000)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from keras.optimizers import RMSprop

import string
import sys
import numpy as np
import itertools
import time
import os
start_time =time.time()

from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# numpy
import numpy

# shuffle
from random import shuffle

# logging
import logging
import os.path
import sys
numpy.set_printoptions(threshold=sys.maxsize)

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))



class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources
        
        flipped = {}
        
        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')
    
    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])
    
    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences
    
    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences



sources = {'Don_Q.txt':'TRAIN1', 'Brothers.txt':'TRAIN0', 'example_Don.txt':'example' }


sentences = LabeledLineSentence(sources)

#print ('sentences:',sentences.objectname() )


model = Doc2Vec(dm=0,min_count=1, window=10, size=200, sample=1e-4, negative=5, workers=10)

model.build_vocab(sentences.to_array())
for epoch in range(50):
    logger.info('Epoch %d' % epoch)
    model.train(sentences.sentences_perm())
model.save('./imdb.d2v')
model = Doc2Vec.load('./imdb.d2v')


model.docvecs['TRAIN1_0']
train_arrays = numpy.zeros((50000, 200))
train_labels = numpy.zeros(50000)
test_x=numpy.zeros((6000,200))
test_y=numpy.zeros(6000)
val_x=numpy.zeros((6000,200))
val_y=numpy.zeros(6000)

new_data=numpy.zeros((50,200))

for i in range(25000):
    prefix_train_pos = 'TRAIN1_' + str(i)
    prefix_train_neg = 'TRAIN0_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_arrays[25000 + i] = model.docvecs[prefix_train_neg]
    train_labels[i] = 1
    train_labels[25000 + i] = 0
##for test data 6000
for i in range(3000):	
	prefix_test_book1 = 'TRAIN1_' + str(25000+i)
	prefix_test_book2 = 'TRAIN0_' + str(25000+i)
	test_x[i] = model.docvecs[prefix_test_book1]
	test_x[3000 + i] = model.docvecs[prefix_test_book2]
	test_y[i] = 1
	test_y[3000 + i] = 0

for i in range(3000):	
	prefix_test_val1= 'TRAIN1_' + str(i+28000)
	prefix_test_val2 = 'TRAIN0_' + str(i+28000)
	val_x[i]= model.docvecs[prefix_test_val1]
	val_x[3000 + i] = model.docvecs[prefix_test_val2]
	val_y[i] = 1
	val_y[3000 + i] = 0

for i in range(50):	
	prefix_test_data= 'example_' + str(i)
	new_data[i]= model.docvecs[prefix_test_data]



X_train=train_arrays
y_train= train_labels

X_test=test_x
y_test=test_y

X_val=val_x
y_val= val_y

X_E=new_data


print( 'X_train:',X_train.shape)
print( 'X_test:',y_train.shape)
print( 'y_train:',X_test.shape)
print( 'y_test:',y_test.shape)
print ('valx:', X_val.shape)
print('valy:', y_val.shape)
print('example_test:', X_E.shape)


# input shape has 3 items. this function changes data shape
def reshape_dataset(train):
    trainX = np.reshape(train, (train.shape[0], 1, train.shape[1]))
    return np.array(trainX)

XX_train = reshape_dataset(X_train)
XX_val= reshape_dataset(X_val)
XX_test= reshape_dataset(X_test)
XX_E = reshape_dataset(X_E)

model = Sequential()
model.add(LSTM(128,input_shape=(1,200)))
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

print('number_of_class_book1', class_one)
print('number_of_class_book0', class_zero)

print("--- %s seconds ---" % (time.time() - start_time))

