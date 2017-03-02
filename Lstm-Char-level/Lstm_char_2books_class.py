'''Trains a LSTM on 2 books classification task
books link are in below:
Don Quixote (2.2MB)  -- http://www.gutenberg.org/ebooks/996
The Brothers Karamazov (1.9 MB) -- https://www.gutenberg.org/ebooks/28054

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
import sys
import time
import os


maxlen = 40  # cut texts after this number of words (among top max_features most common words)
step = 5

start_time =time.time()

'''
Before I thought I can get better result firstly 
with to normalize my book. After training I saw that it doesnot have 
an affect to  the accurancy
    
def normalize_text(text):
    norm_text = text.lower()

    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', ' ')

    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':', '--' ]:
        norm_text = norm_text.replace(char, ' ' + char + ' ')

    return norm_text

'''
 
path1='/home/phd-sevinj2/work/lstm_2book/data/Don_Q.txt'
path2='/home/phd-sevinj2/work/lstm_2book/data/Brothers.txt'


book1 = open(path1).read()
#text1=normalize_text(book1)
text1=book1
print('Length_of_book1:', len(text1))

book2 = open(path2).read()
#text2=normalize_text(book2)
text2=book2
print('Length_of_book2:', len(text2))


ch1=sorted(list(set(text1)))
ch2=sorted(list(set(text2)))

'''I wanted to check that I have new data which after training processs I try find his class, or not.
If I have, then I should the same vectorization with books'''

ex_path=os.path.isfile ('/home/phd-sevinj2/work/lstm_2book/data/example_Don.txt')
if ex_path == True:
           example = open('/home/phd-sevinj2/work/lstm_2book/data/example_Don.txt').read()
           #text5=normalize_text(example)
           text5=example
           print('Length_of_example:', len(text5))           
           ch5=sorted(list(set(example)))
else:
           ch5=[]
'I have maked the set of chars which helps me doing the same vectorization'                   
all_chars= sorted(list(set(ch1+ch2+ch5)))
len_char=len(all_chars)                      
print('total chars:', len(all_chars))
#print('all_chars:', all_chars)                           
char_indices = dict((c, i) for i, c in enumerate(all_chars))
indices_char = dict((i, c) for i, c in enumerate(all_chars))
if ch5!=[]:
          sentences5 = []
          next_chars5 = []

          for i in range(0, len(text5) - maxlen, step):
              sentences5.append(text5[i: i + maxlen])
              next_chars5.append(text5[i + maxlen])
              #print('nb sequences:', len(sentences5))
              exam_length= len(sentences5)
          
          print('Vectorization of example...')
          X_E = np.zeros((len(sentences5), maxlen, len_char), dtype=np.bool)
          for i, sentence in enumerate(sentences5):
              for t, char in enumerate(sentence):
                  X_E[i, t, char_indices[char]] = 1

          print('Example shape:', X_E.shape)


print('Vectorization of book1...')
          
sentences1 = []
next_chars1= []

for i in range(0, len(text1) - maxlen, step):
    sentences1.append(text1[i: i + maxlen])
    next_chars1.append(text1[i + maxlen])
#print('nb sequences:', len(sentences1))

X_b1 = np.zeros((len(sentences1), maxlen, len_char), dtype=np.bool)
for i, sentence in enumerate(sentences1):
    for t, char in enumerate(sentence):
        X_b1[i, t, char_indices[char]] = 1

print('shape of book1:', X_b1.shape)


print('Vectorization of book2...')

sentences3 = []
next_chars3 = []

for i in range(0, len(text2) - maxlen, step):
    sentences3.append(text2[i: i + maxlen])
    next_chars3.append(text2[i + maxlen])
#print('nb sequences:', len(sentences3))


X_b2 = np.zeros((len(sentences3), maxlen, len_char), dtype=np.bool)
for i, sentence in enumerate(sentences3):
    for t, char in enumerate(sentence):
        X_b2[i, t, char_indices[char]] = 1

print('shape_of_book2: ' , X_b2.shape)

'I splitted my data 70%:15%:15% respectively training, validation and test data'
l11=len(sentences1)
l22=len(sentences3)		
l1=int((l11*70)/100) 
l2=int((l22*70)/100)

np.random.shuffle(X_b1)
tr_X_b1, test_X_b1 = X_b1[:l1,:], X_b1[l1:,:]
np.random.shuffle(test_X_b1)
X_b1_test,X_b1_val = test_X_b1[:int((l11-l1)/2),:], test_X_b1[int((l11-l1)/2):,:]

#print('tr_X_b1: ', tr_X_b1.shape)
#print('xtest:' ,X_b1_test.shape)
#print ('val:', X_b1_val.shape)

print('\n')
np.random.shuffle(X_b2)
tr_X_b2, test_X_b2 = X_b2[:l2,:], X_b2[l2:,:]
#print('test xb2:', test_X_b2.shape)
np.random.shuffle(test_X_b2)
X_b2_test,X_b2_val = test_X_b2[:int((l22-l2)/2),:], test_X_b2[int((l22-l2)/2):,:]

'At this step I concatenated book1 and book2 training,test and validation data'
X_train= np.concatenate((tr_X_b1, tr_X_b2), axis=0)
X_test= np.concatenate((X_b1_test, X_b2_test), axis=0)
X_val= np.concatenate((X_b1_val, X_b2_val), axis=0)

print('Xtrain_shape:', X_train.shape)
print('Xtest_shape:', X_test.shape)
print('Xval_shape:', X_val.shape)

y_book1_train=np.ones((tr_X_b1.shape[0],), dtype=np.int)
y_book1_test=np.ones((X_b1_test.shape[0],), dtype=np.int)
y_book1_val=np.ones((X_b1_val.shape[0],), dtype=np.int)


y_book2_train=np.zeros((tr_X_b2.shape[0],), dtype=np.int)
y_book2_test=np.zeros((X_b2_test.shape[0],), dtype=np.int)
y_book2_val=np.zeros((X_b2_val.shape[0],), dtype=np.int)


y_train= np.concatenate((y_book1_train, y_book2_train), axis=0)
y_test= np.concatenate((y_book1_test, y_book2_test), axis=0)
y_val= np.concatenate((y_book1_val, y_book2_val), axis=0)

print('ytrain_shape', y_train.shape)
print('ytest_shape', y_test.shape)
print('yval_shape', y_val.shape)


print('Build model...')

model = Sequential()

model.add(LSTM(128,input_shape=(maxlen, len_char)))  
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())
model.fit(X_train, y_train, validation_data=(X_val, y_val), nb_epoch=5, batch_size=64)
scores= model.evaluate(X_test, y_test,batch_size=64)
print("Accuracy: %.2f%%" % (scores[1]*100))

'''
#I think there is need to save and again load
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
                            
'At this part I tried predict class for eaxmple data'
#class1----book1 and class0---- book2

class_zero=0
class_one=0
predictions = model.predict_classes(X_E,batch_size=64,verbose=0) 
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
print('number_of_class_book2', class_zero)

print("--- %s seconds ---" % (time.time() - start_time))
