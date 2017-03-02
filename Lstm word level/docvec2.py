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

# classifier
from sklearn.linear_model import LogisticRegression

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

		
#path1='C:/Users/Agil/Desktop/Don_Q.txt'
#path2='C:/Users/Agil/Desktop/Brothers.txt'
'''path3='C:/Users/Agil/Desktop/Don_test.txt'
path4='C:/Users/Agil/Desktop/Brot_test.txt'

#text1 = open(path1).read().lower()
text2 = open(path2).read().lower()
text1_test = open(path3).read().lower()
text2_test = open(path4).read().lower()	
print (text2_test)
'''

sources = {'Don_test.txt':'TEST_text1', 'Brothers_test.txt':'TEST_text2', 'Don_Q.txt':'TRAIN1', 'Brothers.txt':'TRAIN2','Val_don.txt':'val1', 'Val_bro.txt':'val2', 'example_Don.txt':'example' }


sentences = LabeledLineSentence(sources)

print ('sentences:',sentences)


model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=16)

model.build_vocab(sentences.to_array())
for epoch in range(50):
    logger.info('Epoch %d' % epoch)
    model.train(sentences.sentences_perm())
model.save('./imdb.d2v')
model = Doc2Vec.load('./imdb.d2v')


model.docvecs['TRAIN1_0']
print('brotest:')
model.docvecs['TEST_text2_0']
train_arrays = numpy.zeros((25000, 100))
train_labels = numpy.zeros(25000)
test_x=numpy.zeros((300,100))
test_y=numpy.zeros(300)
val_x=numpy.zeros((1000,100))
val_y=numpy.zeros(1000)

new_data=numpy.zeros((50,100))

for i in range(12500):
    prefix_train_pos = 'TRAIN1_' + str(i)
    prefix_train_neg = 'TRAIN2_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_arrays[12500 + i] = model.docvecs[prefix_train_neg]
    train_labels[i] = 1
    train_labels[12500 + i] = 0
	##for test data
for i in range(150):	
	prefix_test_book1 = 'TEST_text1_' + str(i)
	prefix_test_book2 = 'TEST_text2_' + str(i)
	test_x[i] = model.docvecs[prefix_test_book1]
	test_x[150 + i] = model.docvecs[prefix_test_book2]
	test_y[i] = 1
	test_y[150 + i] = 0
	
for i in range(500):	
	prefix_test_val1= 'val1_' + str(i)
	prefix_test_val2 = 'val2_' + str(i)
	val_x[i]= model.docvecs[prefix_test_val1]
	val_x[500 + i] = model.docvecs[prefix_test_val2]
	val_y[i] = 1
	val_y[500 + i] = 0
	
for i in range(50):	
	prefix_test_data= 'example_' + str(i)
	new_data[i]= model.docvecs[prefix_test_data]
	
	
	
  


sys.stdout = open('tr_arrays.txt', 'w')
numpy.set_printoptions(threshold=sys.maxsize)
print( train_arrays)
sys.stdout.close()

sys.stdout = open('tr_labels.txt', 'w')
numpy.set_printoptions(threshold=sys.maxsize)
print( train_labels)
sys.stdout.close()

sys.stdout = open('test_x.txt', 'w')
numpy.set_printoptions(threshold=sys.maxsize)
print(test_x)
sys.stdout.close()

sys.stdout = open('test_y.txt', 'w')
numpy.set_printoptions(threshold=sys.maxsize)
print( test_y)
sys.stdout.close()

sys.stdout = open('val_x.txt', 'w')
numpy.set_printoptions(threshold=sys.maxsize)
print( val_x)
sys.stdout.close()

sys.stdout = open('val_y.txt', 'w')
numpy.set_printoptions(threshold=sys.maxsize)
print( val_y)
sys.stdout.close()

sys.stdout = open('test_dat.txt', 'w')
numpy.set_printoptions(threshold=sys.maxsize)
print( new_data)
sys.stdout.close()


classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)
score= classifier.score(test_x, test_y)
print('score: ', score)

