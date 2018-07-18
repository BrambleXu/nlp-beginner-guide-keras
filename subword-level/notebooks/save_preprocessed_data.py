import sys, os
sys.path.append(os.pardir)
from data_helpers import BPE

#=======================All Preprocessing====================

# load data
import numpy as np
import pandas as pd
train_data_source = '../../char-level-cnn/data/ag_news_csv/train.csv'
test_data_source = '../../char-level-cnn/data/ag_news_csv/test.csv'
train_df = pd.read_csv(train_data_source, header=None)
test_df = pd.read_csv(test_data_source, header=None)

# concatenate column 1 and column 2 as one text
for df in [train_df, test_df]:
    df[1] = df[1] + df[2]
    df = df.drop([2], axis=1)
    
# convert string to lower case 
train_texts = train_df[1].values 
train_texts = [s.lower() for s in train_texts]
test_texts = test_df[1].values 
test_texts = [s.lower() for s in test_texts]

# replace all digits with 0
import re
train_texts = [re.sub('\d', '0', s) for s in train_texts]
test_texts = [re.sub('\d', '0', s) for s in test_texts]

# replace all URLs with <url> 
url_reg  = r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b'
train_texts = [re.sub(url_reg, '<url>', s) for s in train_texts]
test_texts = [re.sub(url_reg, '<url>', s) for s in test_texts]

# Convert string to subword, this process may take several minutes
bpe = BPE("../pre-trained-model/en.wiki.bpe.op25000.vocab")
train_texts = [bpe.encode(s) for s in train_texts]
test_texts = [bpe.encode(s) for s in test_texts]

# Build vocab, {token: index}
vocab = {}
for i, token in enumerate(bpe.words):
    vocab[token] = i + 1
    
# Convert subword to index, function version 
def subword2index(texts, vocab):
    sentences = []
    for s in texts:
        s = s.split()
        one_line = []
        for word in s:
            if word not in vocab.keys():
                one_line.append(vocab['unk'])
            else:
                one_line.append(vocab[word])
        sentences.append(one_line)
    return sentences

# Convert train and test 
train_sentences = subword2index(train_texts, vocab)
test_sentences = subword2index(test_texts, vocab)

# Padding
from keras.preprocessing.sequence import pad_sequences
train_data = pad_sequences(train_sentences, maxlen=1014, padding='post')
test_data = pad_sequences(test_sentences, maxlen=1014, padding='post')

# Convert to numpy array
train_data = np.array(train_data)
test_data = np.array(test_data)

#=======================Get classes================
train_classes = train_df[0].values
train_class_list = [x-1 for x in train_classes]
test_classes = test_df[0].values
test_class_list = [x-1 for x in test_classes]

from keras.utils import to_categorical
train_classes = to_categorical(train_class_list)
test_classes = to_categorical(test_class_list)


# Save 
data_dir = '../preprocessed_dataset.npz'
np.savez(data_dir, x_train=train_data, y_train=train_classes, x_test=test_data, y_test=test_classes)
# # This file is very big, 519.6MB