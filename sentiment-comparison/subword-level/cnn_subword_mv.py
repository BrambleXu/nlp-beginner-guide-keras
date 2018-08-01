import os
import numpy as np
import data_helpers
from data_helpers import TrainValTensorBoard
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Input, Embedding, Activation, Flatten, Dense, Concatenate
from keras.layers import Conv1D, MaxPooling1D, Dropout
from keras.models import Model

from data_helpers import BPE

#==================Preprocess===================

# Load data
positive_data_file = "../data/rt-polaritydata/rt-polarity.pos"
negtive_data_file = "../data/rt-polaritydata/rt-polarity.neg"

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(positive_data_file, negtive_data_file)

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


# replace all digits with 0
import re

train_texts = [re.sub('\d', '0', s) for s in x_text]

# replace all URLs with <url>
url_reg = r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b'
train_texts = [re.sub(url_reg, '<url>', s) for s in train_texts]

# Convert string to subword, this process may take several minutes
bpe = BPE("./pre-trained-model/en.wiki.bpe.op25000.vocab")
train_texts = [bpe.encode(s) for s in train_texts]

# Build vocab, {token: index}
vocab = {}
for i, token in enumerate(bpe.words):
    vocab[token] = i + 1

# Convert train and test
train_sentences = subword2index(train_texts, vocab)

# See char level length
length = [len(sent) for sent in train_sentences]
print('The max length is: ', max(length))
print('The min length is: ', min(length))
print('The average length is: ', sum(length)/len(length))

# Padding
from keras.preprocessing.sequence import pad_sequences
train_data = pad_sequences(train_sentences, maxlen=253, padding='post')

# Shuffle data
np.random.seed(42)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = train_data[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train and test
training_rate = 0.9
train_len = int(len(y) * training_rate)
x_train = x_shuffled[:train_len]
y_train = y_shuffled[:train_len]
x_test = x_shuffled[train_len:]
y_test = y_shuffled[train_len:]
print('Training data size is: ', x_train.shape)
print('Validation data size is: ', x_test.shape)

# Embedding Initialization
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format("./pre-trained-model/en.wiki.bpe.op25000.d50.w2v.bin", binary=True)

from keras.layers import Embedding

input_size = 253
embedding_dim = 50
embedding_weights = np.zeros((len(vocab) + 1, embedding_dim)) # (25001, 50)

for subword, i in vocab.items():
    if subword in model.vocab:
        embedding_vector = model[subword]
        if embedding_vector is not None:
            embedding_weights[i] = embedding_vector
    else:
#         print(subword) # print the subword in vocab but not in model
        continue

embedding_layer = Embedding(len(vocab)+1,
                            embedding_dim,
                            weights=[embedding_weights],
                            input_length=input_size)



#===================CNN Model===================
# Model Hyperparameters
embedding_dim = 50
filter_sizes = (3, 8)
num_filters = 10
vocab_size = len(vocab)
dropout_prob = 0.5
hidden_dims = 50
batch_size = 32
num_epochs = 10
sequence_length = 253


# Create model
# Input
input_shape = (sequence_length,)
input_layer = Input(shape=input_shape, name='input_layer')

# Embedding
embedded = embedding_layer(input_layer)

# CNN, iterate filter_size
conv_blocks = []
for fz in filter_sizes:
    conv = Conv1D(filters=num_filters,
                  kernel_size=fz,  # 3 means 3 words
                  padding='valid',  # valid means no padding
                  strides=1,
                  activation='relu',
                  use_bias=True)(embedded)
    conv = MaxPooling1D(pool_size=2)(conv) # (?, 27, 10), (?, 24, 10)
    conv = Flatten()(conv) # (?, 270), (?, 240)
    conv_blocks.append(conv) # [(?, 270), (?, 240)]

concat1max = Concatenate()(conv_blocks)  # (?, 510)
concat1max = Dropout(dropout_prob)(concat1max) # 0.5
output_layer = Dense(hidden_dims, activation='relu')(concat1max) # (?, 50)
output_layer = Dense(2, activation='sigmoid')(output_layer) # (?, 2)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# Train model with Early Stopping
earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
tensorboard = TrainValTensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=True)
model.fit(x_train, y_train,batch_size=batch_size, epochs=num_epochs, callbacks=[earlystopper, tensorboard],
          validation_split=0.1, shuffle=True, verbose=2)

# Evaluate
score = model.evaluate(x_test, y_test)
print('test_loss, test_acc: ', score)