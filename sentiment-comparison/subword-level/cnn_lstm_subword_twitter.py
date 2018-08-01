import os
import numpy as np
import pandas as pd
import data_helpers
import pickle
from data_helpers import TrainValTensorBoard
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Embedding, Activation, Flatten, Dense, Concatenate
from keras.layers import Conv1D, MaxPooling1D, Dropout, LSTM
from keras.models import Model
from keras.callbacks import CSVLogger
from data_helpers import BPE


# read data from saved file
dataset = np.load('../data/twitter/preprocessed_dataset.npz')

x_train = dataset['x_train']
y_train = dataset['y_train']
x_test = dataset['x_test']
y_test = dataset['y_test']

print('Training data size is: ', x_train.shape)
print('Validation data size is: ', x_test.shape)


# Load vocab
bpe = BPE("./pre-trained-model/en.wiki.bpe.op25000.vocab")
# Build vocab, {token: index}
vocab = {}
for i, token in enumerate(bpe.words):
    vocab[token] = i + 1

# Embedding Initialization
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format("./pre-trained-model/en.wiki.bpe.op25000.d50.w2v.bin", binary=True)

from keras.layers import Embedding

input_size = 364
embedding_dim = 50
embedding_weights = np.zeros((len(vocab) + 1, embedding_dim)) # (25001, 50)


for subword, i in vocab.items():
    if subword in model.vocab:
        embedding_vector = model[subword]
        if embedding_vector is not None:
            embedding_weights[i] = embedding_vector
    else:
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
dropout_prob = 0.5
hidden_dims = 50
batch_size = 32
num_epochs = 40
sequence_length = 364


# Create model
# Input
input_shape = (sequence_length,)
input_layer = Input(shape=input_shape, name='input_layer')  # (?, 56)

# Embedding
embedded = embedding_layer(input_layer) # (batch_size, sequence_length, output_dim)=(?, 364, 50),

# CNN+LSTM
# https://github.com/pmsosa/CS291K/blob/master/cnn_lstm.py#L40

# x = Conv1D(filters=num_filters,
#            kernel_size=filter_sizes[1],  # 3 means 3 words
#            padding='valid',  # valid means no padding
#            strides=1,
#            activation='relu',
#            use_bias=True)(x)
# x = MaxPooling1D(pool_size=2)(x)  # (?, 27, 10), (?, 24, 10)
# x = Conv1D(filters=num_filters,
#            kernel_size=filter_sizes[0],  # 3 means 3 words
#            padding='valid',  # valid means no padding
#            strides=1,
#            activation='relu',
#            use_bias=True)(x)
# x = MaxPooling1D(pool_size=2)(x)  # (?, 27, 10), (?, 24, 10)
# # x = TimeDistributed(Flatten()(x))
# # x = LSTM(128, return_sequences=True, activation='relu')(x)
# x = LSTM(128, dropout=0.2, recurrent_dropout=0.2,
#         return_sequences=False, activation='relu')(x)
conv_blocks = []
for fz in filter_sizes:
    conv = Conv1D(filters=num_filters,
                  kernel_size=fz,
                  padding='valid',
                  strides=1,
                  activation='relu',
                  use_bias=True)(embedded) # (None, 362, 10)
    conv = MaxPooling1D(pool_size=2)(conv) # (None, 181, 10)
    conv = LSTM(128, return_sequences=True, activation='relu')(conv)
    conv = Flatten()(conv) # (None, 1810)
    conv_blocks.append(conv)

concat1max = Concatenate()(conv_blocks)  # (?, )
concat1max = Dropout(dropout_prob)(concat1max) # 0.5

x = Dense(hidden_dims, activation='relu')(x) # (?, 50)
output_layer = Dense(2, activation='sigmoid')(x) # (?, 2)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# Train model with Early Stopping
earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
csv_logger = CSVLogger('log.csv', append=False, separator=';')

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=60, callbacks=[earlystopper, csv_logger],
          validation_split=0.1, shuffle=True, verbose=1)

# Evaluate
score = model.evaluate(x_test, y_test)
print('test_loss, test_acc: ', score)

# Write result to txt
result = 'test_loss, test_acc: {0}'.format(score)
f = open('result.txt', 'w')
f.write(result)
f.close()