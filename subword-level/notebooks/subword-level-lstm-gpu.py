import sys, os
sys.path.append(os.pardir)
from data_helpers import BPE

import numpy as np
import pandas as pd
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session  
config = tf.ConfigProto()  
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU  
sess = tf.Session(config=config)  
set_session(sess)  # set this TensorFlow session as the default session for Keras.


bpe = BPE("../pre-trained-model/en.wiki.bpe.op25000.vocab")
# Build vocab, {token: index}
vocab = {}
for i, token in enumerate(bpe.words):
    vocab[token] = i + 1
    
# Load preprocessed data from npz file
dataset = np.load('preprocessed_dataset.npz')
x_train = dataset['x_train']
y_train = dataset['y_train']
x_test = dataset['x_test']
y_test = dataset['y_test']

from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format("../pre-trained-model/en.wiki.bpe.op25000.d50.w2v.bin", binary=True)

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
        
from keras.layers import Embedding

# parameter 
input_size = 1014
embedding_size = 50

num_of_classes = 4
dropout_p = 0.5
optimizer = 'adam'
loss = 'categorical_crossentropy'

embedding_layer = Embedding(len(vocab)+1,
                            embedding_size,
                            weights=[embedding_weights],
                            input_length=input_size,
                            trainable=False)


from keras.layers import Input, Embedding, Dense, Flatten
from keras.layers import LSTM, Dropout
from keras.models import Model


inputs = Input(shape=(input_size,))
embedded_sequence = embedding_layer(inputs)
x = LSTM(256, return_sequences=True, activation='relu')(embedded_sequence)
x = LSTM(256, return_sequences=False, activation='relu')(x) # Change True to False
# x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(dropout_p)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(dropout_p)(x)
prediction = Dense(num_of_classes, activation='softmax')(x)


model = Model(inputs=inputs, outputs=prediction)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()



# training
model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          batch_size=128,
          epochs=1,  # The training time for whole dataset is pretty long, so here we only run 1 epoch
          shuffle=True,
          verbose=1)