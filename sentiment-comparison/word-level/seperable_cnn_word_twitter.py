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
from keras.layers import Conv1D, MaxPooling1D, Dropout, SeparableConv1D
from keras.models import Model

#==================Preprocess===================

# Load data
csv = '../data/twitter/clean_tweet.csv'
df = pd.read_csv(csv, index_col=0)
print(df.head())

# Delete Null row
df = df.dropna()
print(df.target.value_counts())

# See word level length
length = [len(sent) for sent in df['text']]
print('The max length is: ', max(length))
print('The min length is: ', min(length))
print('The average length is: ', sum(length)/len(length))

df['target'] = df['target'].map({0: 0, 4: 1})

x_text = df['text'].values
y = df['target'].values
y = to_categorical(y)

# Tokenizer
tk = Tokenizer(num_words=None, oov_token='UNK')
tk.fit_on_texts(x_text)
# Convert string to index
sequences = tk.texts_to_sequences(x_text)
# Padding
sequences_pad = pad_sequences(sequences, maxlen=137, padding='post')

# Shuffle data
np.random.seed(42)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = sequences_pad[shuffle_indices]
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

#=====================Prepare word embedding vector=============
# Load glove vector
glove_path = 'glove.6B'
# read glove to embedding
embeddings_index = {}
f = open(os.path.join(glove_path, 'glove.6B.50d.txt'), 'rb')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

# Add random vector to 'UNK'
mu, sigma = 0, 0.1 # mean and standard deviation
embeddings_index['UNK'] = np.random.normal(mu, sigma, 50)

# Use embeddings_index to build our embedding_matrix
embedding_dim = 50
embedding_matrix = np.zeros((len(tk.word_index)+1, embedding_dim)) # fist row represent padding with 0
for word, i in tk.word_index.items():  # tk.word_index contain 18765 words
    embedding_vector = embeddings_index.get(word) # if not find in the dict, return None
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else: # For the unknown word in tk.word_index, assign UNK vector
        embedding_vector = embeddings_index.get('UNK')
        embedding_matrix[i] = embedding_vector

#===================CNN Model===================
# Model Hyperparameters
embedding_dim = 50
filter_sizes = (3, 8)
num_filters = 100
vocab_size = len(tk.word_index)
dropout_prob = 0.5
hidden_dims = 50
batch_size = 32
num_epochs = 10
sequence_length = 137


# Embedding layer Initialization
embedding_layer = Embedding(vocab_size+1,
                            embedding_dim,
                            input_length=sequence_length,
                            weights=[embedding_matrix])

# Create model
# Input
input_shape = (sequence_length,)
input_layer = Input(shape=input_shape, name='input_layer')  # (?, 56)

# Embedding
embedded = embedding_layer(input_layer) # (batch_size, sequence_length, output_dim)=(?, 56, 50),

# CNN, iterate filter_size
conv_blocks = []
for fz in filter_sizes:
    conv = SeparableConv1D(filters=num_filters,
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
history = model.fit(x_train, y_train,batch_size=batch_size, epochs=num_epochs, callbacks=[earlystopper, tensorboard],
          validation_split=0.1, shuffle=True, verbose=1)

# Evaluate
score = model.evaluate(x_test, y_test)
print('test_loss, test_acc: ', score)

# Write result to txt
result = 'test_loss, test_acc: {0}'.format(score)
f = open('result.txt', 'wb')
f.write(result)
f.close()

with open('train_history.pickle', 'wb') as handle:
    pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
