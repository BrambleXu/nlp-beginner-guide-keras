import os
import numpy as np
import data_helpers
from data_helpers import TrainValTensorBoard
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Input, Embedding, Activation, Flatten, Dense, Concatenate
from keras.layers import Conv1D, MaxPooling1D, Dropout, LSTM
from keras.models import Model

#==================Preprocess===================

# Load data
positive_data_file = "../data/rt-polaritydata/rt-polarity.pos"
negtive_data_file = "../data/rt-polaritydata/rt-polarity.neg"

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(positive_data_file, negtive_data_file)

# See word level length
length = [len(sent) for sent in x_text]
print('The max length is: ', max(length))
print('The min length is: ', min(length))
print('The average length is: ', sum(length)/len(length))

# Tokenizer
tk = Tokenizer(num_words=None, oov_token='UNK')
tk.fit_on_texts(x_text)
# Convert string to index
sequences = tk.texts_to_sequences(x_text)
# Padding
sequences_pad = pad_sequences(sequences, maxlen=56, padding='post')
print("The whole data size is: ", sequences_pad.shape)

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
f = open(os.path.join(glove_path, 'glove.6B.50d.txt'))
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
    if word in embeddings_index:
        embedding_matrix[i] = embeddings_index.get(word)
    else: # For the unknown word in tk.word_index, assign UNK vector
        embedding_vector = embeddings_index.get('UNK')
        embedding_matrix[i] = embedding_vector

#===================LSTM Model===================
# Model Hyperparameters
embedding_dim = 50
vocab_size = len(tk.word_index)
dropout_prob = 0.5
hidden_dims = 50
batch_size = 32
num_epochs = 10
sequence_length = 56


# Embedding layer Initialization
embedding_layer = Embedding(vocab_size+1,
                            embedding_dim,
                            input_length=sequence_length,
                            weights=[embedding_matrix],
                            mask_zero=True)

# Create model
# Input
inputs = Input(shape=(sequence_length,))
# Embedding
embedded_sequence = embedding_layer(inputs)
x = LSTM(128, return_sequences=True, activation='relu')(embedded_sequence)
x = LSTM(128, return_sequences=False, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dropout(dropout_prob)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(dropout_prob)(x)
prediction = Dense(2, activation='sigmoid')(x)


model = Model(inputs=inputs, outputs=prediction)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

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