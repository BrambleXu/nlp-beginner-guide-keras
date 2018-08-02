import os
import numpy as np
import data_helpers
from data_helpers import TrainValTensorBoard
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from create_model import create_cnn

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
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else: # For the unknown word in tk.word_index, assign UNK vector
        embedding_vector = embeddings_index.get('UNK')
        embedding_matrix[i] = embedding_vector

#===================CNN Model===================
# Model Hyperparameters
embedding_dim = 50
filter_sizes = (3, 8)
num_filters = 10
vocab_size = len(tk.word_index)
dropout_prob = 0.5
hidden_dims = 50
sequence_length = 56
# embedding_matrix is also passed to create_model

# Training Hyperparameters
num_epochs = 10
batch_size = 32

# Create model
model = create_cnn(filter_sizes, num_filters, vocab_size, embedding_dim, sequence_length,
                   embedding_matrix, dropout_prob, hidden_dims)

print(model.summary())


# Train model with Early Stopping and save best model
earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
checkpoint = ModelCheckpoint( filepath='my_model.h5', monitor='val_loss', save_best_only=True)

model.fit(x_train, y_train,batch_size=batch_size, epochs=3, callbacks=[earlystopper, checkpoint],
          validation_split=0.1, shuffle=True, verbose=1)

# Load best model and evaluate
model = create_cnn(filter_sizes, num_filters, vocab_size, embedding_dim, sequence_length,
                   embedding_matrix, dropout_prob, hidden_dims)
model.load_weights('my_model.h5')
print('Load best model: {0}'.format('my_model.h5'))
score = model.evaluate(x_test, y_test)
print('test_loss, test_acc: ', score)