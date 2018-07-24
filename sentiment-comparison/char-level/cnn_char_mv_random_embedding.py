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

#==================Preprocess===================

# Load data
positive_data_file = "../data/rt-polaritydata/rt-polarity.pos"
negtive_data_file = "../data/rt-polaritydata/rt-polarity.neg"

print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(positive_data_file, negtive_data_file)
# =======================Convert string to index================
# Tokenizer
tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
tk.fit_on_texts(x_text)
# If we already have a character list, then replace the tk.word_index
# If not, just skip below part

# -----------------------Skip part start--------------------------
# construct a new vocabulary
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789 ,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
char_dict = {}
for i, char in enumerate(alphabet):
    char_dict[char] = i + 1

# Use char_dict to replace the tk.word_index
tk.word_index = char_dict.copy()
# Add 'UNK' to the vocabulary
tk.word_index[tk.oov_token] = max(char_dict.values()) + 1
# -----------------------Skip part end----------------------------

# Convert string to index
sequences = tk.texts_to_sequences(x_text)

# See char level length
length = [len(sent) for sent in sequences]
print('The max length is: ', max(length)) # 266
print('The min length is: ', min(length))
print('The average length is: ', sum(length)/len(length))

# Padding
sequences_pad = pad_sequences(sequences, maxlen=266, padding='post')
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
vocab_size = len(tk.word_index) # 70 (69 char, 1 UNK)

# # Embedding weights
embedding_dim = 70
# zero_vector = np.zeros((1, embedding_dim)) # Represent pad vector
# embedding_weights = np.concatenate((zero_vector, np.identity(vocab_size)), axis=0) # (71, 70)

from keras.layers import Embedding
input_size = 266

# Embedding layer Initialization
embedding_layer = Embedding(vocab_size + 1,
                            embedding_dim,
                            input_length=input_size)

#===================CNN Model===================
# Model Hyperparameters
embedding_dim = 70
filter_sizes = (3, 8)
num_filters = 10
vocab_size = len(tk.word_index)
dropout_prob = 0.5
hidden_dims = 50
batch_size = 32
num_epochs = 10
sequence_length = 266


# Create model
# Input
input_shape = (sequence_length,)
input_layer = Input(shape=input_shape, name='input_layer')  # (?, 56)

# Embedding
embedded = embedding_layer(input_layer) # (batch_size, sequence_length, output_dim)=(?, 56, 50),

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