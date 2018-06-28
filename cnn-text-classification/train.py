#======================== import module order as three level========================
import numpy as np
import data_helpers






#======================== command line flags parser  ========================








#======================== preprocess data ========================
#

#TODO: After complete all training, use argparse to store the params.
positive_data_file = "./data/rt-polaritydata/rt-polarity.pos"
negtive_data_file = "./data/rt-polaritydata/rt-polarity.neg"

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(positive_data_file, negtive_data_file)

# Pad sentence
print("Padding sentences...")
x_text = data_helpers.pad_sentences(x_text)
print("The sequence length is: ", len(x_text[0]))

# Build vocabulary
vocabulary, vocabulary_inv = data_helpers.build_vocab(x_text)

# Represent sentence with word index, using word index to represent a sentence
x = data_helpers.build_index_sentence(x_text, vocabulary)
y = y.argmax(axis=1) # y: [1, 1, 1, ...., 0, 0, 0]. 1 for positive, 0 for negative

# Shuffle data
np.random.seed(42)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train and test
# TODO: training_rate could be set by user as a parameter
training_rate = 0.9
train_len = int(len(y) * training_rate)
x_train = x_shuffled[:train_len]
y_train = y_shuffled[:train_len]
x_test = x_shuffled[train_len:]
y_test = y_shuffled[train_len:]


#========================end preprocess data ========================
# For this point, we have x_train, y_train, x_test,
# y_test, vocabulary_inv for later use

# Output shape
print('x_train shape: ', x_train.shape)
print('x_test shape:', x_test.shape)
print('Vocabulary Size: {:d}'.format(len(vocabulary_inv)))

#





