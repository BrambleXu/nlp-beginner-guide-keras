#======================== import module order as three level========================
import os
from os.path import join, exists, split
import numpy as np
import data_helpers
from word2vec import train_word2vec
from text_cnn import create_model
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from keras.callbacks import TensorBoard
import tensorflow as tf

#========plot train and validation scalars in a same figure=======
class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()



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

# Word2Vec parameters (see train_word2vec)
embedding_dim = 50
min_word_count = 1
context = 10

#Prepare embedding layer weights for not-static model
embedding_weights = train_word2vec(np.vstack((x_train, x_test)), vocabulary_inv, num_features=embedding_dim,
                                   min_word_count=min_word_count, context=context)


#===========================create model====================
# Model Hyperparameters
embedding_dim = 50
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50
vocab_size = len(vocabulary_inv)
batch_size = 32
num_epochs = 10

# Create model
sequence_length = x_test.shape[1]  # 56
model = create_model(vocab_size, embedding_dim, filter_sizes, num_filters, dropout_prob, hidden_dims, sequence_length)

# Initialize weights with word2vec
weights = np.array([v for v in embedding_weights.values()])
print("Initializing embedding layer with word2vec weights, shape", weights.shape)
embedding_layer = model.get_layer("embedding_layer")
embedding_layer.set_weights([weights])

# Train model with Early Stopping
earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
tensorboard = TrainValTensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=True)
model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, callbacks=[earlystopper, tensorboard],
          validation_split=0.1, verbose=2)

# Evaluate
score = model.evaluate(x_test, y_test)
print(score)

#=============use sklearn to evaluate=============

from sklearn.metrics import accuracy_score

# Prediciton
prediction = model.predict(x_test)
prediction = prediction.flatten()
prediction = np.where(prediction > 0.5, 1, 0)
score = accuracy_score(y_test, prediction)
print(score)

#================Save and Load=================

# Save model
model_dir = 'models'
model_name = 'non_static_cnn.json'
model_name = join(model_dir, model_name)
model_weights = 'non_static_cnn.h5'
model_weights = join(model_dir, model_weights)

if not exists(model_dir):
    os.mkdir(model_dir)

if not exists(model_name):
    print('Saving non static cnn model and its in \'%s\'' % split(model_name)[0])
    # Serialize model to JSON
    model_json = model.to_json()
    with open(model_name, 'w') as json_file:
        json_file.write(model_json)
    # Serialize weights to HDF5
    model.save_weights(model_weights)

# Load json and create model
with open(model_name, 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
# Load weights into new model
loaded_model.load_weights(model_weights)
print('Loaded existing model from \'%s\'' % model_name)


from sklearn.metrics import accuracy_score

# Prediciton
prediction = loaded_model.predict(x_test)
prediction = prediction.flatten()
prediction = np.where(prediction > 0.5, 1, 0)
score = accuracy_score(y_test, prediction)
print(score)

