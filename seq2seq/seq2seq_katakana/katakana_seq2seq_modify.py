from __future__ import print_function

import os
import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


data = pd.read_csv('../data/eng-katakana.csv', header=None, names=['eng', 'katakana'])
data = data.sample(frac=1, random_state=0)

data_input = [s.lower() for s in data['eng']]
data_output = [s for s in data['katakana']]

print(data_input[0:3])
print(data_output[0:3])


# Split train and test
training_rate = 0.7
train_len = int(len(data) * training_rate)
training_input = data_input[:train_len]
training_output = data_output[:train_len]
validation_input = data_input[train_len:]
validation_output = data_output[train_len:]

print(len(training_input))
print(len(validation_input))

START_CHAR_CODE = 1

def encode_characters(titles):
    count = 2
    encoding = {}
    decoding = {1: 'START'}
    for c in set([c for title in titles for c in title]):
        encoding[c] = count
        decoding[count] = c
        count += 1
    return encoding, decoding, count


input_encoding, input_decoding, input_dict_size = encode_characters(data_input)
output_encoding, output_decoding, output_dict_size = encode_characters(data_output)


print('English character dict size:', input_dict_size)
print('Katakana character dict size:', output_dict_size)

# print(input_encoding)
# print(input_decoding)

def transform(encoding, data, vector_size):
    transformed_data = np.zeros(shape=(len(data), vector_size))
    for i in range(len(data)):
        for j in range(min(len(data[i]), vector_size)):
            transformed_data[i][j] = encoding[data[i][j]]
    return transformed_data

INPUT_LENGTH = 20
OUTPUT_LENGTH = 20

encoded_training_input = transform(input_encoding, training_input, vector_size=INPUT_LENGTH)
encoded_training_output = transform(output_encoding, training_output, vector_size=OUTPUT_LENGTH)
encoded_validation_input = transform(input_encoding, validation_input, vector_size=INPUT_LENGTH)
encoded_validation_output = transform(output_encoding, validation_output, vector_size=OUTPUT_LENGTH)


# Encoder Input
training_encoder_input = encoded_training_input

# Decoder Input (need padding py START_CHAR_CODE)
training_decoder_input = np.zeros_like(encoded_training_output)
training_decoder_input[:, 1:] = encoded_training_output[:,:-1] # offset one timestpe
training_decoder_input[:, 0] = START_CHAR_CODE # first timestep is 1, means START

# Decoder Output (one-hot encode)
training_decoder_output = np.eye(output_dict_size)[encoded_training_output.astype('int')]

print('encoder input', training_encoder_input[:1])
print('decoder input', training_decoder_input[:1])
print('decoder output', training_decoder_output[:1].argmax(axis=2))
print('decoder output (one-hot)', training_decoder_output[:1])

# print('input', encoded_training_input)
# print('output', encoded_training_output)

# Model
encoder_input = Input(shape=(INPUT_LENGTH,))
decoder_input = Input(shape=(OUTPUT_LENGTH,))

encoder_input = Input(shape=(INPUT_LENGTH,))
decoder_input = Input(shape=(OUTPUT_LENGTH,))

# Encoder
encoder = Embedding(input_dict_size, 64, input_length=INPUT_LENGTH, mask_zero=True)(encoder_input)
encoder = LSTM(64)(encoder)

decoder = Embedding(output_dict_size, 64, input_length=OUTPUT_LENGTH, mask_zero=True)(decoder_input)
decoder = LSTM(64, return_sequences=True)(decoder, initial_state=[encoder, encoder])
decoder = TimeDistributed(Dense(output_dict_size, activation="softmax"))(decoder)

model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder])
model.compile(optimizer='adam', loss='binary_crossentropy')

earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

model.fit(x=[training_encoder_input, training_decoder_input], y=[training_decoder_output],
      validation_split=0.11,
      verbose=1,
      batch_size=64,
      epochs=20,
      callbacks=[earlystopper])



def generate(text):
    encoder_input = transform(input_encoding, [text.lower()], 20)
    decoder_input = np.zeros(shape=(len(encoder_input), OUTPUT_LENGTH))
    decoder_input[:,0] = START_CHAR_CODE
    for i in range(1, OUTPUT_LENGTH):
        output = model.predict([encoder_input, decoder_input]).argmax(axis=2)
        decoder_input[:,i] = output[:,i]
    return decoder_input[:,1:]

def decode(decoding, sequence):
    text = ''
    for i in sequence:
        if i == 0:
            break
        text += output_decoding[i]
    return text

def to_katakana(text):
    decoder_output = generate(text)
    return decode(output_decoding, decoder_output[0])


