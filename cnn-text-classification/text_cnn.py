from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, GlobalMaxPooling1D, Conv1D, Embedding
from keras.layers.merge import Concatenate
from keras import regularizers
import numpy as np

np.random.seed(0)


def create_base_model(vocab_size, embedding_dim, filter_sizes, num_filters, dropout_prob, hidden_dims, sequence_length):
    # Input
    input_shape = (sequence_length,)
    input_layer = Input(shape=input_shape, name='input_layer')  # (?, 56)

    # Embedding
    embedded = Embedding(input_dim=vocab_size,
                         output_dim=embedding_dim,
                         input_length=sequence_length,
                         name='embedding_layer')(input_layer) # (batch_size, sequence_length, output_dim)=(?, 56, 50),

    # CNN, iterate filter_size
    conv_blocks = []
    for fz in filter_sizes:
        conv = Conv1D(filters=num_filters,
                      kernel_size=fz,
                      padding='valid',  # valid means no padding
                      strides=1,
                      activation='relu',
                      use_bias=True)(embedded)
        conv = GlobalMaxPooling1D()(conv) # 1-Max pooling
        conv_blocks.append(conv)

    concat1max = Concatenate()(conv_blocks)
    concat1max = Dropout(dropout_prob[1])(concat1max) # 0.8
    output_layer = Dense(hidden_dims, activation='relu',
                         kernel_regularizer=regularizers.l2(0.01),
                         bias_regularizer=regularizers.l1(0.01))(concat1max) # (?, 50)
    output_layer = Dense(1, activation='sigmoid')(output_layer) # (?, 1)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_model(vocab_size, embedding_dim, filter_sizes, num_filters, dropout_prob, hidden_dims, sequence_length):
    # Input
    input_shape = (sequence_length,)
    input_layer = Input(shape=input_shape, name='input_layer')  # (?, 56)

    # Embedding
    embedded = Embedding(input_dim=vocab_size,
                         output_dim=embedding_dim,
                         input_length=sequence_length,
                         name='embedding_layer')(input_layer) # (batch_size, sequence_length, output_dim)=(?, 56, 50),

    # CNN, iterate filter_size
    conv_blocks = []
    for fz in filter_sizes:
        conv = Conv1D(filters=num_filters,
                      kernel_size=fz,  # 3 means 3 words
                      padding='valid',  # valid means no padding
                      strides=1,  # see explnation above
                      activation='relu',
                      use_bias=True)(embedded)
        conv = MaxPooling1D(pool_size=2)(conv) # (?, 27, 10), (?, 24, 10)
        conv = Flatten()(conv) # (?, 270), (?, 240)
        conv_blocks.append(conv) # [(?, 270), (?, 240)]

    concat1max = Concatenate()(conv_blocks)  # (?, 510)
    concat1max = Dropout(dropout_prob[1])(concat1max) # 0.8
    output_layer = Dense(hidden_dims, activation='relu',
                         kernel_regularizer=regularizers.l2(0.01),
                         bias_regularizer=regularizers.l1(0.01))(concat1max) # (?, 50)
    output_layer = Dense(1, activation='sigmoid')(output_layer) # (?, 1)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model