from keras.layers import Input, Embedding, Activation, Flatten, Dense, Concatenate
from keras.layers import Conv1D, MaxPooling1D, Dropout
from keras.models import Model
import numpy as np

np.random.seed(0)

def create_cnn(filter_sizes, num_filters, vocab_size, embedding_dim, sequence_length, embedding_matrix,
               dropout_prob, hidden_dims):
    # Embedding layer Initialization
    embedding_layer = Embedding(vocab_size + 1,
                                embedding_dim,
                                input_length=sequence_length,
                                weights=[embedding_matrix])

    # Input
    input_shape = (sequence_length,)
    input_layer = Input(shape=input_shape, name='input_layer')  # (?, 56)

    # Embedding
    embedded = embedding_layer(input_layer)  # (batch_size, sequence_length, output_dim)=(?, 56, 50),

    # CNN, iterate filter_size
    conv_blocks = []
    for fz in filter_sizes:
        conv = Conv1D(filters=num_filters,
                      kernel_size=fz,  # 3 means 3 words
                      padding='valid',  # valid means no padding
                      strides=1,
                      activation='relu',
                      use_bias=True)(embedded)
        conv = MaxPooling1D(pool_size=2)(conv)  # (?, 27, 10), (?, 24, 10)
        conv = Flatten()(conv)  # (?, 270), (?, 240)
        conv_blocks.append(conv)  # [(?, 270), (?, 240)]

    concat1max = Concatenate()(conv_blocks)  # (?, 510)
    concat1max = Dropout(dropout_prob)(concat1max)  # 0.5
    output_layer = Dense(hidden_dims, activation='relu')(concat1max)  # (?, 50)
    output_layer = Dense(2, activation='sigmoid')(output_layer)  # (?, 2)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model