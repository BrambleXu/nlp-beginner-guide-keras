from gensim.models import word2vec
from os.path import join, exists, split
import os
import numpy as np

def train_word2vec(sentence_matrix, vocabulary_inv,
                   num_features=300, min_word_count=1, context=10):
    """
    Trains, saves, loads Word2Vec model
    Returns initial weights for embedding layer.

    inputs:
    sentence_matrix # int matrix: num_sentences x max_sentence_len
    vocabulary_inv  # dict {int: str}
    num_features    # Word vector dimensionality
    min_word_count  # Minimum word count
    context         # Context window size
    """
    model_dir = 'models'
    model_name = '{:d}feature_{:d}minwords_{:d}context'.format(num_features, min_word_count, context)
    model_name = join(model_dir, model_name)

    if exists(model_name):
        embedding_model = word2vec.Word2Vec.load(model_name)
        print('Load existing Word2Vec model \'%s\'' % split(model_name)[-1])
    else:
        # Set values for embedding parameter
        num_workers = 2 # Number of threads to run parallel
        downsampling = 1e-3 # Threshold to downsample frequent words

        # Get the sentences list of words, instead of index
        sentences = [[vocabulary_inv[index] for index in sen] for sen in sentence_matrix]
        # Initialize and train the model
        embedding_model = word2vec.Word2Vec(sentences, size=num_features,
                                            window=context, min_count=min_word_count,
                                            sample=downsampling, workers=num_workers)

        # Save the model for later use. We can load it using Word2Vec.load()
        if not exists(model_dir):
            os.mkdir(model_dir)
        print('Saving Word2Vec model \'%s\'' % split(model_name)[-1])
        embedding_model.save(model_name)

    # Add unknown words
    embedding_weights = {key: embedding_model.wv[word] if word in embedding_model.wv
                            else np.random.uniform(-0.25, 0.25, embedding_model.vector_size)
                         for key, word in vocabulary_inv.items()}
    return embedding_weights
