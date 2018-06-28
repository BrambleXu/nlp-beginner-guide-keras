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
    model_name = '{:d}'