"""
In this file: I will implement some data preprocess function

"""

import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negtive_data_file):
    """
    Load data from files, split data into words and generate labels
    Input: the positive data file path and negative data file path
    Output:
        x_text: list of words for sentences. e.g [['i', 'am', is'], ['word', 'is', 'too', 'long'], ...,]
        y: For each sentence, using `[neg, pos]` to represent the lables.
           - If we have a positive label, we represent it as `[0, 1]`
           - If we have a negative label, we represent it as `[1, 0]`
    """

    # Load data from files
    positive_examples = list(open(positive_data_file, 'r', encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negtive_data_file, 'r', encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]

    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sen) for sen in x_text]
    x_text = [sen.split(" ") for sen in x_text]

    # Generate labels
    positive_lables = [[0, 1] for _ in positive_examples]
    negative_lables = [[1, 0] for _ in negative_examples]
    y = np.concatenate((positive_lables, negative_lables), 0)
    return x_text, y


def pad_sentences(sentences, padding_word='<PAD/>'):
    """
    :param sentences: sentences as list of words,  [['i', 'am', is'], ['word', 'is', 'too', 'long'], ...,]
    :return: pad sentence to longest length, [['i', 'am', is', '<PAD>', '<PAD>'], ['word', 'is', 'too', 'long', '<PAD>'], ...,]
    """
    sequence_length = max(len(sen) for sen in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    :param sentences:  sentences after padding
    :return:
        vocabulary: a dict object, key is word and value is index. e.g. {'i': 0, 'am':1}
        vocabulary_inv: a dict object, the inverse of vocabulary. e.g. {0: 'i', 1:'am'}
    """
    # Count words
    word_counts = Counter(itertools.chain(*sentences))
    # Sort the word as frequency order
    vocabulay_inv = [x[0] for x in word_counts.most_common()]
    # Build vocabulary, word: index
    vocabulay = {word: i for i, word in enumerate(vocabulay_inv)}
    # Build inverse vocabulary, index: word
    vocabulay_inv = {value: key for key, value in vocabulay.items()}

    return [vocabulay, vocabulay_inv]

def build_index_sentence(sentences, vocabulary):
    # x = []
    # for sen in sentences:
    #     one_sen = []
    #     for word in sen:
    #         one_sen.append(vocabulary[word])
    #     x.append(one_sen)
    # return np.array(x)

    # write above code as one line
    x = np.array([[vocabulary[word] for word in sen] for sen in sentences])
    return x

