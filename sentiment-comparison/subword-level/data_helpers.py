import re
import os
from keras.callbacks import TensorBoard
import tensorflow as tf
from math import log


class BPE(object):

    def __init__(self, vocab_file):
        with open(vocab_file, encoding="utf8") as f:
            self.words = [l.split()[0] for l in f]
            log_len = log(len(self.words))
            self.wordcost = {
                k: log((i+1) * log_len)
                for i, k in enumerate(self.words)}
            self.maxword = max(len(x) for x in self.words)

    def encode(self, s):
        """Uses dynamic programming to infer the location of spaces in a string
        without spaces."""

        s = s.replace(" ", "▁")

        # Find the best match for the i first characters, assuming cost has
        # been built for the i-1 first characters.
        # Returns a pair (match_cost, match_length).
        def best_match(i):
            candidates = enumerate(reversed(cost[max(0, i - self.maxword):i]))
            return min(
                (c + self.wordcost.get(s[i-k-1:i], 9e999), k+1)
                for k, c in candidates)

        # Build the cost array.
        cost = [0]
        for i in range(1, len(s) + 1):
            c, k = best_match(i)
            cost.append(c)

        # Backtrack to recover the minimal-cost string.
        out = []
        i = len(s)
        while i > 0:
            c, k = best_match(i)
            assert c == cost[i]
            out.append(s[i-k:i])

            i -= k

        return " ".join(reversed(out))


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
    # x_text = [sen.split(" ") for sen in x_text]

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

