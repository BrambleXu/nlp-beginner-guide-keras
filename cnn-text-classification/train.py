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
# print(x_test[:3])










