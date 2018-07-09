In this repo, I will implement some NLP models for the nlp beginner learner. In each project folder contains a notebook floder to show the output in detail. I think this would help the beginner to understand what happens in the model.

I will list what resource used for each model implementation. All project based on Python3.6 and Keras2.1.6 with TensorFlow1.8 backend. 


## Deep Models for NLP beginners

You can find detail instruction in each project. Here I will list what you can learn in each project.

#### [Cnn-text-classification](https://github.com/BrambleXu/nlp-beginner-guide-keras/tree/master/cnn-text-classification)

- A keras implementation of CNN-non-static for text classification
- train a word2vec model
- Load pre-trained word2vec weights
- Save and load a keras model
- Use tensorboard to visualize your neural networks

#### [word_embedding](https://github.com/BrambleXu/nlp-beginner-guide-keras/tree/master/word_embedding)

- Load pre-trained GloVe weights
- How Keras deal with OOV token


#### [char-level-cnn](https://github.com/BrambleXu/nlp-beginner-guide-keras/tree/master/char-level-cnn)


What you can learn in this implementation:
- Using Keras function to preprocess char level text, [article](https://medium.com/@zhuixiyou/how-to-preprocess-character-level-text-with-keras-349065121089), [notebook](https://github.com/BrambleXu/nlp-beginner-guide-keras/blob/f2fdfdd20e73ae16208b3ac63962a769fac51065/char-level-cnn/notebooks/char-level-text-preprocess-with-keras-summary.ipynb)
- Constructing the char-cnn-zhang model, [article](https://medium.com/@zhuixiyou/character-level-cnn-with-keras-50391c3adf33), [notebook](https://github.com/BrambleXu/nlp-beginner-guide-keras/blob/f2fdfdd20e73ae16208b3ac63962a769fac51065/char-level-cnn/notebooks/char-cnn-zhang-with-keras-pipeline.ipynb)


## Requirements

- Python 3.6
- Keras 2.1.6
- Tensorflow 1.8
- tensorboard 1.8
- scikit-learn 0.19.1
- numpy 1.14.3
- h5py 2.8

