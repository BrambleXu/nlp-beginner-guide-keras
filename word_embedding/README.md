What you can learn in this implementation:
- How to read Glove word vector for embedding layer
- How to use Tokenizer to handle the OOV words

## Notebooks

In the notebook, you can check the output in each phase. I hope this could help you better understand what happend.

- [keras word_embedding_tutorial](https://github.com/BrambleXu/nlp-beginner-guide-keras/blob/ba5af260f9f999af4235cd03b36c5217db8a0cf9/word_embedding/keras_word_embedding_tutorial.ipynb)
- [Keras-tokenizer-oov](https://github.com/BrambleXu/nlp-beginner-guide-keras/blob/ba5af260f9f999af4235cd03b36c5217db8a0cf9/word_embedding/Keras-tokenizer-oov.ipynb)

## Dataset

Data is in the `20_newsgroup` floder. You can see the data detail and download it from [here](http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html).

Glove data we choose the `glove.6B.50d`, which means each word have 40 demensions. You can download it from [here](https://nlp.stanford.edu/projects/glove/), click `glove.6B.zip` to start download.

The directory structure is like this:
```
word_embedding/
    20_newsgroup/
        alt.atheism/
        comp.windows.x/
        .....

    glove.6B/
        glove.6B.50d.txt

    keras_word_embedding_tutorial.ipynb
```

## Requirements

- Python 3.6
- Keras 2.1.6
- Tensorflow 1.8
- tensorboard 1.8
- scikit-learn 0.19.1
- numpy 1.14.3
- h5py 2.8


<!--## References-->

<!--**Article**-->
<!--- [Understanding Convolutional Neural Networks for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)-->
<!--- [Implementing a CNN for Text Classification in Tensorflow blog post](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)-->