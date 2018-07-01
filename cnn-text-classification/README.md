
This code is a keras implementation of Kim's [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) paper.
Here we only implement the CNN-non-static version, which fine-tune the embedding layer during training.

What you can learn in this implementation:
- A keras implementation of CNN-non-static for text classification
- train a word2vec model
- load a pre-trained word2vec weights
- Save and load a keras model
- Use tensorboard to visualize your neural networks

## Requirements

- Python 3.6
- Keras 2.1.6
- Tensorflow 1.8
- tensorboard 1.8
- scikit-learn 0.19.1
- numpy 1.14.3
- h5py 2.8

## Two versions

Here I implement one baseline version and modified version. The parameters of baseline version are similar with the setting in Kim's paper.

| Setting        | Modified           | Baseline  |
| ------------- |:-------------:| :-----:|
|  Embedding Dimension  | 50 | 300 |
| Filter Size      | (3, 8)     |   (3, 4, 5) |
| Filter Number | 10 |   100  |
| Pooling | Max Pooling      |   Global Pooling(1-max pooling) |
| Result | |    |
| Training Time(10 epoch) |  636s |  86s  |
| Accuracy |  0.73 |  0.7  |

The reason I create the baseline is mainly to show the implementation part of 1-max pooling.
Because the dimensions in the pooling layer are different between Global Pooling and Max Pooling.
It is very important to have a clear understanding for the dimension changes. You can find it in the `create_base_model()` of `text_cnn.py`.


## Notebooks

- Preprocess
- word2vec
- base model
- some experiments

## Training

If you are working in the terminal, you current directory should be `cnn-text-classification`.

Training the base model:

```
python train_base.py
```


Train the modified model:

```
python train.py
```
After this, you will get new log files in the floder `logs`.

## Visualization

I already save the training logs in floder `logs`, you can run the command blow to see the visualization in tensorboard.
In the terminal, you current directory should be `cnn-text-classification`.
```
tensorboard --logdir logs/
```
Then open the browser and input the `http://localhost:xxxx/`. xxxx is the port, you can find it in the terminal.


## References

Here I will list some useful resource when I learn to implement the paper.


**Paper**
- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)

It better to read both of them. If you feel hard to understand the what is filter and filter maps, here is one article for you:

- [Understanding how Convolutional Neural Network (CNN) perform text classification with word embeddings](http://www.joshuakim.io/understanding-how-convolutional-neural-network-cnn-perform-text-classification-with-word-embeddings/)


**Article**
- [Understanding Convolutional Neural Networks for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)
- [Implementing a CNN for Text Classification in Tensorflow blog post](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)

These two article could give you a overview how to use CNN for NLP and the tensorflow implementation.

**github repo**
- [cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf)
- [CNN-for-Sentence-Classification-in-Keras](https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras)

These two repo are very useful if you are learn to implement the model by yourself.