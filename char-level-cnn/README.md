
This code is a keras implementation Character-level Convolutional Neural Networks for text classification on AG's News Topic Classification Dataset.
- Xiang Zhang, Junbo Zhao, Yann LeCun. [Character-level Convolutional Networks for Text Classification](http://arxiv.org/abs/1509.01626). NIPS 2015


What you can learn in this implementation:
- Using Keras function to preprocess char level text, [article](https://medium.com/@zhuixiyou/how-to-preprocess-character-level-text-with-keras-349065121089), [notebook](https://github.com/BrambleXu/nlp-beginner-guide-keras/blob/f2fdfdd20e73ae16208b3ac63962a769fac51065/char-level-cnn/notebooks/char-level-text-preprocess-with-keras-summary.ipynb)
- Constructing the char-cnn-zhang model，[notebook](https://github.com/BrambleXu/nlp-beginner-guide-keras/blob/f2fdfdd20e73ae16208b3ac63962a769fac51065/char-level-cnn/notebooks/char-cnn-zhang-with-keras-pipeline.ipynb)

The implementation file is in `char_cnn.py`, you can type below command to run the training. I write every step in this file, it would be easy for reading.

```
python char_cnn.py
```


If you run `python main.py`, this will run the model in from work, [CharCnn_Keras](https://github.com/mhjabreel/CharCnn_Keras).


## References

Here I will list some useful resource when I learn to implement the paper.

**github repo**
- [CharCnn_Keras](https://github.com/mhjabreel/CharCnn_Keras)
- [char-CNN-text-classification-tensorflow](https://github.com/Irvinglove/char-CNN-text-classification-tensorflow)

