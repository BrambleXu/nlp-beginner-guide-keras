
# Survey of Word/Character/Subword level for Sentiment Analysis

In the three projects we have done before, we take three approaches for the represent the text.
- Word level for [Cnn-text-classification](https://github.com/BrambleXu/nlp-beginner-guide-keras/tree/master/cnn-text-classification)(word2vec), [word_embedding](https://github.com/BrambleXu/nlp-beginner-guide-keras/tree/master/word_embedding)(glove)
- Character level for [char-level-cnn](https://github.com/BrambleXu/nlp-beginner-guide-keras/tree/master/char-level-cnn)
- Subword level for [subword-level](https://github.com/BrambleXu/nlp-beginner-guide-keras/tree/master/subword-level)

After do some research in the ACL2018, we found that most works still choose word level for the sentiment analysis. Dose this means word representation is better character/subword representation?
In order to analyze the performance of these representation for the sentiment analysis task, we do some experiments for the the comparison.

Here we choose two datasets, the sentence polarity dataset and twitter dataset.


## Sentiment Analysis in Movie Review


We choose the sentence polarity dataset as the dataset, and it is a binary classification problem.
You can find more information about the dataset [here](https://github.com/abromberg/sentiment_analysis/blob/master/polarityData/rt-polaritydata.README.1.0.txt).
The total number of samples is 10662.

We take 8635 as training data, 960 as validation data, and 1067 as test data.


### Experiment results (Word/Character/Subword & CNN/LSTM)

**Word level CNN**

```
Train on 8635 samples, validate on 960 samples
Epoch 1/10
 - 3s - loss: 0.6775 - acc: 0.5709 - val_loss: 0.6175 - val_acc: 0.6661
Epoch 2/10
 - 3s - loss: 0.5638 - acc: 0.7108 - val_loss: 0.5505 - val_acc: 0.7068
Epoch 3/10
 - 3s - loss: 0.4616 - acc: 0.7808 - val_loss: 0.5008 - val_acc: 0.7396
Epoch 4/10
 - 3s - loss: 0.3714 - acc: 0.8360 - val_loss: 0.5000 - val_acc: 0.7552
Epoch 5/10
 - 3s - loss: 0.2992 - acc: 0.8748 - val_loss: 0.5381 - val_acc: 0.7719
Epoch 6/10
 - 3s - loss: 0.2247 - acc: 0.9111 - val_loss: 0.5335 - val_acc: 0.7729
Epoch 7/10
 - 3s - loss: 0.1757 - acc: 0.9302 - val_loss: 0.6405 - val_acc: 0.7630
Epoch 8/10
 - 3s - loss: 0.1344 - acc: 0.9506 - val_loss: 0.6503 - val_acc: 0.7755
Epoch 9/10
 - 3s - loss: 0.0958 - acc: 0.9624 - val_loss: 0.7699 - val_acc: 0.7688
Epoch 00009: early stopping
1067/1067 [==============================] - 0s 86us/step
test_loss, test_acc: [0.8930901289265516, 0.7535145266545411]
```

**Word level LSTM**
```
Train on 8635 samples, validate on 960 samples
Epoch 1/10
 - 34s - loss: 0.6546 - acc: 0.6244 - val_loss: 0.6108 - val_acc: 0.7234
Epoch 2/10
 - 33s - loss: 0.5260 - acc: 0.7559 - val_loss: 0.5203 - val_acc: 0.7448
Epoch 3/10
 - 30s - loss: 0.4028 - acc: 0.8321 - val_loss: 0.5121 - val_acc: 0.7615
Epoch 4/10
 - 30s - loss: 0.2947 - acc: 0.8905 - val_loss: 0.5390 - val_acc: 0.7776
Epoch 5/10
 - 29s - loss: 0.1837 - acc: 0.9353 - val_loss: 0.6027 - val_acc: 0.7526
Epoch 6/10
 - 29s - loss: 0.1316 - acc: 0.9602 - val_loss: 0.9718 - val_acc: 0.7750
Epoch 7/10
 - 30s - loss: 0.0723 - acc: 0.9812 - val_loss: 1.0995 - val_acc: 0.7760
Epoch 8/10
 - 30s - loss: 0.0377 - acc: 0.9895 - val_loss: 1.2665 - val_acc: 0.7630
Epoch 00008: early stopping
1067/1067 [==============================] - 1s 1ms/step
test_loss, test_acc:  [1.5654271245561366, 0.7403936269357032]
```

**Char level CNN**

As for the embedding wegihts, we first use the one-hot encoding.

```
Train on 8635 samples, validate on 960 samples
Epoch 1/10
 - 11s - loss: 0.6934 - acc: 0.5140 - val_loss: 0.6925 - val_acc: 0.5177
Epoch 2/10
 - 11s - loss: 0.6795 - acc: 0.5672 - val_loss: 0.6801 - val_acc: 0.5677
Epoch 3/10
 - 10s - loss: 0.6562 - acc: 0.6085 - val_loss: 0.6813 - val_acc: 0.5714
Epoch 4/10
 - 10s - loss: 0.6270 - acc: 0.6425 - val_loss: 0.6739 - val_acc: 0.5833
Epoch 5/10
 - 10s - loss: 0.6055 - acc: 0.6687 - val_loss: 0.6746 - val_acc: 0.5807
Epoch 6/10
 - 10s - loss: 0.5885 - acc: 0.6845 - val_loss: 0.6681 - val_acc: 0.5885
Epoch 7/10
 - 11s - loss: 0.5644 - acc: 0.7041 - val_loss: 0.6608 - val_acc: 0.5927
Epoch 8/10
 - 12s - loss: 0.5542 - acc: 0.7119 - val_loss: 0.6758 - val_acc: 0.6193
Epoch 9/10
 - 14s - loss: 0.5424 - acc: 0.7193 - val_loss: 0.6772 - val_acc: 0.6198
Epoch 10/10
 - 14s - loss: 0.5298 - acc: 0.7283 - val_loss: 0.6696 - val_acc: 0.6021
1067/1067 [==============================] - 0s 433us/step
test_loss, test_acc:  [0.6921688347859369, 0.5993439547347487]
```

Char level CNN with Random Embedding Weights
`test_loss, test_acc:  [0.6814835615379294, 0.6002811620809704]`

We can see the random embedding weights is better than the one-hot encoding wegihts.

**Char level LSTM**
```
Train on 8635 samples, validate on 960 samples
Epoch 1/10
 - 156s - loss: 0.6935 - acc: 0.4977 - val_loss: 0.6932 - val_acc: 0.4979
Epoch 2/10
 - 156s - loss: 0.6933 - acc: 0.5023 - val_loss: 0.6931 - val_acc: 0.5000
Epoch 3/10
 - 138s - loss: 0.6933 - acc: 0.4986 - val_loss: 0.6930 - val_acc: 0.5125
Epoch 4/10
 - 155s - loss: 0.6933 - acc: 0.4974 - val_loss: 0.6928 - val_acc: 0.5188
Epoch 5/10
 - 155s - loss: 0.6929 - acc: 0.5113 - val_loss: 0.6896 - val_acc: 0.5312
Epoch 6/10
 - 163s - loss: 0.7470 - acc: 0.5206 - val_loss: 0.6914 - val_acc: 0.5286
Epoch 7/10
 - 155s - loss: 0.6908 - acc: 0.5240 - val_loss: 0.6891 - val_acc: 0.5167
Epoch 8/10
 - 153s - loss: 0.6884 - acc: 0.5420 - val_loss: 0.6881 - val_acc: 0.5167
Epoch 9/10
 - 158s - loss: 0.6932 - acc: 0.5428 - val_loss: 0.6863 - val_acc: 0.5458
Epoch 10/10
 - 147s - loss: 0.6897 - acc: 0.5298 - val_loss: 0.6862 - val_acc: 0.5453
1067/1067 [==============================] - 5s 5ms/step
test_loss, test_acc:  [0.6856050494126102, 0.5459231490159325]
```

**Subword level CNN**
```
Train on 8635 samples, validate on 960 samples
Epoch 1/10
 - 12s - loss: 0.6684 - acc: 0.5840 - val_loss: 0.6253 - val_acc: 0.6490
Epoch 2/10
 - 12s - loss: 0.5673 - acc: 0.7087 - val_loss: 0.5627 - val_acc: 0.7141
Epoch 3/10
 - 12s - loss: 0.4492 - acc: 0.7906 - val_loss: 0.5197 - val_acc: 0.7469
Epoch 4/10
 - 12s - loss: 0.3559 - acc: 0.8477 - val_loss: 0.5149 - val_acc: 0.7526
Epoch 5/10
 - 12s - loss: 0.2744 - acc: 0.8893 - val_loss: 0.5725 - val_acc: 0.7552
Epoch 6/10
 - 12s - loss: 0.2091 - acc: 0.9159 - val_loss: 0.6149 - val_acc: 0.7562
Epoch 7/10
 - 12s - loss: 0.1551 - acc: 0.9372 - val_loss: 0.7048 - val_acc: 0.7526
Epoch 8/10
 - 12s - loss: 0.1260 - acc: 0.9501 - val_loss: 0.8029 - val_acc: 0.7625
Epoch 9/10
 - 12s - loss: 0.0966 - acc: 0.9625 - val_loss: 0.8579 - val_acc: 0.7490
Epoch 00009: early stopping
1067/1067 [==============================] - 0s 294us/step
test_loss, test_acc:  [0.9087518500522016, 0.7357075913218326]
```

**Subword level LSTM**
```
Train on 8635 samples, validate on 960 samples
Epoch 1/10
 - 138s - loss: 0.6688 - acc: 0.5917 - val_loss: 0.6050 - val_acc: 0.7021
Epoch 2/10
 - 136s - loss: 0.5558 - acc: 0.7423 - val_loss: 0.6002 - val_acc: 0.7307
Epoch 3/10
 - 167s - loss: 0.4229 - acc: 0.8209 - val_loss: 0.5945 - val_acc: 0.7495
Epoch 4/10
 - 148s - loss: 0.3586 - acc: 0.8573 - val_loss: 0.6814 - val_acc: 0.7469
Epoch 5/10
 - 171s - loss: 0.2653 - acc: 0.8997 - val_loss: 0.8331 - val_acc: 0.7583
Epoch 6/10
 - 159s - loss: 0.2831 - acc: 0.9011 - val_loss: 1.0862 - val_acc: 0.7573
Epoch 7/10
 - 147s - loss: 0.2246 - acc: 0.9299 - val_loss: 0.8011 - val_acc: 0.7396
Epoch 8/10
 - 160s - loss: 0.1966 - acc: 0.9276 - val_loss: 1.1622 - val_acc: 0.7516
Epoch 00008: early stopping
1067/1067 [==============================] - 5s 5ms/step
test_loss, test_acc:  [1.2638732995960125, 0.7225866916029947]
```

