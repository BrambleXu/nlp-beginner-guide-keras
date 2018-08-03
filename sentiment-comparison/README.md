
# Report of Word/Character/Subword level for Sentiment Analysis

In the three projects I have done before, I use three levels to represent the text, word/character/subword.
- Word level for [Cnn-text-classification](https://github.com/BrambleXu/nlp-beginner-guide-keras/tree/master/cnn-text-classification)(word2vec), [word_embedding](https://github.com/BrambleXu/nlp-beginner-guide-keras/tree/master/word_embedding)(glove)
- Character level for [char-level-cnn](https://github.com/BrambleXu/nlp-beginner-guide-keras/tree/master/char-level-cnn)
- Subword level for [subword-level](https://github.com/BrambleXu/nlp-beginner-guide-keras/tree/master/subword-level)

After do some research in the ACL2018, I found that most works still choose word level for the sentiment analysis. Dose this means word-level representation is better character/subword level representation?
In order to analyze the performance of these representation for the sentiment analysis task, I ran some experiments for the the comparison.

<!--Read the complete report on my blog-->

According to the result, subword-level embedding is useful for the dataset with many unknown words. The CNN not only achieve the better performance, but also take less training time. So if you want to implement a simple and powerful sentiment classification model, I highly recommend to use the CNN model.

![image.png](https://upload-images.jianshu.io/upload_images/283834-14d6ca0c3911bca6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
