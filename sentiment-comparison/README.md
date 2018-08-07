
# Report of Word/Character/Subword level for Sentiment Analysis

In the three projects I have done before, I use three levels to represent the text, word/character/subword.
- Word level for [Cnn-text-classification](https://github.com/BrambleXu/nlp-beginner-guide-keras/tree/master/cnn-text-classification)(word2vec), [word_embedding](https://github.com/BrambleXu/nlp-beginner-guide-keras/tree/master/word_embedding)(glove)
- Character level for [char-level-cnn](https://github.com/BrambleXu/nlp-beginner-guide-keras/tree/master/char-level-cnn)
- Subword level for [subword-level](https://github.com/BrambleXu/nlp-beginner-guide-keras/tree/master/subword-level)

After do some research in the ACL2018, I found that most works still choose word level for the sentiment analysis. Dose this means word-level representation is better character/subword level representation?
In order to analyze the performance of these representation for the sentiment analysis task, I ran some experiments for the the comparison.

Read the complete report on my blog: [Report on Sentiment Analysis using Word/Character/Subword level Embedding
](https://medium.com/@zhuixiyou/blog-md-34c5d082a8c5)

According to the result, subword-level embedding is useful for the dataset with many unknown words. The CNN not only achieve the better performance, but also take less training time. So if you want to implement a simple and powerful sentiment classification model, I highly recommend to use the CNN model.

![image.png](https://qiita-image-store.s3.amazonaws.com/0/82724/be2cdfab-a407-35c3-f54e-a2d6f4543996.png)

