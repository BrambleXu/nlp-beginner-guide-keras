# Subword level representation

In this project, we will preprocess the data to represent sentences in a subword level.
The data set is `ag_news`, same with `char-level-cnn` [project](https://github.com/BrambleXu/nlp-beginner-guide-keras/tree/master/char-level-cnn).
The reason that I create [nlp-beginner-guide-keras](https://github.com/BrambleXu/nlp-beginner-guide-keras) is to learn different techniques, so here we use a different approach to do the preprocess.
We will use subword level word representation, instead of character level word representation.


## What is subword level representation

Just like the name, subword means some part or a word. See the example below.

```
bpe = BPE("../pre-trained-model/en.wiki.bpe.op25000.vocab")
print(bpe.encode(' this is our house in boomchakalaka'))

▁this ▁is ▁our ▁house ▁in ▁boom ch ak al aka
```

We can see `this`, `is`, `our`, `house`, `in` are still representation as a word level.
But for the word `boomchakalaka`, it is represented as `▁boom` `ch` `ak` `al` `aka`.
This is the subword repersentation.


## Why use subword level representation

The subword representation is useful for tasks with many unknown words.
This method can contain many information for unknown words compared with the word level representation.
It did improve the performance in the machine translation task.
But according to the [From Characters to Words to in Between: Do We Capture Morphology?
](https://arxiv.org/abs/1704.08352), subword representation do not outperform the character representation for most tasks.

## How to get subword

The segmentation is based on Byte Pair Encoding (BPE), which is simple data compression technique that iteratively replaces the most frequent pair of bytes in a sequence with a single, unused byte.
You can find more info in [wiki](https://en.wikipedia.org/wiki/Byte_pair_encoding)

As for the preprocessing, you can find detail explanation in this notebook [subword-preprocess](https://github.com/BrambleXu/nlp-beginner-guide-keras/blob/master/char-level-rnn/notebooks/subword-preprocess.ipynb).



## References

Here I will list some useful resource when I learn to implement the paper.

**github repo**
- [SentencePiece](https://github.com/google/sentencepiece)
- [BPEmb](https://github.com/bheinzerling/bpemb)

