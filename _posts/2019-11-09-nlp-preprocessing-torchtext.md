
---
layout: post
published: true
title: >-
  NLP Processing with Torchtext
---

```python
import numpy as np
import pandas as pd
import torchtext
import torch
```

### NLP Data preprocessing

Most of the NLP tasks involves a set of preprocessing steps after which they can be fed into any machine learning model. The common ones are

- **Text cleaning** - involves punctuations and stop-words removal, lemmetization etc. These steps make sense when you have simple model, not necessary for deep learning etc.
- **Tokenization** - splitting the sentence into individual tokens, and generate ngram tokens if necessary.
- **numericalization** - build a vocabulary and assign a number to each word (and ngrams) in it. Special characters like < unk > < pad > are also assigned a number
- **Generate sentence vectors** - converts a sentence into a list of numbers from the vocab, can include padding
- **Use pretrained word embeddings** - this step is task specific. In some tasks it makes sense to use pretrained embeddings instead of training embeddings from scratch.
    
These steps are usually handled by writing ad-hoc code, and it's very common that some bugs introduced in these steps will affect the whole pipeline. So given this background I came across torchtext. In the outset it provides an API that takes a dataset as input to a wrapper that has all these steps configured in it. Beyond this it provides various features which are needed for specific tasks like language modelling, language translation etc.

In addition to this it integrates well with pytorch's entire ecosystem.

### Example NLP dataset

I'll try to apply these steps to apply these steps to AG_NEWS dataset. This dataset is readily available in torchtext, but to get a better understanding I'll load the dataset from the files.


```python
!ls ./data/ag_news_csv/
```

    classes.txt  readme.txt  test.csv  train.csv



```python
train_df = pd.read_csv('./data/ag_news_csv/test.csv', header=None)
```


```python
train_df.head()
# label, name and text
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3</td>
      <td>Fears for T N pension after talks</td>
      <td>Unions representing workers at Turner   Newall...</td>
    </tr>
    <tr>
      <td>1</td>
      <td>4</td>
      <td>The Race is On: Second Private Team Sets Launc...</td>
      <td>SPACE.com - TORONTO, Canada -- A second\team o...</td>
    </tr>
    <tr>
      <td>2</td>
      <td>4</td>
      <td>Ky. Company Wins Grant to Study Peptides (AP)</td>
      <td>AP - A company founded by a chemistry research...</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>Prediction Unit Helps Forecast Wildfires (AP)</td>
      <td>AP - It's barely dawn when Mike Fitzpatrick st...</td>
    </tr>
    <tr>
      <td>4</td>
      <td>4</td>
      <td>Calif. Aims to Limit Farm-Related Smog (AP)</td>
      <td>AP - Southern California's smog-fighting agenc...</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_df[0].nunique()
```




    4



### Define Fields

The Field class is a wrapper that holds all the configuration of all the steps needed to covert text to tensors. Going over steps in our process that are solved in `Field` class.

- Text Cleaning - `preprocessing` parameter takes in function, to which the **tokenized** list of the sentence is passed.

- Tokenization - `tokenize` parameter takes in a function to tokenize a sentence or it takes in names of the tokenizer to use (e.g. spacy, moses, toktok, revtok, subword). `tokenizer_language` can specify the language if the tokenization library supports it. and also tokenization is applied only to `sequential` field

- numericalization - To let Field know that it has to be numericalized `use_vocab` must be set to `True`, the actual `Vocab` building will happen after we've built the dataset.

Few more interesting params to mind are `batch_first` - controls if the batch_size should be first dim of the tensor. `include_lengths` - this field in the Example will return the lengths of the sentences as a tensor along with padded sentence tensors. Most other params are self-explanatory.


```python
from torchtext.data import Field

text_field = Field(
    sequential=True,
    use_vocab=True,
    tokenize='spacy',
    lower=True,
    tokenizer_language='en',
    include_lengths=True,
    batch_first=True,
    preprocessing=None
)

label_field = Field(
    sequential=False,
    use_vocab=False,
    preprocessing=lambda x: int(x) - 1,
    is_target=True,
    pad_token=None,
    unk_token=None
)
```

In most deep learning approaches it doesn't we don't include n-grams, because RNNs learn from a sequence anyway. But it case we need to generate n-grams, it can be done with torchtext using a few `utils` function and passing this function to `tokenize` parameter of the `Field`.


```python
from torchtext.data.utils import get_tokenizer, ngrams_iterator
from functools import partial
```


```python
def ngram_tokenizer(sentence, tokenizer, ngrams):
    return list(ngrams_iterator(tokenizer(sentence), ngrams=ngrams))

spacy_tokenizer = get_tokenizer('spacy')
spacy_ngram_tokenizer = partial(ngram_tokenizer,tokenizer=spacy_tokenizer, ngrams=2)
```


```python
sentence = 'I want 2-grams'
spacy_ngram_tokenizer(sentence)
```




    ['I', 'want', '2-grams', 'I want', 'want 2-grams']



Same `Field` can be mapped to multiple columns, but for sake of simplicity. I'll skip the name column


```python
data_fields = [
    ('label', label_field),
    ('name', None),
    ('text', text_field)
]
```

### Define dataset

`TabularDataset`can be used to load data from .csv, .tsv, of .json files. torchtext Dataset can basically thought of as a link between the data file and the `Field`s we've defined for the data.

We can use the class method `splits` to get train, validation and test dataset.


```python
train_dataset, test_dataset = torchtext.data.TabularDataset.splits(path='./data/ag_news_csv',
                                                               format='csv',
                                                               train='train.csv',
                                                               test='test.csv',
                                                               fields=data_fields
                                                              )
```

Each dataset is composed of `Example`s. Each `Example` object has the fields as it's attributes.


```python
sample = test_dataset[1]
```


```python
sample
```




    <torchtext.data.example.Example at 0x7f1d051aeef0>




```python
print(sample.label)
print(sample.text)
```

    3
    ['space.com', '-', 'toronto', ',', 'canada', '--', 'a', 'second\\team', 'of', 'rocketeers', 'competing', 'for', 'the', ' ', '#', '36;10', 'million', 'ansari', 'x', 'prize', ',', 'a', 'contest', 'for\\privately', 'funded', 'suborbital', 'space', 'flight', ',', 'has', 'officially', 'announced', 'the', 'first\\launch', 'date', 'for', 'its', 'manned', 'rocket', '.']


## Build vocabulary and load pretrained vectors

Building vocabulary and assigning pretrained vectors to them is a breeze with torchtext. `Vocab` of a `Field` can be built by using the `build_vocab` fn of the `Field` class. Let's go over the params of this fn.

- datasets - pass datasets from which the `Vocab` must be built as parameters. The vocabulary is built from all columns of this `Field`

- max_size, min_freq - Both of these can be used control number of tokens. max_size, set the max number of tokens for the vocabulary, this is basically decided based on decreasing order of count. min_freq sets the minumum number of occurances to consider a word as token

- vectors - pass pretrained vectors as a `Vector` object or pass strings of the available pretrained vectors in torchtext. Currently availble vectors are charngram.100d fasttext.en.300d fasttext.simple.300d glove.42B.300d glove.840B.300d glove.twitter.27B.25d glove.twitter.27B.50d glove.twitter.27B.100d glove.twitter.27B.200d glove.6B.50d glove.6B.100d glove.6B.200d glove.6B.300d.

- unk_init - by default OOV vectors are assigned to 0s. This function can be used to initialize to other values. The current strategy I use is to set it to zero and make the embedding layer trainable. I'll add other strategies as I come across it.


```python
text_field.build_vocab(train_dataset, test_dataset, max_size=100000, min_freq=2, vectors='fasttext.en.300d')
```


```python
len(text_field.vocab.itos), text_field.vocab.itos[20]
```




    (45515, 'as')




```python
text_field.vocab.stoi['as']
```




    20



Now once the vocab is build we could numericalize any sentence using `numericalize` method. 


```python
sentence = 'India attempted soft landing on the moon with Chandrayan-2'
tokens = text_field.tokenize(sentence)
print(tokens)
text_field.numericalize(([tokens], [len(tokens)]))
```

    ['India', 'attempted', 'soft', 'landing', 'on', 'the', 'moon', 'with', 'Chandrayan-2']





    (tensor([[   0, 4719, 2580, 3042,   11,    2, 1221,   19,    0]]), tensor([9]))



Let's check the percentage og OOV vectors in our vocab. It helps to check this, and go back improve the preprocessing step.


```python
oov_mask = (text_field.vocab.vectors.sum(1) == 0).numpy().astype(int)
```


```python
oov_mask.mean()
```




    0.17205316928485115




```python
pd.Series(text_field.vocab.itos)[oov_mask == 1]
```




    0               <unk>
    1               <pad>
    12                   
    16               39;s
    27                 's
                 ...     
    45467              z3
    45473        zahameel
    45480          zardas
    45500           zlc.n
    45514    zz:001107539
    Length: 7831, dtype: object



### Split Train and validation dataset

This can be done with `split` method available in the `Dataset`


```python
train_split_dataset, valid_split_dataset = train_dataset.split(split_ratio=0.7, stratified=True)
```


```python
print(len(train_split_dataset))
print(len(valid_split_dataset))
```

    84000
    36000


### Load numericalized vectors in batches

`Iterator` can be thought of as the `DataLoader` class in pytorch, with other NLP specific features. There are few things to take note while batching NLP datasets. Generally, in order to process a batch of sentences, each vector in the batch must be of the same length this acheived by adding padding tokens < pad > at the end. So if the data is loaded without any ordering, it's possible that a batch can have a very high number of padding tokens. 

To counter this we could feed the data by sorting it by sentence lengths. This improves the solution but the problem feeding data in increasing sentence lengths can make the model learn this distribution of data, and it might not have the same results when this changes in the real world. We could overcome this by introducing some randomness in the sorting, which is exactly what `BucketIterator` does. 

We will again use the classmethod `splits` to generate Iterators for train, valid and test. Going over the params

`sort_key` - function that gets a `Example` from the dataset as input and must output the sort key
`sort_within_batch` - Within the batch sort the examples in descending order by sort_key, this is necessary in some cases, for Example, `pack_padded_sequence` expects a batch to be sorted by descending order.

others are self-explanatory.


```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_batchloader, valid_batchloader, test_textloader = torchtext.data.BucketIterator.splits(
    datasets=(train_split_dataset, valid_split_dataset, test_dataset),
    batch_size=32,
    sort_key=lambda x: len(x.text),
    device=device,
    sort_within_batch=True,
    repeat=False
)
```


```python
batch = next(iter(train_textloader))
```


```python
batch
```




    
    [torchtext.data.batch.Batch of size 32]
    	[.label]:[torch.LongTensor of size 32]
    	[.text]:('[torch.LongTensor of size 32x40]', '[torch.LongTensor of size 32]')




```python
batch.text, batch.label
```




    ((tensor([[   12,   201,    17,  ...,    12, 11141,     4],
              [ 2996,  2773,    34,  ...,    29,   614,     4],
              [ 1090,   272,     3,  ...,    68,    83,     4],
              ...,
              [11602,  2999,  2632,  ...,  9744,  1291,     4],
              [    5,  1346,  3671,  ...,  1257,   190,     9],
              [ 2508,   566,    17,  ...,   230,  1513,   164]]),
      tensor([40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
              40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40])),
     tensor([2, 1, 0, 0, 0, 0, 0, 2, 0, 0, 3, 1, 3, 3, 0, 0, 2, 3, 2, 0, 3, 3, 0, 2,
             0, 1, 1, 0, 2, 1, 0, 2]))



Most training loops expect a `(X,y)` tuple as the batch, so lets write a wrapper over this to acheive the same.


```python
class FormatedBatchLoader:
    def __init__(self, dataloader, x_attr, y_attr):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.x_attr = x_attr
        self.y_attr = y_attr
        
    def __len__(self):
        return len(self.dataloader)
        
    def __iter__(self):
        for batch in self.dataloader:
            x = getattr(batch, self.x_attr)
            y = getattr(batch, self.y_attr)
            yield (x, y)
```


```python
train_dataloader = FormatedBatchLoader(train_textloader, 'text', 'label')
valid_dataloader = FormatedBatchLoader(valid_textloader, 'text', 'label')
```


```python
batch = next(iter(train_dataloader))
```


```python
batch
```




    ((tensor([[  655,  1015,    45,  ...,  1521,   154,    26],
              [  221,   735, 10198,  ...,   648,  1312,     4],
              [   91,   347,   544,  ...,   459,  7442,     4],
              ...,
              [    2,   158,   278,  ...,    18,   140,     4],
              [    2,   373,  1114,  ...,    10,     0,   983],
              [   12,    29,    69,  ...,    12,  2882,     4]]),
      tensor([44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44,
              44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44])),
     tensor([1, 3, 0, 1, 2, 1, 1, 0, 2, 3, 2, 1, 0, 0, 2, 0, 1, 0, 0, 3, 3, 1, 2, 2,
             3, 0, 0, 3, 0, 0, 3, 3]))



This process with little tweaks can be repeated for most NLP tasks. I'll update if I found better ways to handle any of the steps above.

Along with torchtext documentation these writeups helped me in putting this together, and inspired the ideas presented here.

https://medium.com/@sonicboom8/sentiment-analysis-torchtext-55fb57b1fab8 

http://anie.me/On-Torchtext/

