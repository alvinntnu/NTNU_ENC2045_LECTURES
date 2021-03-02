# Text Enrichment

The objective of text enrichment is to utilize computational techniques of automatic annotations and extract additional linguistic information from the text.

## Parts-of-Speech (POS) Tagging

- Every POS tagger needs to first operationlize a tagset, i.e., a complete list of possible tags for the entire corpus.
- A common tagset for English is [Penn-treebank tagset](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html).


### NLTK

- NLTK default POS tagger:

import nltk
text = "This is a sentence."
text_word = nltk.word_tokenize(text)
text_pos = nltk.pos_tag(text_word)
print(text_pos)

- Using NLTK to Train Taggers


from nltk.corpus import treebank
print(len(treebank.tagged_sents())) # total number of sents

# train-test for training and testing taggers
test_sents = treebank.tagged_sents()[3000:]
train_sents = treebank.tagged_sents()[:3000]

test_sents[1]

- NgramTagger

```{figure} ../images/nltk-fig-5-1-tag-context.png
---
height: 250px
name: ngram-tagger
---
From NLTK Book Ch 5. Figure 5-1
```

from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger

unigram_tagger = UnigramTagger(train_sents)
bigram_tagger = BigramTagger(train_sents)
trigram_tagger = TrigramTagger(train_sents)


unigram_tagger.evaluate(test_sents)

bigram_tagger.evaluate(test_sents)

trigram_tagger.evaluate(test_sents)

- NLTK Taggers (`nltk.tag`):
    - `DefaultTagger`
    - `UnigramTagger`, `BigramTagger`, `TrigramTagger`
    - `RegexpTagger`
    - `AffixTagger`
    - `ClassifierBasedPOSTagger`

- Backoff Tagging

unigram_tagger = UnigramTagger(train_sents)
bigram_tagger = BigramTagger(train_sents,backoff=unigram_tagger)
trigram_tagger = TrigramTagger(train_sents, backoff=bigram_tagger)

trigram_tagger.evaluate(test_sents)

:::{tip}
Sometimes it may take a while to train a tagger. We can pickle a trained tagger for later usage.

```
import pickle
f = open('trigram-backoff-tagger.pickle', 'wb')
f.close()
f = open('trigram-backoff-tagger.pickle', ;rb)
tagger = pickle.load(f)
```
:::

- Classifier-based Tagger

from nltk.tag.sequential import ClassifierBasedPOSTagger
cbtagger = ClassifierBasedPOSTagger(train=train_sents)
cbtagger.evaluate(test_sents)

```{note}
By default, the `ClassifierBasedPOSTagger` uses a `NaiveBayesClassifier` for training.
```

To try other classifers, e.g., Maximum Entropy Classifier:

import numpy
# warning is not logged here. Perfect for clean unit test output
with numpy.errstate(divide='ignore'):
    numpy.float64(1.0) / 0.0

from nltk.classify import MaxentClassifier
metagger = ClassifierBasedPOSTagger(train=train_sents,
                                   classifier_builder=MaxentClassifier.train)
metagger.evaluate(test_sents)

- Classifier-based with Cut-off Probability

from nltk.tag import DefaultTagger

default = DefaultTagger('NN')
cbtagger2 = ClassifierBasedPOSTagger(train=train_sents,
                                    backoff=default,
                                    cutoff_prob=0.3)

cbtagger2.evaluate(test_sents)

### spacy

## Chunking

- Chunk extraction is the processing of extracting short phrases from a part-of-speech tagged sentence.
- This is different from parsing in that we are only interested in standalone chunks or phrases, instead of the full parsed syntactic tree.

- A `ChunkRule` class specifies what to *include* in a chunk, while a `ChinkRule` class specifies what to *exclude* from a chunk.
- Chunking creates chunks and chinking breaks up those chunks.

- For example, we can extract NP chunks.

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150 # default for me was 75
plt.rcParams['figure.figsize'] = (10,10)

from nltk.chunk import RegexpParser
np_chunker = RegexpParser(r'''
NP:
{<DT><NN.*><.*>*<NN.*>} # chunk
}<VB.*>{ # chink
''')

s = "This course has many interesting topics"
np_chunker.parse(nltk.pos_tag(nltk.word_tokenize(s)))

- We can extract proper noun chunks.

## Parsing

## References

- [NLTK Ch.5 Categorizing and Tagging Words](http://www.nltk.org/book/ch05.html)