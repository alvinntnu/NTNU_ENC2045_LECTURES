# Text Enrichment

The objective of text enrichment is to utilize computational techniques of automatic annotations and extract additional linguistic information from the text.

## Parts-of-Speech (POS) Tagging

- Every POS tagger needs to first operationlize a tagset, i.e., a complete list of possible tags for the entire corpus.
- A common tagset for English is [Penn-treebank tagset](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html).


### NLTK

- NLTK default POS tagger

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

![](../images/nltk-fig-5-1-tag-context.png) <small>(From NLTK Book Ch 5. Figure 5-1)</small>


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
    - The idea of **backoff** is that for longer sequences, we are more likely to encounter *unseen* n-grams in the test data.
    - To avoid the zero probability issue due to the unseen **n**-grams, we can backoff the probability estimates using the lower-order (**n-1**)-grams.

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
f = open('trigram-backoff-tagger.pickle', 'rb')
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

- Another useful module for English POS tagging is to use `spacy`. We will come back to this module when we talk about parsing.

## Chunking

- **Chunk** extraction is the processing of extracting short phrases from a part-of-speech tagged sentence.
- This is different from parsing in that we are only interested in standalone chunks or phrases, instead of the full parsed syntactic tree.

- In NLTK, A `ChunkRule` class specifies what to *include* in a chunk, while a `ChinkRule` class specifies what to *exclude* from a chunk.
- **Chunking** creates chunks and **chinking** breaks up those chunks.
- Both rules utilize **regular expressions**.

- For example, we can extract NP chunks.

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150 # default for me was 75
plt.rcParams['figure.figsize'] = (20,20)
%matplotlib inline

from nltk.chunk import RegexpParser
np_chunker = RegexpParser(r'''
NP:
{<DT><NN.*><.*>*<NN.*>} # chunk
}<VB.*>{ # chink
''')

s = "This course has many interesting topics"
np_chunker.parse(nltk.pos_tag(nltk.word_tokenize(s)))

We can define a function to extract chunks from the tree.

def sub_leaves(tree, label):
    return [t.leaves() for t in 
            tree.subtrees(lambda s:s.label()==label)]

s_chunk_tree = np_chunker.parse(
    nltk.pos_tag(
        nltk.word_tokenize(s)))

sub_leaves(s_chunk_tree, "NP")

- Named Entity Chunker (NLTK)

from nltk.chunk import ne_chunk

ne_chunks=ne_chunk(treebank.tagged_sents()[0])
ne_chunks

sub_leaves(ne_chunks, "PERSON")

sub_leaves(ne_chunks, "ORGANIZATION")

## Parsing

For Parsing, we will use `spacy`, a powerful package for natural language processing.

import spacy
from spacy import displacy
# load language model
nlp_en = spacy.load('en_core_web_sm') 
# parse text 
doc = nlp_en('This is a sentence')

for token in doc:
    print((token.text, 
            token.lemma_, 
            token.pos_, 
            token.tag_,
            token.dep_,
            token.shape_,
            token.is_alpha,
            token.is_stop,
            ))

## Check meaning of a POS tag
spacy.explain('VBZ')

# Visualize
displacy.render(doc, style="dep")

options = {"compact": True, "bg": "#09a3d5",
           "color": "white", "font": "Source Sans Pro"}
displacy.render(doc, style="dep", options=options)

- To get the dependency relations, we first need to extract NP chunks, on which dependency relations are annotated.
- Please refer to [spacy documentation](https://spacy.io/usage/linguistic-features#dependency-parse) for more detail on dependency parsing.

doc2 = nlp_en(' '.join(treebank.sents()[0]))

for c in doc2.noun_chunks:
    print((c.text, 
           c.root.text, 
           c.root.dep_, 
           c.root.head.text))

Each NP chunk includes several important pieces of information:
- **Text**: The original noun chunk text.
- **Root text**: The original text of the word connecting the noun chunk to the rest of the parse.
- **Root dep**: Dependency relation connecting the root to its head.
- **Root head text**: The text of the root token’s head.

displacy.render(doc2, style="dep")

- Named Entity Extraction

for ent in doc2.ents:
    print((ent.text,
           ent.start_char,
           ent.end_char,
           ent.label_))

- Please check the documentation of [Universal Dependency Types](https://universaldependencies.org/docs/u/dep/index.html) proposed by [Marneffe et al.](https://nlp.stanford.edu/pubs/USD_LREC14_paper_camera_ready.pdf)

## References

- [NLTK Ch.5 Categorizing and Tagging Words](http://www.nltk.org/book/ch05.html)