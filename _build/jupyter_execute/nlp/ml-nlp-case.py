# Machine Learning: NLP Tasks

Let's take a look at a few more  classification tasks in NLP.

- Document Sentiment/Topic Classification
    - Unit: Document
    - Label: Document's sentiment
- POS Classification
    - Unit: Word
    - Label: Word's POS
- Sentence Segmentation
    - Unit: Word
    - Label: Whether the word is sentence boundary or not
- Dialogue Act Classification
    - Unit: Utterance
    - Label: The dialogue act of the utterance

---

```{tip}
For NLP classification tasks, it is very important to determine the base units on which the classification is being made. 

This should always be made explicit when we come up with the research questions.

```

import nltk, random

## Document Sentiment Classification

from nltk.corpus import movie_reviews
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

- Find the top 2000 words in the entire corpus
- Use these words as the document features

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

print(document_features(movie_reviews.words('pos/cv957_8737.txt'))) 


featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print(nltk.classify.accuracy(classifier, test_set))

classifier.show_most_informative_features(5)

## Parts-of-Speech Tagging

from nltk.corpus import brown
suffix_fdist = nltk.FreqDist()

for word in brown.words():
    word = word.lower()
    suffix_fdist[word[-1:]] += 1
    suffix_fdist[word[-2:]] += 1
    suffix_fdist[word[-3:]] += 1

common_suffixes = [suffix for (suffix, count) in 
                   suffix_fdist.most_common(100)]

print(common_suffixes)

def pos_features(word):
    features = {}
    for suffix in common_suffixes:
        features['endswith({})'.format(suffix)] = word.lower().endswith(suffix)
        return features

tagged_words = brown.tagged_words(categories='news')
featuresets = [(pos_features(n), g) for (n,g) in tagged_words]

size = int(len(featuresets) * 0.1)

train_set, test_set = featuresets[size:], featuresets[:size]


classifier = nltk.DecisionTreeClassifier.train(train_set)

nltk.classify.accuracy(classifier, test_set)

classifier.classify(pos_features('cats'))

## Sentence Boundary

sents = nltk.corpus.treebank_raw.sents()
tokens = []
boundaries = set()
offset = 0
for sent in sents:
    tokens.extend(sent) # append tokens of each sent to `tokens`
    offset += len(sent) # update the index of each word token
    boundaries.add(offset-1) # record the index of sent boundary token


def punct_features(tokens, i):
    return {'next-word-capitalized': tokens[i+1][0].isupper(),
            'prev-word': tokens[i-1].lower(),
            'punct': tokens[i],
            'prev-word-is-one-char': len(tokens[i-1]) == 1}

# create featuresets
# by selecting only sentence boundary tokens
featuresets = [(punct_features(tokens, i), (i in boundaries))
               for i in range(1, len(tokens)-1) 
               if tokens[i] in '.?!']

size = int(len(featuresets) * 0.1)

train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)

classifier.classify(punct_features(tokens, 2))
tokens[0:2]

def segment_sentences(words):
    start = 0
    sents = []
    #for i, word in enumerate(words): ## modified
    for i in range(1, len(words)-1):
        word = words[i]
        if word in '.?!' and classifier.classify(punct_features(words, i)) == True:
            sents.append(words[start:i+1])
            start = i+1
    if start < len(words):
        sents.append(words[start:])
    return sents

text = "Python is an interpreted, high-level and general-purpose programming language. Python's design philosophy emphasizes code readability with its notable use of significant indentation. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects."
nltk.word_tokenize(text)[:20]


segment_sentences(nltk.word_tokenize(text))

## Dialogue Act Classification

- NPS Chat Corpus consists of over 10,000 posts from instant messaging sessions.
- Thse poasts have been labeled with one of  15 dialogue act types.

posts = nltk.corpus.nps_chat.xml_posts()[:10000]

[p.text for p in posts[:10]]

# bag-of-words
def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
        return features

featuresets = [(dialogue_act_features(post.text), post.get('class'))
               for post in posts]

size = int(len(featuresets) * 0.1)

train_set, test_set = featuresets[size:], featuresets[:size]

classifier = nltk.NaiveBayesClassifier.train(train_set)

print('Accuracy: {:4.2f}'.format(nltk.classify.accuracy(classifier, test_set))) 

test_featureset = [f for (f, l) in test_set]
test_label = [l for (f, l) in test_set] 


test_label_predicted = [classifier.classify(f) for f in test_featureset]

cm=nltk.ConfusionMatrix(test_label, test_label_predicted)
print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))

## References

- [NLTK Book Chapter 6: Learning to Classify Text](https://www.nltk.org/book/ch06.html)