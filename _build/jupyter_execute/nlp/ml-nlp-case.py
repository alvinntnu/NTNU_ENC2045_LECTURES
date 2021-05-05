#!/usr/bin/env python
# coding: utf-8

# # Machine Learning: NLP Tasks

# - Let's take a look at a few more classification tasks in NLP.
# 
# - In more complex NLP tasks, feature engineering (text vectorization) can be more complicated. We often need to come up with heuristics to extract features from the texts.
# 
# - In this lecture, we demonstrate a few NLP tasks and focus on a heuristics-based approach to feature engineering.

# ## NLP Tasks and Base Units for Classification

# - Document Sentiment/Topic Classification
#     - Unit: Document
#     - Label: Document's sentiment
# - POS Classification
#     - Unit: Word
#     - Label: Word's POS
# - Sentence Segmentation
#     - Unit: Word
#     - Label: Whether the word is sentence boundary or not
# - Dialogue Act Classification
#     - Unit: Utterance
#     - Label: The dialogue act of the utterance
# 
# ---

# ```{tip}
# For NLP classification tasks, it is very important to determine the base units on which the classification is being made. 
# 
# This should always be made explicit when we come up with the research questions.
# 
# ```

# In[1]:


import nltk, random


# ## Document Sentiment Classification

# In[2]:


from nltk.corpus import movie_reviews
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)


# - Find the top 2000 words in the entire corpus
# - Use these words as the document features

# In[3]:


all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features


# In[4]:


print(document_features(movie_reviews.words('pos/cv957_8737.txt'))) 


# In[5]:


featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)


# In[6]:


print(nltk.classify.accuracy(classifier, test_set))


# In[7]:


classifier.show_most_informative_features(5)


# ## Parts-of-Speech Tagging

# In[8]:


from nltk.corpus import brown
suffix_fdist = nltk.FreqDist()

for word in brown.words():
    word = word.lower()
    suffix_fdist[word[-1:]] += 1
    suffix_fdist[word[-2:]] += 1
    suffix_fdist[word[-3:]] += 1


# In[9]:


common_suffixes = [suffix for (suffix, count) in 
                   suffix_fdist.most_common(100)]


# In[10]:


print(common_suffixes)


# In[11]:


def pos_features(word):
    features = {}
    for suffix in common_suffixes:
        features['endswith({})'.format(suffix)] = word.lower().endswith(suffix)
        return features


# In[12]:


tagged_words = brown.tagged_words(categories='news')
featuresets = [(pos_features(n), g) for (n,g) in tagged_words]


# In[13]:


size = int(len(featuresets) * 0.1)


# In[14]:


train_set, test_set = featuresets[size:], featuresets[:size]


# In[15]:


classifier = nltk.DecisionTreeClassifier.train(train_set)


# In[16]:


nltk.classify.accuracy(classifier, test_set)


# In[17]:


classifier.classify(pos_features('cats'))


# ## Sentence Boundary

# In[18]:


sents = nltk.corpus.treebank_raw.sents()
tokens = []
boundaries = set()
offset = 0
for sent in sents:
    tokens.extend(sent) # append tokens of each sent to `tokens`
    offset += len(sent) # update the index of each word token
    boundaries.add(offset-1) # record the index of sent boundary token


# In[19]:


def punct_features(tokens, i):
    return {'next-word-capitalized': tokens[i+1][0].isupper(),
            'prev-word': tokens[i-1].lower(),
            'punct': tokens[i],
            'prev-word-is-one-char': len(tokens[i-1]) == 1}


# In[20]:


# create featuresets
# by selecting only sentence boundary tokens
featuresets = [(punct_features(tokens, i), (i in boundaries))
               for i in range(1, len(tokens)-1) 
               if tokens[i] in '.?!']


# In[21]:


size = int(len(featuresets) * 0.1)


# In[22]:


train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)


# In[23]:


classifier.classify(punct_features(tokens, 2))
tokens[0:2]


# In[24]:


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


# In[25]:


text = "Python is an interpreted, high-level and general-purpose programming language. Python's design philosophy emphasizes code readability with its notable use of significant indentation. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects."
nltk.word_tokenize(text)[:20]


# In[26]:



segment_sentences(nltk.word_tokenize(text))


# ## Dialogue Act Classification

# - NPS Chat Corpus consists of over 10,000 posts from instant messaging sessions.
# - Thse poasts have been labeled with one of  15 dialogue act types.

# In[27]:


posts = nltk.corpus.nps_chat.xml_posts()[:10000]


# In[28]:


[p.text for p in posts[:10]]


# In[29]:


# bag-of-words
def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
        return features


# In[30]:


featuresets = [(dialogue_act_features(post.text), post.get('class'))
               for post in posts]


# In[31]:


size = int(len(featuresets) * 0.1)

train_set, test_set = featuresets[size:], featuresets[:size]


# In[32]:


classifier = nltk.NaiveBayesClassifier.train(train_set)


# In[33]:


print('Accuracy: {:4.2f}'.format(nltk.classify.accuracy(classifier, test_set))) 


# In[34]:


test_featureset = [f for (f, l) in test_set]
test_label = [l for (f, l) in test_set] 


# In[35]:



test_label_predicted = [classifier.classify(f) for f in test_featureset]

cm=nltk.ConfusionMatrix(test_label, test_label_predicted)
print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))


# ## References
# 
# - [NLTK Book Chapter 6: Learning to Classify Text](https://www.nltk.org/book/ch06.html)
