#!/usr/bin/env python
# coding: utf-8

# # Word Embeddings

# - The state-of-art method of vectorizing texts is to learn the numeric representations of words using deep learning methods.
# - These deep-learning based numeric representations of linguistic units are commonly referred to as **embeddings**.
# - Word embeddings can be learned either along with the target NLP task (e.g., the `Embedding` layer in RNN Language Model) or via an **unsupervised** method based on a large number of texts.
# - In this tutorial, we will look at two main algorithms in `word2vec` that allow us to learn the word embeddings in an **unsupervised** manner from a large collection of texts.

# - Strengths of word embeddings
#     - They can be learned using **unsupervised** methods.
#     - They include quite a proportion of the lexical **semantics**.
#     - They can be learned by **batch**. We don't have to process the entire corpus and create the word-by-document matrix for vectorization. 
#     - Therefore, it is less likely to run into the **memory** capacity issue for huge corpora.

# ## Overview

# ### What is `word2vec`?
# 
# - `Word2vec` is one of the most popular techniques to learn word embeddings using a two-layer neural network.
# - The input is a **text corpus** and the output is a set of **word vectors**.
# - Research has shown that these embeddings include rich semantic information of words, which allow us to perform interesting **semantic computation** (See Mikolov et al's works in References).

# ### Basis of Word Embeddings: Distributional Semantics
# 
# - "*You shall know a word by the company it keeps*" (Firth, 1975).
# - Word distributions show a considerable amount of **lexical semantics**.
# - Construction/Pattern distributions show a considerable amount of the **constructional semantics**.
# - Semantics of linguistic units are implicitly or explicitly embedded in their distributions (i.e., *occurrences* and *co-occurrences*) in language use (**Distributional Semantics**).

# ### Main training algorithms of `word2vec`
# 
# - Continuous Bag-of-Words (**CBOW**): The general language modeling task for embeddings training is to learn a model that is capable of using the ***context*** words to predict a ***target*** word.
# - **Skip-Gram**: The general language modeling task for embeddings training is to learn a model that is capable of using a ***target word*** to predict its ***context*** words.

# ![](../images/word2vec.png)

# - Other variants of embeddings training:
#   - `fasttext` from Facebook
#   - `GloVe` from Stanford NLP Group
# - There are many ways to train work embeddings.
#   - `gensim`: Simplest and straightforward implementation of `word2vec`.
#   - Training based on deep learning packages (e.g., `keras`, `tensorflow`)
#   - `spacy` (It comes with the pre-trained embeddings models, using GloVe.)
# - See Sarkar (2019), Chapter 4, for more comprehensive reviews.

# ### An Intuitive Understanding of CBOW

# ![](../images/word2vec-text-to-sequences.gif)

# ![](../images/word2vec-cbow.gif)

# ### An Intuitive Understanding of Skip-gram

# ![](../images/word2vec-skipgram.gif)

# ## Import necessary dependencies and settings

# In[1]:


import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 300
pd.options.display.max_colwidth = 200


# In[2]:


# # Google Colab Adhoc Setting
# !nvidia-smi
# nltk.download(['gutenberg','punkt','stopwords'])
# !pip show spacy
# !pip install --upgrade spacy
# #!python -m spacy download en_core_web_trf
# !python -m spacy download en_core_web_lg


# ## Sample Corpus: A Naive Example

# In[3]:


corpus = [
    'The sky is blue and beautiful.', 'Love this blue and beautiful sky!',
    'The quick brown fox jumps over the lazy dog.',
    "A king's breakfast has sausages, ham, bacon, eggs, toast and beans",
    'I love green eggs, ham, sausages and bacon!',
    'The brown fox is quick and the blue dog is lazy!',
    'The sky is very blue and the sky is very beautiful today',
    'The dog is lazy but the brown fox is quick!'
]
labels = [
    'weather', 'weather', 'animals', 'food', 'food', 'animals', 'weather',
    'animals'
]

corpus = np.array(corpus)
corpus_df = pd.DataFrame({'Document': corpus, 'Category': labels})
corpus_df = corpus_df[['Document', 'Category']]
corpus_df


# ### Simple text pre-processing
# 
# - Usually for unsupervised `word2vec` learning, we don't really need much text preprocessing.
# - So we keep our preprocessing to the minimum.
#     - Remove only symbols/punctuations, as well as redundant whitespaces.
#     - Perform word tokenization, which would also determine the base units for embeddings learning.
# 

# ### Suggestions
# 
# - If you are using `keras` to build the network for embeddings training, please prepare your input corpus data for `Tokenizer()`in the format where each **token** is delimited by a **whitespace**.
# - If you are using `gensim` to train word embeddings, please tokenize your corpus data first. That is, the `gensim` only requires a tokenized version of the corpus and it will learn the word embeddings for you. 

# In[4]:


wpt = nltk.WordPunctTokenizer()
# stop_words = nltk.corpus.stopwords.words('english')
def preprocess_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I | re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    doc = ' '.join(tokens)
    return doc

corpus_norm = [preprocess_document(text) for text in corpus]
corpus_tokens = [preprocess_document(text).split(' ') for text in corpus]


# In[5]:


print(corpus_norm)
print(corpus_tokens)


# ### Training Embeddings Using word2vec
# 
# - The expected inputs of `gensim.model.word2vec` is token-based corpus object.

# In[6]:


get_ipython().run_cell_magic('time', '', '\nfrom gensim.models import word2vec\n\n# Set values for various parameters\nfeature_size = 10  \nwindow_context = 5  \nmin_word_count = 1  \n\nw2v_model = word2vec.Word2Vec(\n    corpus_tokens,\n    size=feature_size,        # Word embeddings dimensionality\n    window=window_context,    # Context window size\n    min_count=min_word_count, # Minimum word count\n    sg=1,                     # `1` for skip-gram; otherwise CBOW.\n    seed = 123,               # random seed\n    workers=1,                # number of cores to use\n    negative = 5,             # how many negative samples should be drawn\n    cbow_mean = 1,            # whether to use the average of context word embeddings or sum(concat)\n    iter=10000,               # number of epochs for the entire corpus\n    batch_words=10000,        # batch size\n)')


# ### Visualizing Word Embeddings
# 
# - Embeddings represent words in multidimensional space.
# - We can inspect the quality of embeddings using dimensional reduction and visualize words in a 2D plot.

# In[7]:


from sklearn.manifold import TSNE

words = w2v_model.wv.index2word ## get the word forms of voculary
wvs = w2v_model.wv[words] ## get embeddings of all word forms

tsne = TSNE(n_components=2, random_state=0, n_iter=5000, perplexity=5)
np.set_printoptions(suppress=True)
T = tsne.fit_transform(wvs)
labels = words

plt.figure(figsize=(12, 6))
plt.scatter(T[:, 0], T[:, 1], c='orange', edgecolors='r')
for label, x, y in zip(labels, T[:, 0], T[:, 1]):
    plt.annotate(label,
                 xy=(x + 1, y + 1),
                 xytext=(0, 0),
                 textcoords='offset points')


# - All trained word embeddings are included in `w2v_model.wv`.
# - We can extract all word forms in the vocabulary from `w2v_model.wv.index2word`.
# - We can easily extract embeddings for any specific words from `w2v_model.wv`.

# In[8]:


w2v_model.wv.index2word[:5]


# In[9]:


[w2v_model.wv[w] for w in w2v_model.wv.index2word[:5]]


# ### From Word Embeddings to Document Embeddings
# 
# - With word embeddings, we can compute the **average embeddings** for the entire document, i.e., the ***document embeddings***.
# - These document embeddings are also assumed to have included considerable semantic information of the document.
# - We can for example use them for document classification/clustering.

# In[10]:


def average_word_vectors(words, model, vocabulary, num_features):

    feature_vector = np.zeros((num_features, ), dtype="float64")
    nwords = 0.

    for word in words:
        if word in vocabulary:
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model[word])

    if nwords:
        feature_vector = np.divide(feature_vector, nwords)

    return feature_vector


def averaged_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index2word)
    features = [
        average_word_vectors(tokenized_sentence, model, vocabulary,
                             num_features) for tokenized_sentence in corpus
    ]
    return np.array(features)


# In[11]:


w2v_feature_array = averaged_word_vectorizer(corpus=corpus_tokens,
                                             model=w2v_model,
                                             num_features=feature_size)
pd.DataFrame(w2v_feature_array, index=corpus_norm)


# - Let's cluster these documents based on their **document embeddings**.

# In[12]:


from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

similarity_doc_matrix = cosine_similarity(w2v_feature_array)
similarity_doc_df = pd.DataFrame(similarity_doc_matrix)
similarity_doc_df


# In[13]:


from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(similarity_doc_matrix, 'ward')
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data point')
plt.ylabel('Distance')
dendrogram(Z,
           labels=corpus_norm,
           leaf_rotation=0,
           leaf_font_size=8,
           orientation='right',
           color_threshold=0.5)
plt.axvline(x=0.5, c='k', ls='--', lw=0.5)


# In[14]:


## Other Clustering Methods

from sklearn.cluster import AffinityPropagation

ap = AffinityPropagation()
ap.fit(w2v_feature_array)
cluster_labels = ap.labels_
cluster_labels = pd.DataFrame(cluster_labels, columns=['ClusterLabel'])
pd.concat([corpus_df, cluster_labels], axis=1)

## PCA Plotting
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=0)
pcs = pca.fit_transform(w2v_feature_array)
labels = ap.labels_
categories = list(corpus_df['Category'])
plt.figure(figsize=(8, 6))

for i in range(len(labels)):
    label = labels[i]
    color = 'orange' if label == 0 else 'blue' if label == 1 else 'green'
    annotation_label = categories[i]
    x, y = pcs[i]
    plt.scatter(x, y, c=color, edgecolors='k')
    plt.annotate(annotation_label,
                 xy=(x + 1e-4, y + 1e-3),
                 xytext=(0, 0),
                 textcoords='offset points')


# ## Using Pre-trained Embeddings:  GloVe in `spacy`

# In[15]:


import spacy


nlp = spacy.load('en_core_web_lg',disable=['parse','entity'])

total_vectors = len(nlp.vocab.vectors)
print('Total word vectors:', total_vectors)


# In[16]:


print(spacy.__version__)


# ### Visualize GloVe word embeddings
# 
# - Let's extract the GloVe pretrained embeddings for all the words in our simple corpus.
# - And we visualize their embeddings in a 2D plot via dimensional reduction.

# :::{warning}
# When using pre-trained embeddings, there are two important things:
# - Be very careful of the **tokenization** methods used in your text preprocessing. If you use a very different word tokenization method, you may find a lot of **unknown** words that are not included in the pretrained model.
# - Always check the **proportion of the unknown words** when vectorizing your corpus texts with pre-trained embeddings.
# :::

# In[17]:


# get vocab of the corpus
unique_words = set(sum(corpus_tokens,[]))

# extract pre-trained embeddings of all words
word_glove_vectors = np.array([nlp(word).vector for word in unique_words])
pd.DataFrame(word_glove_vectors, index=unique_words)


# In[18]:


from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=0, n_iter=5000, perplexity=5)
np.set_printoptions(suppress=True)
T = tsne.fit_transform(word_glove_vectors)
labels = unique_words

plt.figure(figsize=(12, 6))
plt.scatter(T[:, 0], T[:, 1], c='orange', edgecolors='r')
for label, x, y in zip(labels, T[:, 0], T[:, 1]):
    plt.annotate(label,
                 xy=(x + 1, y + 1),
                 xytext=(0, 0),
                 textcoords='offset points')
    


# - It is clear to see that when embeddings are trained based on a larger corpus, they reflect more lexical semantic contents.
# - Semantically similar words are indeed closer to each other in the 2D plot.

# - We can of course perform the document-level clustering again using the GloVe embeddings.
# - The good thing about `spacy` is that it can compute the document average embeddings automatically.

# In[19]:


doc_glove_vectors = np.array([nlp(str(doc)).vector for doc in corpus_norm])

import sklearn
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=0)
km.fit_transform(doc_glove_vectors)
cluster_labels = km.labels_
cluster_labels = pd.DataFrame(cluster_labels, columns=['ClusterLabel'])
pd.concat([corpus_df, cluster_labels], axis=1)


# ## `fasttext`
# 
# - This section shows a quick example how to train word embeddings based on the `nltk.corpus.brown` using another algorithm, i.e., `fasttext`.
# - The FastText model was introduced by Facebook in 2016 as an improved and extended version of the `word2vec` (See Bojanowski et al [2017] in References below).
# - We will focus more on the implementation. Please see the Bojanowski et al (2017) as well as Sarkar (2019) Chapter 4 for more comprehensive descriptions of the method.
# - Pretrained FastText Embeddings are available [here](https://fasttext.cc/docs/en/english-vectors.html).

# In[20]:


from gensim.models.fasttext import FastText
from nltk.corpus import brown

brown_tokens = [brown.words(fileids=f) for f in brown.fileids()]


# In[21]:


get_ipython().run_cell_magic('time', '', '# Set values for various parameters\nfeature_size = 100  # Word vector dimensionality\nwindow_context = 5  # Context window size\nmin_word_count = 5  # Minimum word count\n\nft_model = FastText(brown_tokens,\n                    size=feature_size,\n                    window=window_context,\n                    min_count=min_word_count,\n                    sg=1,\n                    iter=50)')


# - We can use the trained embeddings model to identify words that are similar to a set of seed words.
# - And then we plot all these words (i.e., the seed words and their semantic neighbors) in one 2D plot based on the dimensional reduction of their embeddings.

# In[22]:


# view similar words based on gensim's model
similar_words = {
    search_term:
    [item[0] for item in ft_model.wv.most_similar([search_term], topn=5)]
    for search_term in
    ['think', 'say','news', 'report','nation', 'democracy']
}
similar_words


# In[23]:


from sklearn.decomposition import PCA

words = sum([[k] + v for k, v in similar_words.items()], [])
wvs = ft_model.wv[words]

pca = PCA(n_components=2)
np.set_printoptions(suppress=True)
P = pca.fit_transform(wvs)
labels = words

plt.figure(figsize=(12, 10))
plt.scatter(P[:, 0], P[:, 1], c='lightgreen', edgecolors='g')
for label, x, y in zip(labels, P[:, 0], P[:, 1]):
    plt.annotate(label,
                 xy=(x + 0.03, y + 0.03),
                 xytext=(0, 0),
                 textcoords='offset points')


# In[24]:


ft_model.wv['democracy']


# In[25]:


print(ft_model.wv.similarity(w1='taiwan', w2='freedom'))
print(ft_model.wv.similarity(w1='china', w2='freedom'))


# ## Wrap-up
# 
# - Two fundamental deep-learning-based models of word representation learning: CBOW and Skip-Gram.
# - From word embeddings to document embeddings
# - More advanced representation learning models: GloVe and FastText.
# - What is more challenging is how to assess the quality of the learned representations (embeddings). Usually embedding models can be evaluated based on their performance on semantics related tasks, such as word similarity and analogy. For those who are interested, you can start with the following two papers on Chinese embeddings:
#     - Chi-Yen Chen, Wei-Yun Ma. 2018. "[Word Embedding Evaluation Datasets and Wikipedia Title Embedding for Chinese](http://www.lrec-conf.org/proceedings/lrec2018/pdf/159.pdf)," Language Resources and Evaluation Conference. 
#     - Chi-Yen Chen, Wei-Yun Ma. 2017. "[Embedding Wikipedia Title Based on Its Wikipedia Text and Categories](https://ieeexplore.ieee.org/document/8300566)," International Conference on Asian Language Processing.
# 

# ## References
# 
# - Sarkar (2020) Ch 4 Feature Engineering for Text Representation
# - Major Readings:
#     - Harris,Zellig. 1956. [Distributional structure](http://www.tandfonline.com/doi/pdf/10.1080/00437956.1954.11659520).
#     - Bengio, Yoshuan, et. al. 2003. [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf).
#     - Collobert, Ronana and Jason Weston. 2008. [A Unified Architecture for Natural Language Processing: Deep Neural Networks with Multitask Learning](https://ronan.collobert.com/pub/matos/2008_nlp_icml.pdf).
#     - Schwenk, Holger. 2007.[Continuous space language models](https://pdfs.semanticscholar.org/0fcc/184b3b90405ec3ceafd6a4007c749df7c363.pdf).
#     - Mikolov, Tomas, et al. 2013. [Efficient estimation of word representations in vector space](https://arxiv.org/abs/1301.3781). arXiv preprint arXiv:1301.3781. 
#     - Mikolov, Tomas, et al. 2013. [Distributed representations of words and phrases and their compositionally](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf). *Advances in neural information processing systems*. 2013.
#     - Baroni, Marco, et. al. 2014. [Don’t count, predict! A systematic comparison of context-counting vs. context-predicting semantic vectors](https://www.aclweb.org/anthology/P14-1023/). *ACL*(1).
#     - Pennington, Jeffrey, et al. 2014. [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf). *EMNLP*. Vol. 14.
#     - Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). [Enriching word vectors with subword information](https://doi.org/10.1162/tacl_a_00051). *Transactions of the Association for Computational Linguistics*, 5, 135-146.
# - [GloVe Project Official Website](https://nlp.stanford.edu/projects/glove/): You can download their pre-trained GloVe models.
# - [FastText Project Website](https://fasttext.cc/docs/en/english-vectors.html): You can download the English pre-trained FastText models.
# 
