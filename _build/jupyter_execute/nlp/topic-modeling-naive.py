#!/usr/bin/env python
# coding: utf-8

# # Topic Modeling: A Naive Example

# ## What is Topic Modeling?

# - Topic modeling is an **unsupervised** learning method, whose objective is to extract the underlying semantic patterns among a collection of texts. These underlying semantic structures are commonly referred to as **topics** of the corpus.
# - In particular, topic modeling first extracts features from the words in the documents and use mathematical structures and frameworks like matrix factorization and SVD (Singular Value Decomposition) to identify clusters of words that share greater semantic coherence.
# - These clusters of words form the notions of topics.
# - Meanwhile, the mathematical framework will also determine the distribution of these **topics** for each document.
# 

# - In short, an intuitive understanding of Topic Modeling:
#     - Each **document** consists of several **topics** (a distribution of different topics).
#     - Each topic is connected to particular groups of **words** (a distribution of different words).

# ## Flowchart for Topic Modeling

# ![](../images/topic-modeling-pipeline.jpeg)

# ## Data Preparation and Preprocessing

# ### Import Necessary Dependencies and Settings

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt

pd.options.display.max_colwidth = 200
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Simple Corpus
# 

# - We will be using again a simple corpus for illustration.
# - It is a corpus consisting of eight documents, each of which is a sentence.

# In[2]:


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


# ### Simple Text Pre-processing

# - Depending on the nature of the raw corpus data, we may need to implement more specific steps in text preprocessing.
# - In our current naive example, we consider:
#     - removing symbols and punctuations
#     - normalizing the letter case
#     - stripping unnecessary/redundant whitespaces
#     - removing stopwords (which requires an intermediate tokenization step)
# 

# :::{tip}
# 
# Other important considerations in text preprocessing include:
# - whether to remove hyphens
# - whether to lemmatize word forms
# - whether to stemmatize word forms
# - whether to remove short word tokens
# - whether to remove unknown words (e.g., words not listed in WordNet)
# 
# :::

# In[3]:


wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')


def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I | re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokeanize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc


normalize_corpus = np.vectorize(normalize_document)


# In[4]:


norm_corpus = normalize_corpus(corpus)
norm_corpus


# - The `norm_corpus` will be the input for our next step, text vectorization.

# ## Text Vectorization

# ### Bag of Words Model

# - In topic modeling, the simplest way of text vectorization is to adopt the feature-based Bag-of-Words model.
# - Recap of the characteristics of BOW model
#     - It is a naive way to vectorize texts into numeric representations using their word frequency lists
#     - The sequential order of words in the text is naively ignored.
#     - We can filter the document-by-word matrix in many different ways (Please see the lecture notes on [Lecture Notes: Text Vectorization](../nlp/text-vec-traditional.ipynb)
# - Please use the **count-based** vectorizer for topic modeling because most of the topic modeling algorithms will take care of the weightings automatically during the mathematical computing.

# In[5]:


from sklearn.feature_extraction.text import CountVectorizer
# get bag of words features in sparse format
cv = CountVectorizer(min_df=0., max_df=1.)
cv_matrix = cv.fit_transform(norm_corpus)
cv_matrix


# In[6]:


# view dense representation
# warning might give a memory error if data is too big
cv_matrix = cv_matrix.toarray()
cv_matrix


# In[7]:


# get all unique words in the corpus
vocab = cv.get_feature_names()
# show document feature vectors
pd.DataFrame(cv_matrix, columns=vocab)


# ## Latent Dirichlet Allocation

# ### Intuition of LDA

# - Latent Dirichlet [diʀiˈkleː] Allocation learns the relationships between **words**, **topics**, and **documents** by assuming documents are generated by a particular probabilistic model.
# - A topic in LDA is a multinomial distribution over the words in the vocabulary of the corpus. (That is, given a topic, it's more likely to see specific sets of words).

# - What LDA gives us is:
#     - Which words are more likely to be connected to specific topics? (Topic by Word Matrix)
#     - Which topics are more likely to be connected to specific documents? (Document by Topic Matrix)
#     

# - To interpret a topic in LDA, we examine the ranked list of the most probable (top N) words in that topic.
# - Common words in the corpus often appear near the top of the words for each topic, which makes it hard to differentiate the meanings of these topics sometimes.
# - When inspecting the word rankings for each topic, we can utilize two types of information provided by LDA:
#     - The frequencies of the words under each topic
#     - The exclusivity of the words to the topic (i.e., the degree to which the word appears in that particular topic to the exclusion of others).

# ### Building LDA Model

# - We should use `CountVectorizer` when fitting LDA instead of `TfidfVectorizer` because LDA is based on term count and document count. 
# - Fitting LDA with `TfidfVectorizer` will result in rare words being dis-proportionally sampled. 
# - As a result, they will have greater impact and influence on the final topic distribution.

# In[8]:


get_ipython().run_cell_magic('time', '', 'from sklearn.decomposition import LatentDirichletAllocation\n\nlda = LatentDirichletAllocation(n_components=3, max_iter=10000, random_state=0)\ndoc_topic_matrix = lda.fit_transform(cv_matrix)\n')


# ### Model Performance Metrics

# - We can diagnose the model performance using **perplexity** and **log-likelihood**.
#     - The higher the log-likelihood, the better.
#     - The lower the perplexity, the better.

# In[9]:


# log-likelihood
print(lda.score(cv_matrix))
# perplexity
print(lda.perplexity(cv_matrix))


# ## Interpretation

# - To properly interpret the results provided by LDA, we need to get two important matrices:
#     - **Document-by-Topic** Matrix: This is the matrix returned by the `LatentDirichletAllocation` object when we `fit_transform()` the model with the data.
#     - **Word-by-Topic** Matrix: We can retrieve this matrix from a fitted `LatentDirichletAllocation` object. i.e., `LatentDirichletAllocation.components_`

# ### Document-by-Topic Matrix

# - In **Document-by-Topic** matrix, we can see how each document in the corpus (**row**) is connected to each **topic**.
# - In particular, the numbers refer to the probability value of a specific document being connected to a particular topic.

# In[10]:


## doc-topic matrix
doc_topic_df = pd.DataFrame(doc_topic_matrix, columns=['T1', 'T2', 'T3'])
doc_topic_df


# ### Topic-by-Word Matrix

# - In **Topic-by-Word** matrix, we can see how each topic (**row**) is connected to each word in the BOW.
# - In particular, the numbers refer to the importance of the word with respect to each topic.

# In[11]:


topic_word_matrix = lda.components_


# In[12]:


pd.DataFrame(topic_word_matrix, columns=vocab)


# - We can transpose the matrix for clarity of inspection.

# In[13]:


pd.DataFrame(np.transpose(topic_word_matrix), index=vocab)


# ### Interpreting the Meanings of Topics

# - This is the most crucial step in topic modeling. The LDA does not give us a label for each topic.
# - It is the analyst who determines the **meanings** of the topics.
# - These decisions are based on the words under each topic that show high importance weights.

# In[14]:


## This function sorts the words importances under each topic
## and the selectional criteria include (a) ranks based on weights, or (b) cutoff on weights
def get_topics_meanings(tw_m,
                        vocab,
                        display_weights=False,
                        topn=5,
                        weight_cutoff=0.6):
    for i, topic_weights in enumerate(tw_m):  ## for each topic row
        topic = [(token, np.round(weight, 2))
                 for token, weight in zip(vocab, topic_weights)
                 ]  ## zip (word, importance_weight)
        topic = sorted(topic,
                       key=lambda x: -x[1])  ## rank words according to weights
        if display_weights:
            topic = [item for item in topic if item[1] > weight_cutoff
                     ]  ## output words whose weights > 0.6
            print(f"Topic #{i} :\n{topic}")
            print("=" * 20)
        else:
            topic_topn = topic[:topn]
            topic_topn = ' '.join([word for word, weight in topic_topn])
            print(f"Topic #{i} :\n{topic_topn}")
            print('=' * 20)


# - To use the above function:
#   - If we are to display the weights of words, then we need to specify the `weight_cutoff`.
#   - If we are to display only the top N words, then we need to specify the `topn`.

# In[37]:


get_topics_meanings(topic_word_matrix,
                    vocab,
                    display_weights=True,
                    weight_cutoff=2)


# In[35]:


get_topics_meanings(topic_word_matrix, vocab, display_weights=False, topn=3)


# ## Topics in Documents

# - After we determine the meanings of the topics, we can now analyze how each document is connected to these topics.
# - That is, we can now look at the **Document-by-Topic** Matrix.

# In[17]:


topics = ['weather', 'food', 'animal']
doc_topic_df.columns = topics
doc_topic_df['corpus'] = norm_corpus
doc_topic_df


# - We can visualize the topics distribution for each document using stack plot.

# In[18]:


x_axis = ['DOC' + str(i) for i in range(len(norm_corpus))]
y_axis = doc_topic_df[['weather', 'food', 'animal']]

fig, ax = plt.subplots(figsize=(15, 8))

# Plot a stackplot - https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/stackplot_demo.html
ax.stackplot(x_axis, y_axis.T, baseline='wiggle', labels=y_axis.columns)

# Move the legend off of the chart
ax.legend(loc=(1.04, 0))


# ## Clustering documents using topic model features

# In[19]:


from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, random_state=0)
km.fit_transform(doc_topic_matrix)
cluster_labels = km.labels_
cluster_labels = pd.DataFrame(cluster_labels, columns=['ClusterLabel'])
pd.concat([corpus_df, cluster_labels], axis=1)


# ## Visualizing Topic Models

# In[20]:


import pyLDAvis
import pyLDAvis.sklearn
import dill
import warnings
warnings.filterwarnings('ignore')

pyLDAvis.enable_notebook()


# In[21]:


cv_matrix = cv.fit_transform(norm_corpus)
pyLDAvis.sklearn.prepare(lda, cv_matrix, cv, mds='mmds')


# ## Hyperparameter Tuning

# - One thing we haven't made explicit is that the **number of topics** so far has been pre-determined. 
# - How to find the optimal number of topics can be challenging in topic modeling.
# - We can take this as a hyperparameter of the model and use **Grid Search** to find the most optimal number of topics.
# - Similarly, we can fine tune the other hyperparameters of LDA as well (e.g., `learning_decay`).

# - `learning_method`: The default is `batch`; that is, use all training data for parameter estimation. If it is `online`, the model will update the parameters on a token by token basis.
# - `learning_decay`: If the `learning_method` is `online`, we can specify a parameter that controls learning rate in the online learning method (usually set between (0.5, 1.0]). 

# :::{tip}
# 
# Doing Grid Search with LDA models can be very slow. There are some other topic modeling algorithms that are a lot faster. Please refer to Sarkar (2019) Chapter 6 for more information.
# 
# :::

# ### Grid Search for Topic Number

# In[22]:


get_ipython().run_cell_magic('time', '', "from sklearn.decomposition import LatentDirichletAllocation\nfrom sklearn.model_selection import GridSearchCV\n\n# Options to try with our LDA\n# Beware it will try *all* of the combinations, so it'll take ages\nsearch_params = {'n_components': range(3,8), 'learning_decay': [.5, .7]}\n\n# Set up LDA with the options we'll keep static\nmodel = LatentDirichletAllocation(learning_method='online', ## `online` for large datasets\n                                  max_iter=10000,\n                                  random_state=0)\n\n# Try all of the options\ngridsearch = GridSearchCV(model,\n                          param_grid=search_params,\n                          n_jobs=-1,\n                          verbose=1)\ngridsearch.fit(cv_matrix)\n\n## Save the best model\nbest_lda = gridsearch.best_estimator_\n")


# In[23]:


# What did we find?
print("Best Model's Params: ", gridsearch.best_params_)
print("Best Log Likelihood Score: ", gridsearch.best_score_)
print('Best Model Perplexity: ', best_lda.perplexity(cv_matrix))


# ### Examining the Grid Search Results

# In[24]:


cv_results_df = pd.DataFrame(gridsearch.cv_results_)
cv_results_df


# In[25]:


import seaborn as sns
sns.set(rc={"figure.dpi":150, 'savefig.dpi':150})
sns.pointplot(x="param_n_components",
              y="mean_test_score",
              hue="param_learning_decay",
              data=cv_results_df)


# In[38]:


get_topics_meanings(best_lda.components_,
                    vocab,
                    display_weights=True,
                    weight_cutoff=2)


# ## Topic Prediction

# - We can use our LDA to make predictions of topics for new documents.

# In[39]:


new_texts = ['The sky is so blue', 'Love burger with ham']

new_texts_norm = normalize_corpus(new_texts)
new_texts_cv = cv.transform(new_texts_norm)
new_texts_cv.shape


# In[40]:


new_texts_doc_topic_matrix = best_lda.transform(new_texts_cv)
topics = ['weather', 'food', 'animal']
new_texts_doc_topic_df = pd.DataFrame(new_texts_doc_topic_matrix,
                                      columns=topics)
new_texts_doc_topic_df['predicted_topic'] = [
    topics[i] for i in np.argmax(new_texts_doc_topic_df.values, axis=1)
]

new_texts_doc_topic_df['corpus'] = new_texts_norm
new_texts_doc_topic_df


# ## Additional Notes

# - We can calculate a metric to evaluate the coherence of each topic.
# - The coherence computation is implemented in `gensim`. To apply the coherence comptuation to a `sklearn`-trained LDA, we need `tmtoolkit` (`tmtoolkit.topicmod.evaluate.metric_coherence_gensim`).
# 
# - I leave notes here in case in the future we need to compute the coherence metrics.
# 
# :::{warning}
# 
# `tmtoolkit` does not support `spacy` 3+. Also, `tmtoolkit` will downgrade several important packages to lower versions. Please use it with caution. I would suggest creating another virtual environment for this.
# 
# :::

# - The following codes demonstrate how to find the optimal topic number based on the coherence scores of the topic models.

# In[29]:


import tmtoolkit
from tmtoolkit.topicmod.evaluate import metric_coherence_gensim
def topic_model_coherence_generator(topic_num_start=2,
                                    topic_num_end=6,
                                    norm_corpus='',
                                    cv_matrix='',
                                    cv=''):
    norm_corpus_tokens = [doc.split() for doc in norm_corpus]
    models = []
    coherence_scores = []

    for i in range(topic_num_start, topic_num_end):
        print(i)
        cur_lda = LatentDirichletAllocation(n_components=i,
                                            max_iter=10000,
                                            random_state=0)
        cur_lda.fit_transform(cv_matrix)
        cur_coherence_score = metric_coherence_gensim(
            measure='c_v',
            top_n=5,
            topic_word_distrib=cur_lda.components_,
            dtm=cv.fit_transform(norm_corpus),
            vocab=np.array(cv.get_feature_names()),
            texts=norm_corpus_tokens)
        models.append(cur_lda)
        coherence_scores.append(np.mean(cur_coherence_score))
    return models, coherence_scores


# In[30]:


get_ipython().run_cell_magic('time', '', 'ts = 2\nte = 10\nmodels, coherence_scores = topic_model_coherence_generator(\n    ts, te, norm_corpus=norm_corpus, cv=cv, cv_matrix=cv_matrix)\n')


# In[31]:


coherence_scores


# In[32]:


coherence_df = pd.DataFrame({
    'TOPIC_NUMBER': [str(i) for i in range(ts, te)],
    'COHERENCE_SCORE': np.round(coherence_scores, 4)
})

coherence_df.sort_values(by=["COHERENCE_SCORE"], ascending=False)


# In[33]:


import plotnine
from plotnine import ggplot, aes, geom_point, geom_line, labs
plotnine.options.dpi = 150

g = (ggplot(coherence_df) + aes(x="TOPIC_NUMBER", y="COHERENCE_SCORE") +
     geom_point(stat="identity") + geom_line(group=1, color="lightgrey") +
     labs(x="Number of Topics", y="Average Coherence Score"))
g


# ## References
# 
# - Sarkar (2019), Chapter 6: Text Summarization and Topic Models
