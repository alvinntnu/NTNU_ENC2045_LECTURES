#!/usr/bin/env python
# coding: utf-8

# # Midterm Exam (Sample Solutions)

# ## Question 1-1
# 
# ----
# 

# In[1]:


## Loading packages
import re
import nltk
from nltk.corpus import PlaintextCorpusReader
import pandas as pd
import unicodedata
import re

## Notebook Settings
pd.options.display.max_colwidth = 200


# In[2]:


## Loading corpus files into one CSV 

# jay_dir = 'midterm_inputdata/jay/'
# jay_corpus = PlaintextCorpusReader(jay_dir,'.*\.txt')

# jay = pd.DataFrame(
#     [(re.sub(r'\.txt$','',f), jay_corpus.raw(f)) for f in jay_corpus.fileids()],
#     columns=['title','lyric'])


# In[3]:


## Loading CSV (from the original CSV)
jay = pd.read_csv('midterm_inputdata/jay.csv')
jay.head(10)


# In[4]:


## Preprocessing Function
## remove extra linebreaks whitespaces and unicode category punctuations and symbols
def preprocess(doc):
    doc = re.sub(r'\n+','\n', doc)
    doc = ''.join([c if unicodedata.category(c)[0] not in ["P", "S", "N"] else ' ' for c in doc]) ## symbols
    #doc= re.sub(r'[0-9a-zA-Z]+'," ", doc) ## remove english letters and numbers
    doc = ''.join([c if unicodedata.category(c) not in ["Ll", "Lu"] else ' ' for c in doc]) ## alphabets
    doc = re.sub(r'[ \u3000]+', ' ', doc)
    doc = '\n'.join([line.strip() for line in doc.split('\n')])
    return doc

## Check preprocessed results
x= list(jay.lyric)[198]
print(x)
print("="*10)
print(preprocess(x))


# In[5]:


## Preprocess the corpus
jay['lyric_pre'] = [preprocess(l) for l in jay.lyric]


# In[6]:


jay.iloc[[100,200],:]


# ## Question 1-2
# 
# -----
# 

# In[7]:


## packages
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib
import matplotlib.pyplot as plt

## plotting settings
plt.style.use('ggplot')
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.sans-serif']=["PingFang HK"] ## set ur own chinese font


# In[8]:


# ##############################################
# ## Uncommment this when word seg is needed####
# ##############################################

# import ckip_transformers
# from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger
# #Initialize drivers
# ws_driver = CkipWordSegmenter(level=3, device=-1)
# pos_driver = CkipPosTagger(level=3, device=-1)

# def my_tokenizer(doc):
#     # `doc`: a list of corpus documents (each element is a document long string)
#     cur_ws = ws_driver(doc, use_delim = True, delim_set='\n')
#     cur_pos = pos_driver(cur_ws)
#     doc_seg = [[(x,y) for (x,y) in zip(w,p)]  for (w,p) in zip(cur_ws, cur_pos)]
#     return doc_seg


# In[9]:


# %%time

# ##############################################
# ## Uncommment this when word seg is needed####
# ##############################################

## Perform word seg in Google Colab
## It takes about 40s in Google Colab

# jay_lyric_wordseg = my_tokenizer(list(jay.lyric_pre))

# import pickle
# with open('midterm-jay-lyric-wordseg.pickle', 'wb') as f:
#     pickle.dump(jay_lyric_wordseg, f, protocol=pickle.HIGHEST_PROTOCOL)


# In[10]:


with open('midterm-jay-lyric-wordseg.pickle', 'rb') as f:
    jay_lyric_wordseg = pickle.load(f)


# In[11]:


fileids = list(jay.title)


# In[12]:


## select words whose POS starts with N or V but NOT pronouns (Nh) or numbers (Neu)
jay_words = [[(w,p) for (w,p) in text if re.match(r'^[NV](?!(h|eu))',p)] for text in jay_lyric_wordseg]
jay_norm = [' '.join([w for w,p in text]) for text in jay_words]


# In[13]:


## use words len >=2
cv = CountVectorizer(token_pattern=r'[^\s]{2,}', min_df = 2)
jay_bow = cv.fit_transform(jay_norm)
jay_array = jay_bow.toarray()

# show document feature vectors
# arrange data frame according to column sums
jay_bow_df = pd.DataFrame(jay_array, columns=cv.get_feature_names(), index = fileids)
s = jay_bow_df.sum()
# jay_bow_df[s.sort_values(ascending=False).index[:50]]


# In[14]:


print(jay_bow_df.shape)


# In[15]:


jay_bow_df


# In[16]:


tv = TfidfVectorizer(min_df=2,
                     max_df=1.0,
                     norm='l2',
                     use_idf=True,
                     smooth_idf=True,
                     token_pattern=r'[^\s]{2,}')
tv_matrix = tv.fit_transform(jay_norm)


# In[17]:


print(tv_matrix.shape)


# In[18]:


# arrange data frame according to column sums
jay_tv_df = pd.DataFrame(tv_matrix.toarray(), columns=tv.get_feature_names(), index = fileids)
s = jay_tv_df.sum()
jay_tv_df[s.sort_values(ascending=False).index[:50]].round(2)


# In[19]:


jay_tv_df.round(2)


# In[20]:


similarity_doc_matrix = cosine_similarity(tv_matrix)
similarity_doc_df = pd.DataFrame(similarity_doc_matrix, index=fileids, columns=fileids)

Z = linkage(similarity_doc_matrix, 'ward')


# In[21]:


similarity_doc_df.round(2)


# In[22]:


## Plot Dendrogram
plt.figure(figsize=(15, 40))
plt.title("Jay Chou Analysis")
plt.xlabel("Song Titles")
plt.ylabel('Distance')
color_threshold = 2
dendrogram(Z, labels=fileids,
           orientation = 'right',
           leaf_rotation = 0, 
           leaf_font_size= 10,
           color_threshold = color_threshold, above_threshold_color='b')
plt.axvline(x=color_threshold, c='k', ls='--', lw=0.5)
plt.tight_layout()
#plt.savefig('midterm/question1-2-output-dendrogram.jpeg',dpi=300)


# ## Question 2-1
# 
# -----
# 

# In[23]:


import nltk
import numpy as np
import random
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
pd.options.display.max_colwidth = 200


# In[24]:


## Import train and test
with open('midterm_inputdata/chinese_name_gender_train.txt', 'r') as f:
    train = [l.replace('\n','').split(',') for l in f.readlines() if len(l.split(','))==2]

with open('midterm_inputdata/chinese_name_gender_test.txt', 'r') as f:
    test = [l.replace('\n','').split(',') for l in f.readlines() if len(l.split(','))==2]

## Sentiment Distrubtion for Train and Test
print(Counter([label for (words, label) in train]))
print(Counter([label for (words, label) in test]))

X_train = [name for (name, gender) in train]
X_test = [name for (name, gender) in test]
y_train = [gender for (name, gender) in train]
y_test = [gender for (name, gender) in test]


# In[25]:


## self-defined tokenzier
def myTokenizer(text):
    ngrams=[]
    ngrams.append(text[1:])
    ngrams.append(text[1])
    ngrams.append(text[2])
    return ngrams

## text vectorization
cv = CountVectorizer(min_df = 100,
                     tokenizer=myTokenizer)

X_train_bow = cv.fit_transform(X_train)
X_test_bow=cv.transform(X_test)

print(X_train_bow.shape)
print(X_test_bow.shape)


# In[26]:


# get all unique words in the corpus
vocab = cv.get_feature_names()

# show document feature vectors
X_train_bow_df = pd.DataFrame(X_train_bow.toarray(), columns=vocab, index = X_train)


# In[27]:


X_train_bow_df.head()


# In[28]:


## Check bigram features (make sure no last name was included)
X_train_bow_df[[col for col in X_train_bow_df.columns if len(col)>1]].head()


# ## Question 2-2
# 
# ------
# 

# In[29]:


import numpy as np
import sklearn
from sklearn.metrics import f1_score,confusion_matrix, plot_confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import lime
from lime.lime_text import LimeTextExplainer


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('classic')

matplotlib.rcParams['font.sans-serif']=["PingFang HK"]
matplotlib.rcParams['figure.dpi']=300


# In[30]:


get_ipython().run_cell_magic('time', '', 'model_gnb = GaussianNB()\nmodel_gnb_acc = cross_val_score(estimator=model_gnb, X=X_train_bow.toarray(), y=y_train, cv=10, n_jobs=None)\nmodel_gnb_acc')


# In[31]:


get_ipython().run_cell_magic('time', '', 'model_lg = LogisticRegression(max_iter = 1000)\nmodel_lg_acc = cross_val_score(estimator=model_lg, X=X_train_bow, y=y_train, cv=10, n_jobs=None)\nmodel_lg_acc')


# In[32]:


print("Mean Accuracy of Naive Bayes Model: ", model_gnb_acc.mean())
print("Mean Accuracy of Logistic Regression Model:", model_lg_acc.mean())


# In[33]:


get_ipython().run_cell_magic('time', '', "## Grid Search\nparameters = {'C': (1,5,10)}\nclf = GridSearchCV(model_lg, parameters, cv=10, n_jobs=None) ## `-1` run in parallel\nclf.fit(X_train_bow, y_train)")


# In[34]:


clf.best_params_


# In[35]:


plot_confusion_matrix(clf, X_test_bow, y_test, normalize='all')
plt.title("Confusion Matrix (Normalized %)")


# In[36]:


plot_confusion_matrix(clf, X_test_bow, y_test, normalize=None)
plt.title("Confusion Matrix (Frequencies)")


# In[37]:


## Pipeline for LIME
pipeline = Pipeline([
  ('vectorizer',cv), 
  ('clf', LogisticRegression(C=10, max_iter = 1000))])
pipeline.fit(X_train, y_train)


# In[38]:


explainer = LimeTextExplainer(class_names=['女','男'],char_level=True, bow=False)
test_name = ["王貴瑜",'林育恩','張純映','陳英雲']
explanations = []
for n in test_name:
    explanations.append(explainer.explain_instance(n, pipeline.predict_proba))


# In[39]:


explanations[0].show_in_notebook(text=True)


# In[40]:


explanations[1].show_in_notebook(text=True)


# In[41]:


explanations[2].show_in_notebook(text=True)


# In[42]:


explanations[3].show_in_notebook(text=True)


# In[43]:


## Feature Importance Analysis
importances = pipeline.named_steps['clf'].coef_.flatten()

## Select top 10 positive/negative weights
top_indices_pos = np.argsort(importances)[::-1][:10]## top 10 for positve weights
top_indices_neg = np.argsort(importances)[::-1][-10:] ## bottom 10 for negative weights

## Get featnames from tfidfvectorizer
feature_names = np.array(cv.get_feature_names()) # List indexing is different from array
feature_importance_df = pd.DataFrame({'FEATURE': feature_names[np.concatenate((top_indices_pos, top_indices_neg))],
                                     'IMPORTANCE': importances[np.concatenate((top_indices_pos, top_indices_neg))],
                                     'SENTIMENT': ['pos' for _ in range(len(top_indices_pos))]+['neg' for _ in range(len(top_indices_neg))]})
feature_importance_df


# In[44]:


## Visualize feature importance

plt.style.use('ggplot')

matplotlib.rcParams['font.sans-serif']=["PingFang HK"]
matplotlib.rcParams['figure.dpi']=300
plt.figure(figsize=(8,5))
pal = sns.color_palette("viridis", len(feature_importance_df.index))
sns.barplot(x=feature_importance_df['FEATURE'], y=feature_importance_df['IMPORTANCE'], palette=np.array(pal[::-1]))
plt.title("Male Preference << --- >> Female Preference\n")
plt.savefig('midterm/_question2-2-output-featimportance.jpeg', bbox_inches='tight',dpi=300)


# ## Question 3-1
# 
# 
# ----
# 

# In[45]:


import pandas as pd
import unicodedata
import re
import nltk
import pickle
import numpy as np
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
sns.set(font_scale=0.7)
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.sans-serif']=["PingFang HK"]
pd.options.display.float_format = '{:,.2f}'.format


# In[46]:


## remove extra linebreaks whitespaces and unicode category punctuations and symbols
def preprocess(doc):
    doc = re.sub(r'\n+','\n', doc)
    doc = ''.join([c if unicodedata.category(c)[0] not in ["P", "S","N"] else ' ' for c in doc]) ## symbols
    #doc= re.sub(r'[0-9a-zA-Z]+'," ", doc) ## remove english letters and numbers
    doc = ''.join([c if unicodedata.category(c) not in ["Ll", "Lu"] else ' ' for c in doc]) 
    doc = re.sub(r'[ \u3000]+', ' ', doc)
    doc = '\n'.join([line.strip() for line in doc.split('\n')])
    return doc

apple_df = pd.read_csv('midterm_inputdata/apple5000.csv')
apple_df['text_pre'] = [preprocess(text) for text in apple_df.text]

apple_df.head()


# In[47]:


# %%time

## Spacy Parsing

# import spacy
# nlp = spacy.load("zh_core_web_lg")

# mod_head=[]

# for doc in nlp.pipe(apple_df.text_pre, n_process=-1):
#     for t in doc:
#         if (t.dep_ == "amod"):
#             mod_head.append((t.text, t.head.text))
# len(mod_head)

# import pickle
# with open('midterm-apple-mod-head-lg.pickle', 'wb') as f:
#     pickle.dump(mod_head, f, protocol=pickle.HIGHEST_PROTOCOL)


# In[48]:


with open('midterm-apple-mod-head-lg.pickle', 'rb') as f:
    mod_head=pickle.load(f)


# In[49]:


## filter two-syllable NOUNS only
mod_head_fd = nltk.FreqDist([(m,n) for (m, n) in mod_head if len(n)>=2])
mod_head_fd2 = nltk.FreqDist([m+'_'+n for (m, n) in mod_head if len(n)>=2])
mod_head_df = pd.DataFrame(list(mod_head_fd2.items()), columns = ["MOD-NOUN","Frequency"]) 


# In[50]:


mod_head_df.sort_values(['Frequency'],ascending=[False]).head(20)


# ## Question 3-2
# 
# -------

# In[51]:


## Get nouns and mods dict indices
nouns = {noun:i for i, noun in enumerate(set([head for (mod, head) in mod_head]))}
mods = {mod:i for i, mod in enumerate(set([mod for (mod, head) in mod_head]))}

## Create Noun by Modifiers Matrix
noun_by_mod = np.zeros(shape = (len(nouns), len(mods)), dtype='float32')
for ((m,n),c) in mod_head_fd.items():
    noun_by_mod[nouns[n],mods[m]] = noun_by_mod[nouns[n],mods[m]]+c


# In[52]:


print(noun_by_mod.shape)


# In[53]:


col_sum_ind = np.argsort(-noun_by_mod.sum(axis=0))
row_sum_ind = np.argsort(-noun_by_mod.sum(axis=1))

col_cut = 10
row_cut = 70
col_ind = [i for i,s in enumerate(noun_by_mod.sum(axis=0)) if s > col_cut]
row_ind = [i for i,s in enumerate(noun_by_mod.sum(axis=1)) if s > row_cut]
print(len(row_ind))
print(len(col_ind))


# In[54]:


noun_by_mod_filtered_df = pd.DataFrame(noun_by_mod, columns = mods, index=nouns).iloc[row_ind, col_ind]
print(noun_by_mod_filtered_df.shape)

similarity_noun = cosine_similarity(noun_by_mod_filtered_df)
similarity_noun_df = pd.DataFrame(similarity_noun, index=noun_by_mod_filtered_df.index, columns=noun_by_mod_filtered_df.index)


# In[55]:


plt.figure(figsize=(20,15))
cf_hm1 = sns.heatmap(similarity_noun_df, annot=True, fmt='.2f', xticklabels=similarity_noun_df.index, yticklabels=similarity_noun_df.index, linewidths=.5, linecolor='black', cmap="Greens")
plt.yticks(rotation=0)
plt.title('Pairwise Cosine Similarity')


# In[56]:


Z = linkage(similarity_noun, 'ward')
color_threshold=0.6
plt.figure(figsize=(10, 12))
plt.title("Cluster Nouns According to Their Modifiers")
plt.xlabel("Distance")
plt.ylabel('Top Nouns in Apple News')
color_threshold = 2
dendrogram(Z, labels=list(noun_by_mod_filtered_df.index),
           orientation = 'right',
           leaf_rotation = 0, 
           leaf_font_size= 10,
           color_threshold = color_threshold, above_threshold_color='b')
plt.axvline(x=color_threshold, c='k', ls='--', lw=0.5)

