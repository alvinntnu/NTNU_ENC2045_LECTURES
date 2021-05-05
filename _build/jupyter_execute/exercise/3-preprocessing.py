#!/usr/bin/env python
# coding: utf-8

# # Assignment III: Preprocessing

# ## Question 1
# 
# Please use the `grutenberg` corpus provided in `nltk` and extract the text written by Lewis Caroll, i.e., `fileid == 'carroll-alice.txt'`, as your corpus data.
# 
# With this corpus data, please perform text preprocessing on the **sentences** of the corpus.
# 
# In particular, please:
# 
# - pos-tag all the sentences to get the parts-of-speech of each word
# - lemmatize all words using `WordNetLemmatizer` in NLTK on a sentential basis
# 
# Please provide your output as shown below:
# 
# - it is a data frame
# - the column `alice_sents` includes the original sentence texts
# - the column `alice_sents_pos` includes annotated version of the sentences with each token as `word/postag` 
# - the column `sents_lem` includes the lemmatized version of the sentences
# 
# 
# ```{note}
# Please note that the lemmatized form of the BE verbs (e.g., *was*) should be *be*. This is a quick check if your lemmatization works successfully.
# ```
# 

# In[2]:


alice_sents_df[:20]


# ## Question 2
# 
# Based on the output of **Question 1**, please create a lemma frequency list of the corpus, `carroll-alice.txt`, using the lemmatized forms by including only lemmas which are:
# - consisting of only alphabets or hyphens
# - at least 5-character long
# 
# The casing is irrelevant (i.e., case normalization is needed).
# 
# The expected output is provided as follows (showing the top 20 lemmas and their frequencies).
# 

# In[4]:


# top 20
alice_df[:21]


# ## Question 3
# 
# Please identify top verbs that co-occcur with the name *Alice* in the text, with *Alice* being the **subject** of the verb. 
# 
# Please use the `en_core_web_sm` model in `spacy` for English dependency parsing.
# 
# To simply the task, please identify all the verbs that have a dependency relation of `nsubj` with the noun `Alice` (where `Alice` is the **dependent**, and the verb is the **head**).
# 
# The expected output is provided below (showing the top 20 heads of `Alice` for the `nsubj` dependency relation.)

# In[6]:


alice_nsubj_df[:21]

