#!/usr/bin/env python
# coding: utf-8

# # Text Tokenization

# The objective of text tokenization is to break the text into smaller units which are often more linguistically meaningful.
# 
# These smaller linguistic units are usually easier to deal with computationally and semantically.

# ## Sentence Tokenization

# In[1]:


from nltk.tokenize import sent_tokenize


# In[2]:


para = '''There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, “Oh dear! Oh dear! I shall be late!” (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge.'''

for s in sent_tokenize(para):
    print(s+'\n')


# - The `sent_tokenize()` function uses an instance of `PunktSentenceTokenizer` from the `ntlk.tokenize.punkt` module. 
# 
# - To process large amount of data, it is recommended to load the pre-trained `PunktSentenceTokenizer` once, and call its `tokenizer()` method for the task.

# In[3]:


import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle')


# In[4]:


tokenizer.tokenize(para)


# The `nltk` also provides many pre-trained `PunktSentenceTokenizer` for other European languages.

# In[16]:


get_ipython().system('ls /Users/Alvinchen/nltk_data/tokenizers/punkt/PY3')
# !dir C:/Users/alvinchen/AppData/Roaming/nltk_data/tokenziers/punkt/PY3


# ## Word Tokenization

# Similarly, the `word_tokenize()` function is a wrapper function that calls the `tokenize()` method on a instance of `TreebankWordTokenizer` class.

# In[6]:


from nltk.tokenize import word_tokenize
print(word_tokenize(para)[:20])


# - To process large amount of data, please create an instance of `TreebankWordTokenizer` and call its `tokenize()` method for more efficient processing.
# 
# - We will get the same results with the following codes as above.

# In[7]:


from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()

print(tokenizer.tokenize(para)[:10])


# The `nltk` module has implemented other more task-oriented word tokenizers, which differ in terms of their specific handling of the punctuations and contractions.

# ![](../images/nltk-tokenizer-class.png)

# ### Comparing different word tokenizers

# - `TreebankWordTokenizer` follows the Penn Treebank conventions for word tokenization.
# - `WordPunctTokenizer` splits all punctuations into separate tokens.

# In[8]:


from nltk.tokenize import WordPunctTokenizer
wpt = WordPunctTokenizer()
tbwt = TreebankWordTokenizer()

sent = "Isn't this great? I can't tell!"


# In[9]:


wpt.tokenize(sent)


# In[10]:


tbwt.tokenize(sent)


# ## Tokenization using regular expressions

# The `nltk` also provides another flexible way for text tokenization based on regular expression.
# 
# The `RegexTokenizer` class allows for text tokenization based on the self-defined regular expression patterns.
# 
# The regular expression can be created/defined for either the token or the delimiter.

# In[11]:


from nltk.tokenize import RegexpTokenizer


# In[12]:



retok1 = RegexpTokenizer(pattern= "[a-zA-Z_'-]+")
retok2 = RegexpTokenizer(pattern= "[a-zA-Z_-]+")
retok3 = RegexpTokenizer(pattern= "\s+", gaps=True)


# In[13]:


print(retok1.tokenize(sent))


# See how this tokenizer deals with the apostrophe?

# In[14]:


print(retok2.tokenize(sent))


# In[15]:


print(retok3.tokenize(sent))


# ## References

# - NLTK Book, Ch 1-3
