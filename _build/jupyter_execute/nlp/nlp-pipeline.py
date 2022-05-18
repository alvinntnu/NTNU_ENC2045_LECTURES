#!/usr/bin/env python
# coding: utf-8

#  
#  # NLP Pipeline

# ## A General NLP Pipeline

# ![nlp-pipeline](../images/nlp-pipeline.png)

# ### Varations of the NLP Pipelines
# 
# - The process may not always be linear.
# - There are loops in between.
# - These procedures may depend on specific task at hand.

# ## Data Collection

# ### Data Acquisition: Heart of ML System
# 
# - Ideal Setting: We have everything needed.
# - Labels and Annotations
# - Very often we are dealing with less-than-ideal scenarios

# ### Less-than-ideal Scenarios
# 
# - Initial datasets with limited annotations/labels
# - Initial datasets labeled based on regular expressions or heuristics
# - Public datasets (cf. [Google Dataset Search](https://datasetsearch.research.google.com/) or [kaggle](https://www.kaggle.com/))
# - Scrape data
# - Product intervention
# - Data augmentation

# ### Data Augmentation
# 
# - It is a technique to exploit language properties to create texts that are syntactically similar to the source text data.
# - Types of strategies:
#     - synonym replacement
#     - Related word replacement (based on association metrics)
#     - Back translation
#     - Replacing entities
#     - Adding noise to data (e.g. spelling errors, random words)

# ## Text Extraction and Cleanup

# ### Text Extraction
# 
# - Extracting raw texts from the input data
#     - HTML
#     - PDF
# - Relevant vs. irrelevant information
#     - non-textual information
#     - markup
#     - metadata
# - Encoding format

# #### Extracting texts from webpages

# In[1]:


import requests 
from bs4 import BeautifulSoup
import pandas as pd
 
 
url = 'https://news.google.com/topics/CAAqJQgKIh9DQkFTRVFvSUwyMHZNRFptTXpJU0JYcG9MVlJYS0FBUAE?hl=zh-TW&gl=TW&ceid=TW%3Azh-Hant'
r = requests.get(url)
web_content = r.text
soup = BeautifulSoup(web_content,'html.parser')
title = soup.find_all('a', class_='DY5T1d')
first_art_link = title[0]['href'].replace('.','https://news.google.com',1)

#print(first_art_link)
art_request = requests.get(first_art_link)
art_request.encoding='utf8'
soup_art = BeautifulSoup(art_request.text,'html.parser')

art_content = soup_art.find_all('p')
art_texts = [p.text for p in art_content]
print(art_texts)


# #### Extracting texts from scanned PDF

# In[2]:


from PIL import Image
from pytesseract import image_to_string


YOUR_DEMO_DATA_PATH = "../../../RepositoryData/data/"  # please change your file path
filename = YOUR_DEMO_DATA_PATH+'pdf-firth-text.png'
text = image_to_string(Image.open(filename))
print(text)


# #### Unicode normalization

# In[5]:


text = 'I feel really 😡. GOGOGO!! 💪💪💪  🤣🤣 ȀÆĎǦƓ'
print(text)
text2 = text.encode('utf-8') # encode the strings in bytes
print(text2)


# In[15]:


import unicodedata
unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')


# - Please check [unicodedata documentation](https://docs.python.org/3/library/unicodedata.html) for more detail on character normalization.
# - Other useful libraries
#     - Spelling check: pyenchant, Microsoft REST API
#     - PDF:  PyPDF, PDFMiner
#     - OCR: pytesseract
#  

# ### Cleanup
# 
# - Preliminaries
#     - Sentence segmentation
#     - Word tokenization
#     

# #### Segmentation and Tokenization

# In[4]:


from nltk.tokenize import sent_tokenize, word_tokenize

text = '''
Python is an interpreted, high-level and general-purpose programming language. Python's design philosophy emphasizes code readability with its notable use of significant whitespace. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects.
'''

## sent segmentation
sents = sent_tokenize(text)

## word tokenization
for sent in sents:
    print(sent)
    print(word_tokenize(sent))


# - Frequent preprocessing
#     - Stopword removal
#     - Stemming and/or lemmatization
#     - Digits/Punctuaions removal
#     - Case normalization
#     

# #### Removing stopwords, punctuations, digits

# In[5]:


from nltk.corpus import stopwords
from string import punctuation

eng_stopwords = stopwords.words('english')

text = "Mr. John O'Neil works at Wonderland, located at 245 Goleta Avenue, CA., 74208."

words = word_tokenize(text)

print(words)

# remove stopwords, punctuations, digits
for w in words:
    if w not in eng_stopwords and w not in punctuation and not w.isdigit():
        print(w)


# #### Stemming and lemmatization

# In[6]:


## Stemming
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

words = ['cars','revolution', 'better']
print([stemmer.stem(w) for w in words])


# In[7]:


## Lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

## Wordnet requires POS of words
poss = ['n','n','a']

for w,p in zip(words,poss):
    print(lemmatizer.lemmatize(w, pos=p))


# - Task-specific preprocessing
#     - Unicode normalization
#     - Language detection
#     - Code mixing
#     - Transliteration (e.g., using piyin for Chinese words in English-Chinese code-switching texts)
#     

# - Automatic annotations
#     - POS tagging
#     - Parsing
#     - Named Entity Recognition
#     - Coreference resolution
#     

# ### Important Reminders for Preprocessing
# 
# - Not all steps are necessary
# - These steps are NOT sequential
# - These steps are task-dependent
# - Goals
#     - Text Normalization
#     - Text Tokenization
#     - Text Enrichment/Annotation

# ## Feature Engineering

# ### What is feature engineering?
# 
# - It refers to a process to feed the extracted and preprocessed texts into a machine-learning algorithm.
# - It aims at capturing the characteristics of the text into a numeric vector that can be understood by the ML algorithms. (Cf. *construct*, *operational definitions*, and *measurement* in experimental science)
# - In short, it concerns how to meaningfully represent texts quantitatively, i.e., text representation.

# ### Feature Engineering for Classical ML
# 
# - Word-based frequency lists
# - Bag-of-words representations
# - Domain-specific word frequency lists
# - Handcrafted features based on domain-specific knowledge

# ### Feature Engineering for DL
# 
# - DL directly takes the texts as inputs to the model.
# - The DL model is capable of learning features from the texts (e.g., embeddings)
# - The price is that the model is often less interpretable.
#     

# ### Strengths and Weakness 

# ![](../images/feature-engineer-strengths.png)

# ![](../images/feature-engineer-weakness.png)

# ## Modeling

# ### From Simple to Complex
# 
# - Start with heuristics or rules
# - Experiment with different ML models
#     - From heuristics to features
#     - From manual annotation to automatic extraction
#     - Feature importance (weights)
# - Find the most optimal model
#     - Ensemble and stacking
#     - Redo feature engineering
#     - Transfer learning
#     - Reapply heuristics

# ## Evaluation

# ### Why evaluation?
# 
# - We need to know how *good* the model we've built is -- "Goodness"
# - Factors relating to the evaluation methods
#     - Model building
#     - Deployment
#     - Production
# - ML metrics vs. Business metrics
# 

# ### Intrinsic vs. Extrinsic Evaluation
# 
# - Take spam-classification system as an example
# - Intrinsic:
#     - the precision and recall of the spam classification/prediction
# - Extrinsic:
#     - the amount of time users spent on a spam email
#     

# ### General Principles
# 
# - Do intrinsic evaluation before extrinsic.
# - Extrinsic evaluation is more expensive because it often invovles project stakeholders outside the AI team.
# - Only when we get consistently good results in intrinsic evaluation should we go for extrinsic evaluation.
# - Bad results in intrinsic often implies bad results in extrinsic as well.

# ### Common Intrinsic Metrics
# 
# - Principles for Evaluation Metrics Selection
# - Data type of the labels (ground truths)
#     - Binary (e.g., sentiment)
#     - Ordinal (e.g., informational retrieval)
#     - Categorical (e.g., POS tags)
#     - Textual (e.g., named entity, machine translation, text generation)
# - Automatic vs. Human Evalation

# ## Post-Modeling Phases

# ### Post-Modeling Phases
# 
# - Deployment of the model in a  production environment (e.g., web service)
# - Monitoring system performance on a regular basis
# - Updating system with new-coming data

# ## References

# - Chapter 2 of Practical Natural Language Processing. {cite}`vajjala2020`
