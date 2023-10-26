#!/usr/bin/env python
# coding: utf-8

# # Google Colab

# - As we are working with more and more data, we may need GPU computing for quicker processing.
# - This lecture note shows how we can capitalize on the free GPU computing provided by Google Colab and speed up the Chinese word segmentation of `ckip-transformers`.

# ## Prepare Google Drive

# - Create a working directory under your Google Drive, named `ENC2045_DEMO_DATA`.
# - Save the corpus files needed in that Google Drive directory.
# - We can access the files on our Google Drive from Google Colab. This can be useful when you need to load your own data in Google Colab.

# :::{note}
# 
# You can of course name the directory in which ever ways you like. The key is that we need to put the data files on the Google Drive so that we can access these files through Google Colab.
# 
# :::

# ## Run Notebook in Google Colab

# - Click on the button on top of the lecture notes website to open this notebook in Google Colab.

# ## Setting Google Colab Environment

# - Important Steps for Google Colab Environment Setting
#     - Change the Runtime for GPU
#     - Install Modules
#     - Mount Google Drive
#     - Set Working Directory

# ## Change Runtime for GPU

# - [Runtime] -> [Change runtime type]
# - For [Hardware accelerator], choose [GPU]

# In[1]:


get_ipython().system('nvidia-smi')


# ## Install Modules

# - Google Colab has been pre-instralled with several popular modules for machine learning and deep learning (e.g., `nltk`, `sklearn`, `tensorflow`, `pytorch`,`numpy`, `spacy`).
# - We can check the pre-installed modules here.

# In[2]:


get_ipython().system('pip list')


# - We only need to install modules that are not pre-installed in Google Colab (e.g., `ckip-transformers`).
# - This installation has to be done every time we work with Google Colab. But don't worry. It's quick.
# - This is how we install the package on Google Colab, exactly the same as we do in our terminal.

# In[3]:


## Google Drive Setting
get_ipython().system('pip install ckip-transformers')


# ## Mount Google Drive
#     

# - To mount our Google Drive to the current Google Colab server, we need the following codes.

# In[4]:


from google.colab import drive
drive.mount('/content/drive')


# - After we run the above codes, we need to click on the link presented, log in with our Google Account in the new window and get the authorization code.
# - Then copy the authorization code from the new window and paste it back to the text box in the notebook window.

# ## Set Working Directory

# - Change Colab working directory to the `ENC2045_demo_data` of the Google Drive

# In[5]:


import os
os.chdir('/content/drive/MyDrive/ENC2045_demo_data')
print(os.getcwd())


# ## Try `ckip-transformers` with GPU

# ### Initialize the `ckip-transformers`

# In[6]:


import ckip_transformers
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger
# Initialize drivers
ws_driver = CkipWordSegmenter(level=3, device=0)
pos_driver = CkipPosTagger(level=3, device=0)


# In[7]:


def my_tokenizer(doc):
    # `doc`: a list of corpus documents (each element is a document long string)
    cur_ws = ws_driver(doc, use_delim = True, delim_set='\n')
    cur_pos = pos_driver(cur_ws)
    doc_seg = [[(x,y) for (x,y) in zip(w,p)]  for (w,p) in zip(cur_ws, cur_pos)]
    return doc_seg


# ### Tokenization Chinese Texts

# In[8]:


import pandas as pd

df = pd.read_csv('dcard-top100.csv')
df.head()
corpus = df['content']
corpus[:10]


# In[9]:


get_ipython().run_cell_magic('time', '', 'corpus_seg = my_tokenizer(corpus)\n')


# In[10]:


corpus_seg[0][:50]

