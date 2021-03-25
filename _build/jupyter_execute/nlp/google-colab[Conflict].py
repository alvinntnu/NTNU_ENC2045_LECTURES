# Google Colab

- As we are working with more and more data, we may need GPU computing for quicker processing.
- This lecture note shows how we can capitalize on the free GPU computing provided by Google Colab and speed up the Chinese word segmentation of `ckip-transformers`.

## Setup Google Drive

- Create a working directory under your Google Drive, named `ENC2045_DEMO_DATA`.
- Save the corpus files needed in that Google Drive directory.
- We can access the files on our Google Drive from Google Colab. This can be useful when you need to load your own data in Google Colab.

:::{note}
You can of course name the directory in which ever ways you like. The key is that we need to put the data files on the Google Drive so that we can access these files through Google Colab.
:::

## Run Notebook in Google Colab

- Click on the button on top of the lecture notes website to open this notebook in Google Colab.

## Setting Google Colab Environment

- GPU Setting:
    - [Runtime] -> [Change runtime type]
    - For [Hardware accelerator], choose [GPU]

!nvidia-smi

- Install modules that are not installed in the current Google Colab

## Google Drive Setting
!pip install ckip-transformers

- Mount Our Google Drive
    - After we run the following codes, click on the link, log in with your Google Account and get the authorization code.
    - Copy-paste the authorization code back to the text box.

from google.colab import drive
drive.mount('/content/drive')

- Change Colab working directory to the `ENC2045_demo_data` of the Google Drive

import os
os.chdir('/content/drive/MyDrive/ENC2045_demo_data')
print(os.getcwd())


## Try `ckip-transformers` with GPU

### Initialize the `ckip-transformers`

import ckip_transformers
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger
# Initialize drivers
ws_driver = CkipWordSegmenter(level=3, device=0)
pos_driver = CkipPosTagger(level=3, device=0)


def my_tokenizer(doc):
    # `doc`: a list of corpus documents (each element is a document long string)
    cur_ws = ws_driver(doc, use_delim = True, delim_set='\n')
    cur_pos = pos_driver(cur_ws)
    doc_seg = [[(x,y) for (x,y) in zip(w,p)]  for (w,p) in zip(cur_ws, cur_pos)]
    return doc_seg

### Tokenization Chinese Texts

import pandas as pd

df = pd.read_csv('dcard-top100.csv')
df.head()
corpus = df['content']
corpus[:10]

%%time
corpus_seg = my_tokenizer(corpus)

corpus_seg[0][:50]