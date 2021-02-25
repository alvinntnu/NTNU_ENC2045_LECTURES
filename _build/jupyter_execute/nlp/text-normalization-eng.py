# Text Normalization (English)


## Overview

- Before real data analysis and mining, we usually need to preprocess the textual data into more easy-to-interpret formats.
- This step is tedious but crucial and often involves a wide variety of techniques that convert raw text into well-defined sequences of linguistic components.
- There are at least two objectives:
    - Text Cleaning
        - HTML tags
        - Unnecessary tokens (stopwords, punctuations, symbols, numbers)
        - Contractions
        - Spelling errors
    - Text Enrichment
        - Tokenization
        - Stemming
        - Lemmatization
        - Tagging
        - Chunking
        - Parsing

import spacy
import unicodedata
#from contractions import CONTRACTION_MAP
import re
from nltk.corpus import wordnet
import collections
#from textblob import Word
from nltk.tokenize.toktok import ToktokTokenizer
from bs4 import BeautifulSoup

## HTML Tags

- One of the most frequent data source is the Internet. Assuming that we web-crawl text data from a wikipedia page, we need to clean up the HTML codes quite a bit.
- Important steps:
    - Download and parse the HTML codes of the webpage
    - Identify the elements from the page that we are interested in
    - Extract the textual contents of the HTML elements
    - Remove unnecessary HTML tags
    - Remove extra spacing/spaces

import requests
from bs4 import BeautifulSoup

data = requests.get('https://en.wikipedia.org/wiki/Python_(programming_language)')
content = data.content
print(content[:500])

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    ## Can include more HTML preprocessing here...
    stripped_html_elements = soup.findAll(name='div',attrs={'id':'mw-content-text'})
    stripped_text = ' '.join([h.get_text() for h in stripped_html_elements])
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text

clean_content = strip_html_tags(content)
print(clean_content[:500])

## Stemming

- Stemming is the process where we standardize word forms into their base stem irrespective of their inflections.
- The `nltk` provides several popular stemmers for English:
    - `nltk.stem.PorterStemmer`
    - `nltk.stem.LancasterStemmer`
    - `nltk.stem.RegexpStemmer`
    - `nltk.stem.SnowballStemmer`

- We can compare the results of different stemmers.

import nltk
from nltk.stem import PorterStemmer, LancasterStemmer, RegexpStemmer, SnowballStemmer

words = ['jumping', 'jumps', 'jumped', 'jumpy']
ps = PorterStemmer()
ls = LancasterStemmer()
ss = SnowballStemmer('english')

rs = RegexpStemmer('ing$|s$|ed$|y$', min=4) # set the minimum of the string to stem


[ps.stem(w) for w in words]

[ls.stem(w) for w in words]

[ss.stem(w) for w in words]

[rs.stem(w) for w in words]

## Lemmatization


- Lemmatization is similar to stemmarization.
- It is a process where we remove word affixes to get the **root word** but not the **root stem**.
- These root words, i.e., lemmas, are lexicographically correct words and always present in the dictionary.

```{admonition} Question
:class: attention
In terms of Lemmatization and Stemmatization, which one requires more computational cost? That is, which processing might be slower?
```

- Two frequently-used lemmatizers
    - `nltk.stem.WordNetLemmatizer`
    - `spacy`

### WordNet Lemmatizer

- WordNetLemmatizer utilizes the dictionary of WordNet.
- It requires the parts of speech of the word for lemmatization.

from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

# nouns
print(wnl.lemmatize('cars','n'))
print(wnl.lemmatize('men', 'n'))

# verbs
print(wnl.lemmatize('running','v'))
print(wnl.lemmatize('ate', 'v'))

# adj
print(wnl.lemmatize('saddest','a'))
print(wnl.lemmatize('fancier','a'))
print(wnl.lemmatize('jumpy','a'))

### Spacy 

```{warning}
To use `spacy` properly, you need to download/install the language models of the English language first before you load the parameter files. Please see [spacy documentation](https://spacy.io/usage/models/) for installation steps.

Also, please remember to install the language models in the right conda environment.
```

import spacy
nlp = spacy.load('en_core_web_sm', parse=False, tag=True, entity=False)

text = 'My system keeps crashing his crashed yesterday, ours crashes daily'
text_tagged = nlp(text)

- `spacy` processes the text by tokenizing it into tokens and enriching the tokens with many annotations.

for t in text_tagged:
    print(t.text+'/'+t.lemma_ + '/'+ t.pos_)

def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

lemmatize_text("My system keeps crashing! his crashed yesterday, ours crashes daily")


## Contractions

- For the English data, contractions are problematic sometimes. 
- These may get even more complicated when different tokenizers deal with contractions differently.
- A good way is to expand all contractions into their original independent word forms.

from contractions import CONTRACTION_MAP
import re

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

expand_contractions("Y'all can't expand contractions I'd think")

expand_contractions("I'm very glad he's here!")

## Accented Characters (Non-ASCII)

- [unicodedata dcoumentation](https://docs.python.org/3/library/unicodedata.html)

import unicodedata

def remove_accented_chars(text):
#     ```
#     (NFKD) will apply the compatibility decomposition, i.e. 
#     replace all compatibility characters with their equivalents. 
#     ```
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


remove_accented_chars('Sómě Áccěntěd těxt')

# print(unicodedata.normalize('NFKD', 'Sómě Áccěntěd těxt'))
# print(unicodedata.normalize('NFKD', 'Sómě Áccěntěd těxt').encode('ascii','ignore'))
# print(unicodedata.normalize('NFKD', 'Sómě Áccěntěd těxt').encode('ascii','ignore').decode('utf-8', 'ignore'))

## Special Characters

## Stopwords

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer() # assuming per line per sentence 
    # for other Tokenizer, maybe sent.tokenize should go first
stopword_list = nltk.corpus.stopwords.words('english')
def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopwords]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

remove_stopwords("The, and, if are stopwords, computer is not")

## Redundant Whitespaces

## Spelling Checks