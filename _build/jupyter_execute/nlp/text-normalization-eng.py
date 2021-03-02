# Text Normalization


## Overview

The objective of text normalization is to clean up the text by removing unnecessary and irrelevant components.

import spacy
import unicodedata
import re
from nltk.corpus import wordnet
import collections
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
    ## create a regex pattern of all contracted forms
    contractions_pattern = re.compile('({})'.format('|'.join(
        contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0) # the whole matched contraction

        # if the matched contraction (=keys) exists in the dict, 
        # get its corresponding uncontracted form (=values)
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())
        
        return expanded_contraction

    
    # find each contraction in the pattern,
    # find it from text,
    # and replace it using the output of 
    # expand_match 
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

:::{note}
In `re.sub(repl, str)`, when `repl` is a function like above, the function is called for every non-overlapping occurrence of pattern `contractions_pattern`. The function `expand_match` takes a single matched contraction, and returns the replacement string, i.e., its uncontracted form in the dictionary. 
:::

print(expand_contractions("Y'all can't expand contractions I'd think"))

print(expand_contractions("I'm very glad he's here! And it ain't here!"))

type(CONTRACTION_MAP)

list(CONTRACTION_MAP.items())[:5] # check the first five items

## Accented Characters (Non-ASCII)

- The `unicodedata` module handles unicode characters very efficiently. Please check [unicodedata dcoumentation] (https://docs.python.org/3/library/unicodedata.html) for more details.
- When dealing with the English data, we may often encounter foreign characters in texts that are not part of the ASCII character set.

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

:::{note}
- `str.encode()` returns an encoded version of the string as a bytes object using the specified encoding.
- `byes.decode()` returns a string decoded from the given bytes using the specified encoding.
:::

- Another common scenario is the case where texts include both English and Chinese characters. What's worse, the English characters are in full-width.

## Chinese characters with full-width English letters and punctuations
text = '中英文abc,，。.．ＡＢＣ１２３'
print(unicodedata.normalize('NFKD', text))
print(unicodedata.normalize('NFKC', text))  # recommended method
print(unicodedata.normalize('NFC', text))
print(unicodedata.normalize('NFD', text))

- Sometimes, we may even want to keep characters of one language only.

text = "中文ＣＨＩＮＥＳＥ。！＝=.= ＾o＾ 2020/5/20 alvin@gmal.cob@%&*"

# remove puncs/symbols
print(''.join(
    [c for c in text if unicodedata.category(c)[0] not in ["P", "S"]]))

# select letters
print(''.join([c for c in text if unicodedata.category(c)[0] in ["L"]]))

# remove alphabets
print(''.join(
    [c for c in text if unicodedata.category(c)[:2] not in ["Lu", 'Ll']]))

# select Chinese chars?
print(''.join([c for c in text if unicodedata.category(c)[:2] in ["Lo"]]))

```{note}
Please check [this page](https://www.fileformat.info/info/unicode/category/index.htm) for unicode category names.

It seems that the unicode catetory `Lo` is good to identify Chinese characters?

We can also make use of the category names to identify punctuations.
```

## Special Characters

- Depending on the research questions and the defined tasks, we often need to decide whether to remove irrelevant characters.
- Common irrelevant (aka. non-informative) characters may include:
    - punctuation marks
    - digits
    - any other non-alphanumeric characters 

def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', text)
    return text

s = "This is a simple case! Removing 1 or 2 symbols is probably ok...:)"
print(remove_special_characters(s))
print(remove_special_characters(s, True))

:::{warning}
In the following example, if we use the same `remove_special_characters()` to pre-processing the text, what additional problems will we encounter?

Any suggestions or better alternative methods?
:::

s = "It's a complex sentences, and I'm not sure if it's ok to replace all symbols then :( What now!!??)"
print(remove_special_characters(s))

## Stopwords

- At the word-token level, there are words that have little semantic information and are usually removed from text in text preprocessing. These words are often referred to as **stopwords**.
- However, there is no universal stopword list. Whether a word is informative or not depends on your research/project objective. It is a linguistic decision.
- The `nltk.corpus.stopwords.words()` provides a standard English language stopwords list.

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

- We can check the languages of the stopwords lists provided by `nltk`.

nltk.corpus.stopwords.fileids()

## Redundant Whitespaces

- Very often we would see redundant duplicate whitespaces in texts. 
- Sometimes, when we remove special characters (punctuations, digits etc.), we may replace those characters with whitespaces (not empty string), which may lead to duplicate whitespaces in texts.

def remove_redundant_whitespaces(text):
    text = re.sub(r'\s+'," ", text)
    return text.strip()

s = "We are humans  and we   often have typos.  "
remove_redundant_whitespaces(s)