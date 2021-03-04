# Chinese Word Segmentation

- Natural language processing methods are connected to the characteristics of the target language.
- To deal with the Chinese language, the most intimidating task is to find out a basic **linguistic unit** to work with.
- In this tutorial, we talk about a few segmentation methods in python.

## Segmentation using `jieba`

- [`jieba`](https://github.com/fxsjy/jieba)

import jieba
text = """
高速公路局說，目前在國道3號北向水上系統至中埔路段車多壅塞，已回堵約3公里。另外，國道1號北向仁德至永康路段路段，已回堵約有7公里。建議駕駛人提前避開壅塞路段改道行駛，行經車多路段請保持行車安全距離，小心行駛。
國道車多壅塞路段還有國1內湖-五堵北向路段、楊梅-新竹南向路段；國3三鶯-關西服務區南向路段、快官-霧峰南向路段、水上系統-中埔北向路段；國6霧峰系統-東草屯東向路段、國10燕巢-燕巢系統東向路段。
"""
text_jb = jieba.lcut(text)  
print(' | '.join(text_jb))

- initialize traditional Chinese dictionary
    - Download the traditional chinese dictionary from [`jieba-tw`](https://raw.githubusercontent.com/ldkrsi/jieba-zh_TW/master/jieba/dict.txt)
    
```
jieba.set_dictionary(file_path)
```

- Add own project-specific dictionary

```
jieba.load_userdict(file_path)
```

- Add add-hoc words to dictionary

```
jieba.add_word(word, freq=None, tag=None)
```

- Remove words

```
jieba.del_word(word)
```

- Word segmentation
    - `jieba.cut()` returns a `generator` object
    - `jieba.lcut()` resuts a `List` object
    
```
# full

jieba.cut(TEXT, cut_all=True)
jieba.lcut(TEXT, cut_all=True

# default
jieba.cut(TEXT, cut_all=False)
jieba.lcut(TEXT, cut_all=False)
```

## Segmentation using CKIP Transformer

- For more detail on the installation of `ckip-transformers`, please read their [documentation](https://github.com/ckiplab/ckip-transformers).

```
pip install -U transformers
```

import ckip_transformers
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker

- When initializing the models, `level` specifies three levels of segmentation resolution. Level 1 is the fastest while Level 3 is the most accurate
- `device = 0` for GPU computing.

%%time

# Initialize drivers
ws_driver  = CkipWordSegmenter(level=3, device = -1)
pos_driver = CkipPosTagger(level=3, device = -1)
ner_driver = CkipNerChunker(level=3, device = -1)

# Input text
text = """
高速公路局說，目前在國道3號北向水上系統至中埔路段車多壅塞，已回堵約3公里。另外，國道1號北向仁德至永康路段路段，已回堵約有7公里。建議駕駛人提前避開壅塞路段改道行駛，行經車多路段請保持行車安全距離，小心行駛。
國道車多壅塞路段還有國1內湖-五堵北向路段、楊梅-新竹南向路段；國3三鶯-關西服務區南向路段、快官-霧峰南向路段、水上系統-中埔北向路段；國6霧峰系統-東草屯東向路段、國10燕巢-燕巢系統東向路段。
"""


# paragraph breaks
text = [p for p in text.split('\n') if len(p)!=0]

# Run pipeline
ws  = ws_driver(text)
pos = pos_driver(ws)
ner = ner_driver(text)

- When doing the word segmentation, there are a few parameters to consider in `ws_driver()`:
    - `use_delim`: by default, ckip transformer breaks the texts into sentences using the following delimiters `'，,。：:；;！!？?'`, and concatenate them back in the outputs.
    - `delim_set`: to specify self-defined sentence delimiters
    - `batch_size`
    - `max_length`

# Enable sentence segmentation
ws  = ws_driver(text, use_delim=True)
ner = ner_driver(text, use_delim=True)

# Disable sentence segmentation
pos = pos_driver(ws, use_delim=False)

# Use new line characters and tabs for sentence segmentation
pos = pos_driver(ws, delim_set='\n\t')

- A quick comparison of the results based on `jieba` and `ckip-transomers`:

print(' | '.join(text_jb))

print( '\n\n'.join([' | '.join(p) for p in ws]))

# Pack word segmentation and part-of-speech results
def pack_ws_pos_sentece(sentence_ws, sentence_pos):
   assert len(sentence_ws) == len(sentence_pos)
   res = []
   for word_ws, word_pos in zip(sentence_ws, sentence_pos):
      res.append(f'{word_ws}({word_pos})')
   return '\u3000'.join(res)

# Show results
for sentence, sentence_ws, sentence_pos, sentence_ner in zip(text, ws, pos, ner):
   print(sentence)
   print(pack_ws_pos_sentece(sentence_ws, sentence_pos))
   for entity in sentence_ner:
      print(entity)
   print()

## Challenges of Word Segmentation

text2 = [
    "女人沒有她男人甚麼也不是",
    "女人沒有了男人將會一無所有",
    "下雨天留客天留我不留",
    "行路人等不得在此大小便",
    "兒的生活好痛苦一點也沒有糧食多病少掙了很多錢",
]
ws2  = ws_driver(text2)
pos2 = pos_driver(ws2)
ner2 = ner_driver(text2)

# Show results
for sentence, sentence_ws, sentence_pos, sentence_ner in zip(text2, ws2, pos2, ner2):
   print(sentence)
   print(pack_ws_pos_sentece(sentence_ws, sentence_pos))
   for entity in sentence_ner:
      print(entity)
   print()

```{tip}
It is still not clear to me how we can include user-defined dictionary in the `ckip-transformers`. However, I know that it is possible to use self-defined dictionary in the `ckiptagger`. 

If you know where and how to incorporate self-defined dictionary in the `ckip-transformers`, please let me know. Thanks!

```

## Other Modules for Chinese Word Segmentation

- You can try other modules and check their segmentation results and qualities:
    - [`ckiptagger`](https://github.com/ckiplab/ckiptagger) (Probably the predecessor of `ckip-transformers`?) 
    - [`pkuseg`](https://github.com/lancopku/pkuseg-python)
    - [`pyhanlp`](https://github.com/hankcs/pyhanlp)
    - [`snownlp`](https://github.com/isnowfy/snownlp)
- When you play with other segmenters, please note a few important issues for determining the segmenter for your project:
    - Most of the segmenters were trained based on the simplified Chinese.
    - It is important to know if the segmenter allows users to add self-defined dictionary to improve segmentation performance.
    - The documentation of the tagsets (i.e., POS, NER) needs to be well-maintained so that users can easily utilize the segmentation results for later downstream projects.
    

## Chinese NLP Using `spacy` 

- `spacy 3+` starts to support transformer-based NLP. Please make sure you have the most recent version of the module.

```
$ pip show spacy
```


- Documentation of [`spacy`](https://spacy.io/usage#quickstart)
- Installation steps:

```
$ pip install -U pip setuptools wheel
$ pip install -U spacy
$ python -m spacy download zh_core_web_trf
$ python -m spacy download zh_core_web_sm
$ python -m spacy download en_core_web_trf
$ python -m spacy download en_core_web_sm
```

- `spacy` provides a good API to help users install the package. Please set up the relevant parameters according to the API and find your own installation codes.

---
![](../images/spacy-install.png)

---

import spacy
from spacy import displacy
# load language model
nlp_zh = spacy.load('zh_core_web_trf')## disable=["parser"]
# parse text 
doc = nlp_zh('這是一個中文的句子')

```{note}
For more information on POS tags, see spaCy [POS tag scheme documentation](https://spacy.io/api/annotation#pos-tagging).

```

# parts of speech tagging
for token in doc:
    print(((token.text, 
            token.pos_, 
            token.tag_,
            token.dep_,
            token.is_alpha,
            token.is_stop,
            )))

' | '.join([token.text+"_"+token.tag_ for token in doc])

## Check meaning of a POS tag (Not working??)
spacy.explain('VC')

### Visualizing linguistic features

# Visualize
displacy.render(doc, style="dep")

options = {"compact": True, "bg": "#09a3d5",
           "color": "white", "font": "Source Sans Pro",
          "distance": 120}
displacy.render(doc, style="dep", options=options)

doc2 = nlp_zh.pipe(text)

for d in doc2:
    print(' | '.join([token.text+"_"+token.tag_ for token in d])+'\n')

```{tip}
Please read the [documentation of `spacy`](https://spacy.io/usage/linguistic-features) very carefully for additional ways to extract other useful linguistic properties.

You may need that for the assignments.

```

## Conclusion

- Different segmenters have very different behaviors.
- The choice of a proper segmenter may boil down to the following crucial questions:
    - How do we evaluate the segmenter's performance?
    - What is the objective of getting the word boundaries?
    - What is a word in Chinese?