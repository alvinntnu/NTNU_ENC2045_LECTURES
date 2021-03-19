# Assignment IV: Chinese Language Processing

## Question 1

The csv file `dcard-top100.csv` includes top 100 posts from Dcard, which a on-line discussion forum for school life in Taiwan. The texts are in the `content` column.

Please preprocess the data by:
- removing symbols, punctuations, emoticons or other non-linguistic symbols
- removing stopwords (Please use the stopword list provided in `demo_data/stopwords/tomlinNTUB-chinese-stopwords.txt`)
- performing word segmentation on the corpus using `ckip-transformer`
- creating a word frequency list of this tiny corpus
- including only word tokens which have at least two characters in the frequency list

```{warning}
Please note that the preprocessing steps are important. Removal of characters from texts may have a lot to do with the word segmentation performance.
```

dcards_df[:21]

## Question 2

Use `ckip-transformer` to extract all named entities and create a frequency list of the named entities.

In particular, please identify named entities of organizations (`ORG`) and geographical names (`GPE`) and provide their frequencies in the Dcard Corpus.

ner_df[:21]

## Question 3

In this exercise, please work with `spacy` for Chinese processing. (Use the model `zh_core_web_trf`)

Please process the same Dcard Corpus (from the csv file) by:

- performing the word tokenization
- identifying all nouns and verbs (i.e., words whose tags start with N or V)
- identifying all words with at least two characters
- removing all words that contain alphabets or digits
- removing all words that are included in the `stopword_list` (cf. Question 1)

Based on the above text-preprocessing criteria, your goal is to create a word frequency list and visualize the result in a Word Cloud.

```{note}
`spacy` uses the `jieba` for Chinese word segmentation. There may be more tagging errors. In the expected results presented below, I did not use any self-defined dictionary. For this exercise, please ignore any tagging errors out of the module for the moment.
```

```{tip}
Please check the module `wordcloud` for the visualization.
```

nouns_verbs_df[:21]

plt.figure(figsize=(7,5))
plt.imshow(wc)
plt.axis("off")
plt.show()

## Question 4

Following Question 3, after you process each article with `spacy`, please extract all the `subject` + `predicate` word pairs from the corpus.

To simplify the task, please extract word token pairs whose dependency relation is `nsubj`, with the predicate being the head and subject being the dependent.

- Remove words that include alphabets and digits

sub_pred_df[:21]