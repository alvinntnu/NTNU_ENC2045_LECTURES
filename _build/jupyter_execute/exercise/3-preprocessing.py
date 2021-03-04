# Assignment III: Preprocessing

## Question 1

Please use the `grutenberg` corpus provided in `nltk` and extract the text written by Lewis Caroll, i.e., `carroll-alice.txt`, as your corpus data.

With this corpus data, please perform text preprocessing on the **sentences** of the corpus.

In particular, please:

- pos-tag all the sentences to get the parts-of-speech of each word
- lemmatize all words using `WordNetLemmatizer` on a sentential basis

Please provide your output as shown below:

- it is a data frame
- the column `alice_sents` includes the original sentence texts
- the column `alice_sents_pos` includes the annotations of the word/postag for each sentence
- the column `sents_lem` includes the lemmatized version of the sentences


```{note}
Please note that the lemmatized form of the BE verbs (e.g., *was*) should be *be*.
```


alice_sents_df

## Question 2

Based on the output of the previous question, please create a lemma frequnecy list of `carroll-alice.txt` using the lemmatized forms by including only lemmas which are:
- consisting of only alphabets or hyphens
- at least 5-character long

The casing is irrelevant (i.e., case normalization is needed).

The expected output is provided as follows.


# Top 20 frequent lemmas
alice_wf.most_common(20)

## Question 3

Please identify top verbs that co-occcur with the name *Alice* in the text, with the name being the **subject** of the verb. 

Please use the `en_core_web-sm` model in `spacy` for English dependency parsing.

To simply the matter, please identify all the verbs that have a dependency relation of `nsubj` with the noun `Alice` (where `Alice` is the **dependent**, and the verb is the **head**).

nltk.FreqDist(np_chunks_targets).most_common(20)