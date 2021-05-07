# Assignment V: Text Vectorization

## Question 1

Use `nltk.corpus.inaugural` as the corpus data for this exercise. It is a collection of US presidential inaugural speeches over the years. Cluster all the inaugural speeches collected in the corpus based on their bag-of-words vectorized representations.

Please consider the following settings for bag-of-words model:
- Use the English stopwords provided in `nltk.corpus.stopwords.words('english')` to remove uninformative words.
- Lemmatize word tokens using `WordNetLemmatizer()`.
- Normalize the letter casing.
- Include in the Bag-of-words model only words consisting as alphabets or hyphens.
- Use `TfIdfVectorizer()` for bag-of-word vectorization.

tfidf_df

similarity_df

plt.figure(figsize=(7, 5))
plt.title('US Presidential Inaugural Speech Analysis')
plt.xlabel('Inaugural Speech')
plt.ylabel('Distance')
color_threshold = 1.8
dendrogram(Z, color_threshold = color_threshold, labels=textsid)
plt.axhline(y=color_threshold, c='k', ls='--', lw=0.5)

## Question 2

Please use the Chinese song lyrics from the directory, `demo_data/ChineseSongLyrics`, as the corpus for this exercise. The directory is a collection of song lyrics from nine Chinese pop-song artists.

Please utilize the bag-of-words method to vectorize each artist's song lyrics and provide a cluster analysis of each artist in terms of their textual similarities in lyrics.

A few notes for data processing:
- Please use `ckip-transformers` for word segmentation and POS tagging.
- Please build the bag-of-words model using the `Tfidfvectorizer()`.
- Please include in the model only words (a) whose POS tags start with 'N' or 'V', and (b) which consist of NO digits, alphabets, symbols and punctuations.
- Please make sure you have the word tokens intact when doing the vectorization using `Tfidfvectorizer()`.

The expected result is a dendrogram as shown below. But please note that depending on how you preprocess the data and adjust the parameters of bag-of-words representations, we may have somewhat different results. Please indicate and justify your parameter settings in the bag-of-words model creation (i.e., using markdown cells in notebook).

tv_df

similarity_doc_df

plt.figure(figsize=(7, 5))
plt.title('Chinese Pop Song Artists')
plt.xlabel('Artists')
plt.ylabel('Distance')
color_threshold = 0.1
dendrogram(Z, labels=fileids, leaf_rotation = 90, color_threshold = color_threshold, above_threshold_color='b')
plt.axhline(y=color_threshold, c='k', ls='--', lw=0.5)