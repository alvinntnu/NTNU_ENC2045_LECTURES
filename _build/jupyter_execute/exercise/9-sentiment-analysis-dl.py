#!/usr/bin/env python
# coding: utf-8

# # Assignment IX: Sentiment Analysis Using Deep Learning

# ## Question 1
# 
# Build a movie review classifier using the dataset in `demo_data/movie_reviews.csv`. The objective of the classifier is to automatically classify a movie review into either positive or negative category.
# 
# The dataset is the famous IMBd moview reviews dataset. You can take a look at the SOTA classification performance on this dataset [here](https://paperswithcode.com/sota/sentiment-analysis-on-imdb).
# 
# In your experiments, please include the following strategies in your considerations:
# 
# - Please use sequence models for this task (you may experiment with networks of different topologies).
# - For embedding layers, please try both self-trained embedding layer along with the sentiment classifier, as well as pre-trained embeddings (either provided in `spacy`, or available on [GloVe](https://nlp.stanford.edu/projects/glove/) website).
# - Please include dropout and regularization layers to avoid overfitting.
