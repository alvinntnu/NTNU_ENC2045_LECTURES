#!/usr/bin/env python
# coding: utf-8

# # Assignment X: Neural Language Model

# ## Question 1
# 
# Use the same simple dataset (as shown below) discussed in class and create a simple **trigram-based** neural language using sequence models.
# 
# 
# ```
# # source text
# data = """ Jack and Jill went up the hill\n
# 		To fetch a pail of water\n
# 		Jack fell down and broke his crown\n
# 		And Jill came tumbling after\n """
# ```

# ## Question 2
# 
# Use the same simple dataset and create a simple line-based neural language model using sequence models.
# 
# For example, given the first line, `Jack and Jill went up the hill`, you need to include all possible sequences from this line as inputs for your neural language model training, including all bigrams, trigrams, ..., ngrams combinations (see below)
# 
# ```
# # source text
# data = """Jack and Jill went up the hill\n
# 		To fetch a pail of water\n
# 		Jack fell down and broke his crown\n
# 		And Jill came tumbling after\n"""
# ```
# 
# ```
# ## seqeunces as inputs from line 2
# ['To', 'fetch']
# ['To', 'fetch', 'a']
# ['To', 'fetch', 'a', 'pail']
# ['To', 'fetch', 'a', 'pail', 'of']
# ['To', 'fetch', 'a', 'pail', 'of', 'water']
# 
# ```

# ## Question 3
# 
# Use the Brown corpus (`nltk.corpus.brown`) to create a trigram-based neural language model.
# 
# Please use the language model to generate 50-word text sequences using the seed text "The news". Provide a few examples from your trained model.
# 
# A few important notes in data preprocessing:
# 
# - When preparing the input sequences of trigrams for model training, please make sure the trigram does not span across "sentence boundaries". You can utilize the sentence tokenization annotations provided by the `ntlk.corpus.brown.sents()`.
# - The neural language model will be trained based on all trigrams that fulfill the above criterion in the entire Brown corpus.
# 
# :::{warning}
# Even though the `brown` corpus is a small-size one, the input sequences of all observed trigrams may still require considerable amount of memory for processing. Some of you may not have enough memory space to store the entire inputs and outputs data. Please find your own solutions if you encounter this memory issue. (Hint: Use `generator()`).
# :::
# 
# :::{warning}
# When you use your trigram-based neural language model to generate sequences, please add **randomness** to the sampling of the next word. If you always ask the language model to choose the next word of highest predicted probability value, your text would be very repetitive. Please consult Ch. 8.1.4 of François Chollet's Deep Learning with Python (one of the course textbooks) for the implementation of **temperature** for this exercise.
# :::

# - Examples of the 50-word text sequences created by the language model:

# In[16]:


[print(s+'\n'*2) for s in generated_sequences]

