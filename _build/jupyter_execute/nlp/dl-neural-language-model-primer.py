#!/usr/bin/env python
# coding: utf-8

# # Neural Language Model: A Start
# 

# - In this tutorial, we will look at a naive example of neural language model building.
# - Given a corpus, we can build a neural language model, which will learn to predict the next word given a specified limited context.
# - Depending on the size of the **limited context**, we can implement different types of neural language model:
#     - Bigram-based neural language model: The LM model only uses one preceding word for the next-word prediction.
#     - Trigram-based neural language model: The model will use two preceding words for the next-word prediction.
#     - *Line*-based neural language model: The model with use all the existing fore-going words for the next-word prediction

# - This tutorial will demonstarte how to build a bigram-based language model.
# - In the Assignments, you need to extend the same rationale to other types of language models.

# ## Workflow of Neural Language Model

# ![](../images/neural-language-model-flowchart.png)

# ## Bigram Model

# - A bigram-based language model assumes that the next word (to be predicted) depends only on the previous word.

# In[1]:


## Dependencies
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding


# ### Tokenization
# 
# - A quick reminder of important parameters for `Tokenzier()`:
#    - `num_words`: the maximum number of words to keep, based on word frequency. Only the most common `num_words-1` words will be kept.
#    - `filters`: a string where each element is a character that will be filtered from the texts. The default is all punctuation, plus tabs and line breaks, minus the `'` character.
#    - `lower`: boolean. Whether to convert the texts to lowercase.
#    - `split`: str. Separator for word splitting.
#    - `char_level`: if True, every character will be treated as a token.
#    - `oov_token`: if given, it will be added to word_index and used to replace out-of-vocabulary words during text_to_sequence calls

# In[2]:


# source text
data = """ Jack and Jill went up the hill\n
		To fetch a pail of water\n
		Jack fell down and broke his crown\n
		And Jill came tumbling after\n """

data = [l.strip() for l in data.split('\n') if l != ""]

# integer encode text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)

# now the data consists of a sequence of word index integers
encoded = tokenizer.texts_to_sequences(data)

# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
print(tokenizer.word_index)


# ### Text-to-Sequences and Training-Testing Sets

# - Principles for bigrams extraction
#     - When we create bigrams as the input sequences for network training, we need to make sure that we do not include **unmeaningful** bigrams, such as bigrams spanning the text boundaries, or sentence boundaries.

# In[3]:


# create bigrams sequences

## bigrams holder
sequences = list()


## Extract bigrams from each text
for e in encoded:
    for i in range(1, len(e)):
        sequence = e[i - 1:i + 1]
        sequences.append(sequence)
print('Total Sequences: %d' % len(sequences))
sequences = np.array(sequences)


# In[4]:


sequences[:5]


# - A sequence contains both our input and also output of the network.
# - That is, for bigram-based LM, the first word is the input *X* and the second word is the expected output *y*.

# In[5]:


# split into X and y elements
X, y = sequences[:, 0], sequences[:, 1]


# In[6]:


print(sequences[:5])
print(X[:5])
print(y[:5])


# ### One-hot Representation of the Next-Word 
# 
# - Because the neural language model is going to be a multi-class classifier (for word prediction), we need to convert our `y` into one-hot encoding.

# In[7]:


# one hot encode outputs
y = to_categorical(y, num_classes=vocab_size)


# In[8]:


y.shape


# In[9]:


print(y[:5])


# ### Define RNN Language Model

# In[10]:


# define model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=10, input_length=1))
model.add(LSTM(50))  # LSTM Complexity
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())


# In[11]:


# compile network
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# fit network
model.fit(X, y, epochs=500, verbose=2)


# In[12]:


plot_model(model)


# ### Text Generation Using the Model

# - After we trained the bigram-based LM, we can use the model for text generation.
# - we can implement a simple text generator: the model always outputs the next-word that has the highest predicted probability values.
# - At every timestep, the model will use the newly predicted word as the input for another next-word prediction.

# In[13]:


# generate a sequence from the model
def generate_seq(model, tokenizer, seed_text, n_words):
    in_text, result = seed_text, seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = np.array(encoded)
        
        # predict a word in the vocabulary
        yhat=np.argmax(model.predict(encoded), axis=-1)
        
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text, result = out_word, result + ' ' + out_word
    return result


# ```{tip}
# When we generate the output sequence, we use a **greedy search**, which selects the most likely word at each time step in the output sequence. While this approach features its efficiency, the quality of the final output sequences may not be necessarily optimal.
# ```

# In[19]:


# evaluate
print(generate_seq(model, tokenizer, 'Jill', 10))


# ### Sampling Strategies for Text Generation

# - Given a trained language model and a seed text chunk, we can generate new text by greedy-search like we've seen above.
# - But we may sometimes like to add a bit vibe to the robotic texts.
# - Possible alternatives:
#     - We can re-normalize the predicted probability distributions of all next-words to reduce probability differences between the highest and the lowest. (Please see Ch.8.1 Text Generation with LSTM in Chollet's Deep Learning with Python. You will need this for the assignment.)
#     - We can use non-greedy search by keeping the top *k* probable candidates in the list for next-word prediction. (cf. **Beam Search** below).

# ## Beam Search (skipped)

# ### Searching in NLP

# - In the previous demonstration, when we generate the predicted next word, we adopt a naive approach, i.e., always choosing the word of the highest probability.
# - It is common in NLP for models to output a probability distribution over words in the vocabulary.
# - This step involves searching through all the possible output sequences based on their likelihood.
# - Choosing the next word of highest probability does not guarantee us the most optimal sequence.
# - The search problem is exponential in the length of the output sequence given the large size of vocabulary.
# 

# ### Beam Search Decoding

# The beam search expands all possible next steps and keeps the **$k$** most likely, where **$k$** is a researcher-specified parameter and controls the number of beams or parallel searches through the sequence of probabilities.

# The search process can stop for each candidate independently either by:
# 
# - reaching a maximum length
# - reaching an end-of-sequence token
# - reaching a threshold likelihood

# ```{note]
# Please see Jason Brownlee's blog post [How to Implement a Beam Search Decoder for Natural Language Processing](https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/) for the python implementation.
# 
# The following codes are based on Jason's code.
# ```

# :::{warning}
# The following codes may not work properly. In Beam Search, when the model predicts `None` as the next character, we should set it as a stopping condition. The following codes have not be optimized with respect to this.
# :::

# In[15]:


# generate a sequence from the model
def generate_seq_beam(model, tokenizer, seed_text, n_words, k):
    in_text = seed_text 
    sequences = [[[in_text], 0.0]]
    # prepare id_2_word map
    id_2_word = dict([(i,w) for (w, i) in tokenizer.word_index.items()])
    
    # start next-word generating
    for _ in range(n_words):
        all_candidates = list()        
        #print("Next word ", _+1)
        # temp list to hold all possible candidates
        # `sequence + next words`


        # for each existing sequence
        # take the last word of the sequence
        # find probs of all words in the next position
        # save the top k
        # all_candidates should have 3 * 22 = 66 candidates
        
        for i in range(len(sequences)):
            # for the current sequence
            seq, score = sequences[i]            
            # next word probablity distribution
            encoded = tokenizer.texts_to_sequences([seq[-1]])[0]
            encoded = np.array(encoded)
            model_pred_prob = model.predict(encoded).flatten()

            # compute all probabilities for `curent_sequence + all_possible_next_word`
            for j in range(len(model_pred_prob)):
                candidate = [seq + [id_2_word.get(j+1)], score-np.log(model_pred_prob[j])]
                all_candidates.append(candidate)

            all_candidates= [[seq, score] for seq, score in all_candidates if seq[-1] is not None]

            # order all candidates (seqence + nextword) by score
            #print("all_condidates length:", len(all_candidates))
            ordered = sorted(all_candidates, key = lambda x:x[1]) # default ascending
            # select k best
            sequences = ordered[:k] ## choose top k

    return sequences


# In[16]:


generate_seq_beam(model, tokenizer, 'Jill', 5, k =10)


# ## References
# 
# - Chollet (2017): Ch 8.1
# - Check Jason Brownlee's blog post [How to Develop Word-Based Neural Language Models in Python with Keras](https://machinelearningmastery.com/develop-word-based-neural-language-models-python-keras/)
