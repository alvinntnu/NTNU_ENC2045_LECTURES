#!/usr/bin/env python
# coding: utf-8

# # Deep Learning: Sentiment Analysis

# - Let's get back to the Senitment Analysis on the NLTK Movie Reviews datasets 

# ![](../images/keras-workflow.png)

# ## Prepare Data

# In[1]:


import numpy as np
import nltk
from nltk.corpus import movie_reviews
import random


# In[2]:


documents = [(' '.join(list(movie_reviews.words(fileid))), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

documents = [(text,1) if label=="pos" else (text, 0) for (text, label) in documents]

random.shuffle(documents)


# In[3]:


documents[1]


# ## Train-Test Split

# In[4]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(documents, test_size = 0.1, random_state=42)
print(len(train_set), len(test_set))


# ## Prepare Input and Output Tensors

# - In deep learning, words or characters are automatically converted into numeric representations.
# - In other words, the feature engineering step is fully automatic.

# - Steps:
#     - Text to Integers
#     - Padding each instance to be of same lengths
#     

# In[5]:


import tensorflow as tf
import tensorflow.keras as keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical, plot_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import SpatialDropout1D
from keras.layers import Bidirectional


# In[6]:


texts = [n for (n, l) in train_set]
labels = [l for (n, l) in train_set] 


# In[7]:


print(len(texts))
print(len(labels))


# ### Tokenizer

# - By default, the token index 0 is reserved for padding token.
# - If `oov_token` is specified, it is default to index 1.
# - Specify `num_words` for tokenizer to include only top N words in the model
# - Tokenizer will automatically remove puntuations.
# - Tokenizer use whitespace as word delimiter.
# - If every character is treated as a token, specify `char_level=True`.

# In[8]:


NUM_WORDS = 10000
tokenizer = Tokenizer(num_words = NUM_WORDS)
tokenizer.fit_on_texts(texts)


# ### Vocabulary

# - When computing the vocabulary size, the plus 1 is due to the addition of the padding token.
# - if `oov_token` is specified, then the vocabulary size needs to be added one more.

# In[9]:


# determine the vocabulary size
# vocab_size = len(tokenizer.word_index) + 1
vocab_size = tokenizer.num_words + 1
print('Vocabulary Size: %d' % vocab_size)


# In[10]:


list(tokenizer.word_index.items())[:20]


# In[11]:


len(tokenizer.word_index)


# ## Define X and Y (Text Vectorization)

# ### From Texts and Tensors
# 
# - There are two main ways of text vectorization:
#     - Texts to Matrix: **One-hot encoding** of texts (similar to bag-of-words model)
#     - Texts to Sequences: **Integer encoding** of all word tokens in texts and we will learn token **embeddings** along with the networks
#     

# ### Method 1: Text to Sequences
# 
# - Text to sequences (integers)
# - Pad sequences

# #### Text to Sequences

# In[12]:


texts_ints = tokenizer.texts_to_sequences(texts)


# #### Padding

# :::{tip}
# When dealing with texts and documents, padding each text to the maximum length may not be ideal. For example, for sentiment classification, it is usually the case that authors would highlight more his/her sentiment at the end of the text. Therefore, we can specify an arbitrary `max_len` in padding the sequences to (a) reduce the risk of including too much noise in our model, and (b) speed up the training steps.
# :::

# In[13]:


texts_lens=[len(n) for n in texts_ints]
texts_lens
import seaborn as sns
sns.displot(texts_lens)


# In[14]:


max_len = texts_lens[np.argmax(texts_lens)]
max_len


# - In this tutorial, we consider only the **final** 400 tokens of each text.
# - `padding` and `truncating` parameters in `pad_sequences`: whether to Pre-padding or removing values from the beginning of the sequence (i.e., `pre`) or the other way (`post`).

# In[15]:


max_len = 400


# In[16]:


texts_ints_pad = sequence.pad_sequences(texts_ints, maxlen = max_len, truncating='pre', padding='pre')
texts_ints_pad[:10]


# In[17]:


X_train = np.array(texts_ints_pad).astype('int32')
y_train = np.array(labels)

X_test_texts = [n for (n, l) in test_set]
X_test = np.array(
    sequence.pad_sequences(tokenizer.texts_to_sequences(X_test_texts),
                           maxlen=max_len,
                           padding='pre',
                           truncating='pre')).astype('int32')
y_test = np.array([l for (n, l) in test_set])


# In[18]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# ### Method 2: Text to Matrix (One-hot Encoding/Bag-of-Words)

# In[19]:


texts_matrix = tokenizer.texts_to_matrix(texts, mode="binary")


# In[20]:


X_train2 = np.array(texts_matrix).astype('int32')
y_train2 = np.array(labels)

X_test2 = tokenizer.texts_to_matrix(X_test_texts, mode="binary").astype('int32')
y_test2 = np.array([l for (n,l) in test_set])


# In[21]:


print(X_train2.shape)
print(y_train2.shape)
print(X_test2.shape)
print(y_test2.shape)


# ## Model Definition

# In[22]:


import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.rcParams['figure.dpi'] = 150

# Plotting results
def plot(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc)+1)
    ## Accuracy plot
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    ## Loss plot
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    
def plot2(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    #plt.gca().set_ylim(0,1)
    plt.show()


# ### Model 1
# 
# - Two layers of fully-connected dense layers
# - The input is the one-hot encoding of the text from text-to-matrix.

# ![](../images/movie-review-classifier-dl/movie-review-classifier-dl.002.jpeg)

# In[23]:


from keras import layers
model1 = keras.Sequential()
model1.add(keras.Input(shape=(NUM_WORDS,)))
model1.add(layers.Dense(16, activation="relu", name="dense_layer_1"))
model1.add(layers.Dense(16, activation="relu", name="dense_layer_2"))
model1.add(layers.Dense(1, activation="sigmoid", name="output"))

model1.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)


# In[24]:


plot_model(model1, show_shapes=True)


# In[25]:


## A few DL hyperparameters
BATCH_SIZE = 128
EPOCHS = 25
VALIDATION_SPLIT = 0.2


# In[26]:


history1 = model1.fit(X_train2, y_train2, 
                    batch_size=BATCH_SIZE, 
                    epochs=EPOCHS, verbose=2,
                   validation_split = VALIDATION_SPLIT)


# In[27]:


plot2(history1)


# In[28]:


model1.evaluate(X_test2, y_test2, batch_size=BATCH_SIZE, verbose=2)


# ### Model 2
# 
# - One Embedding Layer + Two layers of fully-connected dense layers
# - The Input is the integer encodings of texts from the padded text-to-sequence.

# ![](../images/movie-review-classifier-dl/movie-review-classifier-dl.004.jpeg)

# ![](../images/movie-review-classifier-dl/movie-review-classifier-dl.008.jpeg)

# In[29]:


EMBEDDING_DIM = 128
model2 = Sequential()
model2.add(
    Embedding(input_dim=vocab_size,
              output_dim=EMBEDDING_DIM,
              input_length=max_len,
              mask_zero=True))
model2.add(
    layers.GlobalAveragePooling1D()
)  ## The GlobalAveragePooling1D layer returns a fixed-length output vector for each example by averaging over the sequence dimension. This allows the model to handle input of variable length, in the simplest way possible.
model2.add(layers.Dense(16, activation="relu", name="dense_layer_1"))
model2.add(layers.Dense(16, activation="relu", name="dense_layer_2"))
model2.add(layers.Dense(1, activation="sigmoid", name="output"))

model2.compile(loss=keras.losses.BinaryCrossentropy(),
               optimizer=keras.optimizers.Adam(lr=0.001),
               metrics=["accuracy"])


# In[30]:


plot_model(model2, show_shapes=True)


# In[31]:


history2 = model2.fit(X_train, y_train, 
                    batch_size=BATCH_SIZE, 
                    epochs=EPOCHS, verbose=2,
                    validation_split = VALIDATION_SPLIT)


# In[32]:


plot2(history2)


# In[33]:


model2.evaluate(X_test, y_test, batch_size=128, verbose=2)


# ## Issues of Word/Character Representations

# - Generally speaking, we can train our word embeddings along with the downstream NLP task (e.g., the sentiment classification in our current case).
# - Another common method is to train the word embeddings using unsupervised methods on a large amount of data and apply the pre-trained word embeddings to the current downstream NLP task. Typical methods include word2vec (CBOW or skipped-gram, GloVe etc). We will come back to these later.

# ## Sequence Models

# ### Model 3
# 
# - One Embedding Layer + LSTM + Dense Layer
# - Input: the padded text-to-sequences

# ![](../images/movie-review-classifier-dl/movie-review-classifier-dl.012.jpeg)

# In[34]:


EMBEDDING_DIM = 128
model3 = Sequential()
model3.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_len, mask_zero=True))
model3.add(LSTM(16, dropout=0.2, recurrent_dropout=0.5))
model3.add(Dense(1, activation="sigmoid"))

model3.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)


# In[35]:


plot_model(model3, show_shapes=True)


# In[36]:


history3 = model3.fit(X_train, y_train, 
                    batch_size=BATCH_SIZE, 
                    epochs=EPOCHS, verbose=2,
                   validation_split = VALIDATION_SPLIT)


# In[37]:


plot2(history3)


# In[38]:


model3.evaluate(X_test, y_test, batch_size=128, verbose=2)


# ### Model 4
# 
# - One Embedding Layer + Two Stacked LSTM + Dense Layer

# ![](../images/movie-review-classifier-dl/movie-review-classifier-dl.013.jpeg)

# In[39]:


EMBEDDING_DIM = 128
model4 = Sequential()
model4.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_len, mask_zero=True))
model4.add(LSTM(16, return_sequences=True, dropout=0.2, recurrent_dropout=0.5)) #)
model4.add(LSTM(16, dropout=0.2, recurrent_dropout=0.5))
model4.add(Dense(1, activation="sigmoid"))

model4.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)


# In[40]:


plot_model(model4,show_shapes=True)


# In[41]:


history4 = model4.fit(X_train, y_train, 
                    batch_size=BATCH_SIZE, 
                    epochs=EPOCHS, verbose=2,
                   validation_split = 0.2)


# In[42]:


plot2(history4)


# In[43]:


model4.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=2)


# ### Model 5

# - Embedding Layer + Bidirectional LSTM + Dense Layer

# ![](../images/movie-review-classifier-dl/movie-review-classifier-dl.014.jpeg)

# In[44]:


EMBEDDING_DIM = 128
model5 = Sequential()
model5.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_len, mask_zero=True))
model5.add(Bidirectional(LSTM(16, dropout=0.2, recurrent_dropout=0.5)))
model5.add(Dense(1, activation="sigmoid"))

model5.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)


# In[45]:


plot_model(model5, show_shapes=True)


# In[46]:


history5 = model5.fit(X_train, y_train, 
                    batch_size=BATCH_SIZE, 
                    epochs=EPOCHS, verbose=2,
                   validation_split = 0.2)


# In[47]:


plot2(history5)


# In[48]:


model5.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=2)


# ## Even More Complex Sequence Models

# ### Model 6
# 
# - One Embedding Layer + LSTM [hidden state of last time step + cell state of last time step] + Dense Layer

# ![](../images/movie-review-classifier-dl/movie-review-classifier-dl.015.jpeg)

# In[49]:


EMBEDDING_DIM = 128

## Functional API
inputs = keras.Input(shape=(max_len,))
x=layers.Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_len, mask_zero=True)(inputs)
_,x_last_h, x_c = layers.LSTM(16, dropout=0.2, 
                               recurrent_dropout=0.5, 
                               return_sequences=False, return_state=True)(x)
## LSTM Parameters:
#     `return_seqeunces=True`: return the hidden states for each time step
#     `return_state=True`: return the cell state of the last time step
#     When both are set True, the return values of LSTM are:
#     (1) the hidden states of all time steps (when `return_sequences=True`) or the hidden state of the last time step
#     (2) the hidden state of the last time step
#     (3) the cell state of the last time step

x = layers.Concatenate(axis=1)([x_last_h, x_c])
outputs=layers.Dense(1, activation='sigmoid')(x)
model6 = keras.Model(inputs=inputs, outputs=outputs)

plot_model(model6, show_shapes=True)


# In[50]:


model6.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)
history6 = model6.fit(X_train, y_train, 
                    batch_size=BATCH_SIZE, 
                    epochs=EPOCHS, verbose=2,
                   validation_split = VALIDATION_SPLIT)


# In[51]:


plot2(history6)


# In[52]:


model6.evaluate(X_test, y_test, batch_size=128, verbose=2)


# ### Model 7
# 
# - All of the previous RNN-based models only utilize the output of the last time step from the RNN as the input of the subsequent layers.
# - We can also make all the hidden outputs at all time steps from the RNN available to the subsequent layers.
# - This is the idea of **Attention**.
# - Here we add one `AttentionLayer`, which gives us a weighted version of all the hidden states from the RNN. These outputs from AttentionLayer indicate how relevant each hidden state is to computation of the subsequent layer.
#     - Use the hidden state h as the **query** and the **key** is all the hidden states from LSTM.
#     - The Attention layer shows how the last state (query) is connected to all the previous hidden states (key).
#     - The Attention layer will return a weighted version of all the hidden states.

# ![](../images/movie-review-classifier-dl/movie-review-classifier-dl.016.jpeg)

# In[53]:


EMBEDDING_DIM = 128

inputs = keras.Input(shape=(max_len,))
x=layers.Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_len,mask_zero=True)(inputs)
x_all_hs, x_last_h, x_last_c = layers.LSTM(32, dropout=0.2, 
                               recurrent_dropout=0.5, 
                               return_sequences=True, return_state=True)(x)
## LSTM Parameters:
#     `return_seqeunces=True`: return the hidden states for each time step
#     `return_state=True`: return the cell state of the last time step
#     When both are set True, the return values of LSTM are:
#     (1) the hidden states of all time steps (when `return_sequences=True`) or the hidden state of the last time step
#     (2) the hidden state of the last time step
#     (3) the cell state of the last time step

## Self Attention
atten_out = layers.Attention()([x_all_hs, x_all_hs]) # query and key 
#x_all_hs_weighted = layers.GlobalAveragePooling1D()(atten_out)
#x_last_h_plus_x_all_hs_weighted = layers.Concatenate(axis=1)([x_last_h, x_all_hs_weighted])
atten_out_flat = layers.GlobalAveragePooling1D()(atten_out)
x = layers.Dropout(0.1)(atten_out_flat)
x = layers.Dense(16, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs=layers.Dense(1, activation='sigmoid')(x)
model7 = keras.Model(inputs=inputs, outputs=outputs)

plot_model(model7, show_shapes=True)


# In[54]:


model7.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)
history7 = model7.fit(X_train, y_train, 
                    batch_size=BATCH_SIZE, 
                    epochs=EPOCHS, verbose=2,
                   validation_split = VALIDATION_SPLIT)


# In[55]:


plot(history7)


# In[56]:


model7.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=2)


# ## Explanation

# In[57]:


from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=['negative','positive'], char_level=False)


# In[58]:


## Select the best model so far
best_model = model2

## Pipeline for LIME
def model_predict_pipeline(text):
    _seq = tokenizer.texts_to_sequences(text)
    _seq_pad = keras.preprocessing.sequence.pad_sequences(_seq, maxlen=max_len)
    return np.array([[float(1-x), float(x)] for x in best_model.predict(np.array(_seq_pad))])


# In[59]:


text_id = 3
model_predict_pipeline([X_test_texts[text_id]])


# In[60]:


text_id=3
exp = explainer.explain_instance(
X_test_texts[text_id], model_predict_pipeline, num_features=20, top_labels=1)
exp.show_in_notebook(text=True)


# In[61]:


exp.show_in_notebook(text=True)


# ## Check Embeddings

# - Let's check the word embeddings learned along with the Sentiment Classifier.

# In[62]:


word_vectors = best_model.layers[0].get_weights()[0]
word_vectors.shape


# In[63]:


token_labels = [word for (ind, word) in tokenizer.index_word.items() if ind < word_vectors.shape[0]]
token_labels.insert(0,"PAD")
token_labels[:10]


# In[64]:


len(token_labels)


# - Check embeddings of words that are not on the stopword list and whose word length >= 5 (characters)

# In[65]:


from sklearn.manifold import TSNE
stopword_list = nltk.corpus.stopwords.words('english')


# In[66]:


out_index = [i for i, w in enumerate(token_labels) if len(w)>=5 and w not in stopword_list]
len(out_index)


# In[67]:


out_index[:10]


# In[68]:


tsne = TSNE(n_components=2, random_state=0, n_iter=5000, perplexity=50)
np.set_printoptions(suppress=True)
T = tsne.fit_transform(word_vectors[out_index[:100],])
labels = list(np.array(token_labels)[out_index[:100]])

len(labels)

plt.figure(figsize=(10, 7), dpi=150)
plt.scatter(T[:, 0], T[:, 1], c='orange', edgecolors='r')
for label, x, y in zip(labels, T[:, 0], T[:, 1]):
    plt.annotate(label, xy=(x+0.01, y+0.01), xytext=(0, 0), textcoords='offset points')

