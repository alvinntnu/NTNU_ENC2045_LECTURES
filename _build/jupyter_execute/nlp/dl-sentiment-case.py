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

# labeled_names = ([(name, 1) for name in names.words('male.txt')] +
#                  [(name, 0) for name in names.words('female.txt')])
# random.shuffle(labeled_names)


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


# ## Text Vectorization

# ### From Texts and Tensors
# 
# - Like in feature-based machine translation, a computational model only accepts numeric values. It is necessary to convert raw text to numeric tensor for neural network.
# - There are two main ways of text vectorization:
#     - Texts to Matrix: **One-hot encoding** of texts (similar to bag-of-words model)
#     - Texts to Sequences: **Integer encoding** of tokens in texts and learn token **embeddings**

# ## Define X and Y

# ### Method 1: Text to Sequences
# 
# - Text to sequences (integers)
# - Pad sequences

# #### Text to Sequences

# In[12]:


texts_ints = tokenizer.texts_to_sequences(texts)


# #### Padding
# 
# - To make sure each input text consists of the same number of tokens.

# In[13]:


texts_lens=[len(n) for n in texts_ints]
texts_lens
import seaborn as sns
sns.displot(texts_lens)
#print(texts[np.argmax(texts_lens)]) # longest name


# In[14]:


max_len = texts_lens[np.argmax(texts_lens)]
max_len


# - We consider the final 400 tokens of each text.
# - `padding` and `truncating` parameters in `pad_sequences`: whether to Pre-padding or removing values from the beginning of the sequence (i.e., `pre`) or the other way (`post`).

# In[15]:


max_len = 400


# In[16]:


texts_ints_pad = sequence.pad_sequences(texts_ints, maxlen = max_len, truncating='pre', padding='pre')
texts_ints_pad[:10]


# In[17]:


X_train = np.array(texts_ints_pad).astype('int32')
y_train = np.array(labels)

X_test = np.array(
    sequence.pad_sequences(tokenizer.texts_to_sequences(
        [n for (n, l) in test_set]),
                           maxlen=max_len,
                           padding='pre',
                           truncating='pre')).astype('int32')
y_test = np.array([l for (n, l) in test_set])

X_test_texts = [n for (n, l) in test_set]


# In[18]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# ### Method 2: Text to Matrix (One-hot Encoding)
# 
# - text to matrix (one-hot encode)
# - choose modes for bag-of-words (binary, count, tfidf)

# In[19]:


texts_matrix = tokenizer.texts_to_matrix(texts, mode="count")


# In[20]:


X_train2 = np.array(texts_matrix).astype('int32')
y_train2 = np.array(labels)

X_test2 = tokenizer.texts_to_matrix([n for (n,l) in test_set], mode="count").astype('int32')
y_test2 = np.array([l for (n,l) in test_set])

X_test2_texts = [n for (n,l) in test_set]


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
# Plotting results
def plot(history):

    matplotlib.rcParams['figure.dpi'] = 100
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

# In[23]:


from keras import layers
model1 = keras.Sequential()
model1.add(keras.Input(shape=(NUM_WORDS,)))
model1.add(layers.Dense(16, activation="relu", name="dense_layer_1"))
model1.add(layers.Dense(16, activation="relu", name="dense_layer_2"))
model1.add(layers.Dense(2, activation="softmax", name="output"))

model1.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)


# In[24]:


plot_model(model1, show_shapes=True )


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

# In[29]:


EMBEDDING_DIM = 128
model2 = Sequential()
model2.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_len, mask_zero=True))
model2.add(layers.GlobalAveragePooling1D()) ## The GlobalAveragePooling1D layer returns a fixed-length output vector for each example by averaging over the sequence dimension. This allows the model to handle input of variable length, in the simplest way possible.
model2.add(layers.Dense(16, activation="relu", name="dense_layer_1"))
model2.add(layers.Dense(16, activation="relu", name="dense_layer_2"))
model2.add(layers.Dense(2, activation="softmax", name="output"))

model2.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)


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


# ## Check Embeddings

# - Compared to one-hot encodings of characters, embeddings may include more information relating to the characteristics of the characters.
# - We can extract the embedding layer and apply dimensional reduction techniques (i.e., TSNE) to see how embeddings capture the relationships in-between characters.

# In[34]:


ind2word = tokenizer.index_word


# In[35]:


# check first N words for text
' '.join([ind2word.get(i) for i in X_test[1][-50:] if ind2word.get(i)!= None])


# In[36]:


X_test_texts[1][-287:]


# In[37]:


word_vectors = model2.layers[0].get_weights()[0]
word_vectors.shape


# In[38]:


token_labels = [word for (ind, word) in tokenizer.index_word.items()]
token_labels.insert(0,None)
token_labels[:10]


# In[39]:


from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=0, n_iter=5000, perplexity=3)
np.set_printoptions(suppress=True)
T = tsne.fit_transform(word_vectors[:100,])
labels = labels

plt.figure(figsize=(10, 7), dpi=150)
plt.scatter(T[:, 0], T[:, 1], c='orange', edgecolors='r')
for label, x, y in zip(token_labels, T[:, 0], T[:, 1]):
    plt.annotate(label, xy=(x+0.01, y+0.01), xytext=(0, 0), textcoords='offset points')


# ## Issues of Word/Character Representations

# - One-hot encoding does not indicate semantic relationships between characters.
# - For deep learning NLP, it is preferred to convert one-hot encodings of words/characters into embeddings, which are argued to include more semantic information of the tokens.
# - Now the question is how to train and create better word embeddings. We will come back to this issue later.

# - Generally speaking, we can train our word embeddings along with the downstream NLP task (e.g., the sentiment classification in our current case).
# - Another common method is to train the word embeddings using unsupervised methods on a large amount of data and apply the pre-trained word embeddings to the current downstream NLP task. Typical methods include word2vec (CBOW or skipped-gram, GloVe etc). We will come back to these later.

# ## Hyperparameter Tuning

# :::{note}
# 
# Please install keras tuner module in your current conda:
# ```
# pip install -U keras-tuner
# ```
# 
# :::

# - Like feature-based ML methods, neural networks also come with many hyperparameters, which require default values.
# - Typical hyperparameters include:
#     - Number of nodes for the layer
#     - Learning Rates
# - We can utilize the module, [`keras-tuner`](https://keras-team.github.io/keras-tuner/documentation/tuners/), to fine-tune the hyperparameters.

# - Steps for Keras Tuner
#     - First, wrap the model definition in a function, which takes a single `hp` argument. 
#     - Inside this function, replace any value we want to tune with a call to hyperparameter sampling methods, e.g. `hp.Int()` or `hp.Choice()`. The function should return a compiled model.
#     - Next, instantiate a tuner object specifying your optimization objective and other search parameters.
#     - Finally, start the search with the `search()` method, which takes the same arguments as `Model.fit()` in keras.
#     - When search is over, we can retrieve the best model and a summary of the results from the `tunner`.
# 

# In[40]:


import kerastuner


# In[41]:


## Wrap model definition in a function
## and specify the parameters needed for tuning
def build_model(hp):
    model1 = keras.Sequential()
    model1.add(keras.Input(shape=(max_len,)))
    model1.add(layers.Dense(hp.Int('units', min_value=32, max_value=128, step=32), activation="relu", name="dense_layer_1"))
    model1.add(layers.Dense(hp.Int('units', min_value=32, max_value=128, step=32), activation="relu", name="dense_layer_2"))
    model1.add(layers.Dense(2, activation="softmax", name="output"))
    model1.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate',
                      values=[1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model1

# def build_model(hp):
#     inputs = keras.Input(shape=(784,))
#     x = layers.Dense(
#         units=hp.Int('units', min_value=32, max_value=512, step=32),
#         activation='relu'))(inputs)
#     outputs = layers.Dense(10, activation='softmax')(x)
#     model = keras.Model(inputs, outputs)
#     model.compile(
#         optimizer=keras.optimizers.Adam(
#             hp.Choice('learning_rate',
#                       values=[1e-2, 1e-3, 1e-4])),
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy'])
#     return model


# In[42]:


## This is to clean up the temp dir from the tuner
## Every time we re-start the tunner, it's better to keep the temp dir clean

import os
import shutil

if os.path.isdir('my_dir'):
    shutil.rmtree('my_dir')
    


# In[43]:


## Instantiate the tunner

tuner = kerastuner.tuners.RandomSearch(
  build_model,
  objective='val_accuracy',
  max_trials=10,
  executions_per_trial=3,
  directory='my_dir')


# In[44]:


## Check the tuner's search space
tuner.search_space_summary()


# In[45]:


## Start tuning with the tuner
tuner.search(X_train, y_train, validation_split=0.2, batch_size=128)


# In[46]:


## Retrieve the best models from the tuner
models = tuner.get_best_models(num_models=2)


# In[47]:


## Retrieve the summary of results from the tuner
tuner.results_summary()


# ## Sequence Models

# ### Model 3
# 
# - One Embedding Layer + LSTM + Dense Layer
# - Input: the padded text-to-sequences

# In[48]:


EMBEDDING_DIM = 128
model3 = Sequential()
model3.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_len, mask_zero=True))
#model3.add(SpatialDropout1D(0.2))
model3.add(LSTM(16, dropout=0.5, recurrent_dropout=0.5))
model3.add(Dense(2, activation="softmax"))

model3.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)


# In[49]:


plot_model(model3, show_shapes=True)


# In[50]:


history3 = model3.fit(X_train, y_train, 
                    batch_size=BATCH_SIZE, 
                    epochs=EPOCHS, verbose=2,
                   validation_split = VALIDATION_SPLIT)


# In[51]:


plot2(history3)


# In[52]:


model3.evaluate(X_test, y_test, batch_size=128, verbose=2)


# ### Model 4
# 
# - One Embedding Layer + Two Stacked LSTM + Dense Layer

# In[53]:


EMBEDDING_DIM = 128
model4 = Sequential()
model4.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_len, mask_zero=True))
model4.add(LSTM(16, return_sequences=True, dropout=0.1, recurrent_dropout=0.5)) #)
model4.add(LSTM(16, dropout=0.1, recurrent_dropout=0.5))
model4.add(Dense(2, activation="softmax"))

model4.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)


# In[54]:


plot_model(model4,show_shapes=True)


# In[55]:


history4 = model4.fit(X_train, y_train, 
                    batch_size=BATCH_SIZE, 
                    epochs=EPOCHS, verbose=2,
                   validation_split = 0.2)


# In[56]:


plot2(history4)


# ### Model 5

# - Embedding Layer + Bidirectional LSTM + Dense Layer

# In[57]:


EMBEDDING_DIM = 128
model5 = Sequential()
model5.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_len, mask_zero=True))
#model3.add(SpatialDropout1D(0.2))
model5.add(Bidirectional(LSTM(16, dropout=0.5, recurrent_dropout=0.5)))
model5.add(Dense(2, activation="softmax"))

model5.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)


# In[58]:


plot_model(model5)


# In[59]:


history5 = model5.fit(X_train, y_train, 
                    batch_size=BATCH_SIZE, 
                    epochs=EPOCHS, verbose=2,
                   validation_split = 0.2)


# In[60]:


plot2(history5)


# ### Model 6
# 
# - One Embedding Layer + LSTM [hidden state of last time step + cell state of last time step] + Dense Layer

# In[61]:


EMBEDDING_DIM = 128

inputs = keras.Input(shape=(max_len,))
x=layers.Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_len, mask_zero=True)(inputs)
#x=layers.SpatialDropout1D(0.2)(x)
x_all_h,x_last_h, x_c = layers.LSTM(16, dropout=0.2, 
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
outputs=layers.Dense(2, activation='softmax')(x)
model6 = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")

plot_model(model6, show_shapes=True)


# In[62]:


model6.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)
history6 = model6.fit(X_train, y_train, 
                    batch_size=BATCH_SIZE, 
                    epochs=EPOCHS, verbose=2,
                   validation_split = VALIDATION_SPLIT)


# In[63]:


plot2(history6)


# In[64]:


model6.evaluate(X_test, y_test, batch_size=128, verbose=2)


# ### Model 7
# 
# - Adding AttentionLayer
#     - Use the hidden state h of the last time step and the cell state c of the last time step
#     - Check their attention
#     - And use [attention out + hidden state h of the last time step] for decision

# In[65]:


EMBEDDING_DIM = 128

inputs = keras.Input(shape=(max_len,))
x=layers.Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_len)(inputs)
#x=layers.SpatialDropout1D(0.2)(x)
x_all_hs, x_last_h, x_last_c = layers.LSTM(16, dropout=0.2, 
                               recurrent_dropout=0.5, 
                               return_sequences=True, return_state=True)(x)
## LSTM Parameters:
#     `return_seqeunces=True`: return the hidden states for each time step
#     `return_state=True`: return the cell state of the last time step
#     When both are set True, the return values of LSTM are:
#     (1) the hidden state of the last time step
#     (2) the hidden states of all time steps (when `return_sequences=True`) or the hidden state of the last time step
#     (3) the cell state of the last time step


atten_out = layers.Attention()([x_last_h, x_last_c])

x = layers.Concatenate(axis=1)([x_last_h, atten_out])
outputs=layers.Dense(2, activation='softmax')(x)
model7 = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")

plot_model(model7, show_shapes=True)


# In[66]:


model7.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)
history7 = model7.fit(X_train, y_train, 
                    batch_size=BATCH_SIZE, 
                    epochs=EPOCHS, verbose=2,
                   validation_split = VALIDATION_SPLIT)


# In[67]:


plot(history6)


# ## Explanation

# In[68]:


from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=['negative','positive'], char_level=False)


# In[69]:


def model_predict_pipeline(text):
    _seq = tokenizer.texts_to_sequences(text)
    _seq_pad = keras.preprocessing.sequence.pad_sequences(_seq, maxlen=max_len)
    #return np.array([[float(1-x), float(x)] for x in model.predict(np.array(_seq_pad))])
    return model2.predict(np.array(_seq_pad))



# np.array(sequence.pad_sequences(
#     tokenizer.texts_to_sequences([n for (n,l) in test_set]),
#     maxlen = max_len)).astype('float32')


# In[70]:


reversed_word_index = dict([(index, word) for (word, index) in tokenizer.word_index.items()])


# In[71]:


text_id =150


# In[72]:


X_test[text_id]


# In[73]:


X_test_texts[text_id]


# In[74]:


' '.join([reversed_word_index.get(i, '?') for i in X_test[text_id]])


# In[75]:


print(X_test[22])
print(X_test_texts[22])


# In[76]:


X_test_texts[text_id]


# In[77]:


model_predict_pipeline([X_test_texts[text_id]])


# In[78]:


text_id=3
exp = explainer.explain_instance(
X_test_texts[text_id], model_predict_pipeline, num_features=20, top_labels=1)
exp.show_in_notebook(text=True)


# In[79]:


exp.show_in_notebook(text=True)


# In[80]:


y_test[text_id]

