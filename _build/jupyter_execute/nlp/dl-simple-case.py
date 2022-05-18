#!/usr/bin/env python
# coding: utf-8

# # Deep Learning: A Simple Example

# - Let's get back to the Name Gender Classifier.

# ![](../images/keras-workflow.png)

# ## Prepare Data

# In[1]:


## Packages Dependencies
import os
import shutil
import numpy as np
import nltk
from nltk.corpus import names
import random

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.dpi'] = 150
    
from lime.lime_text import LimeTextExplainer

import tensorflow
import tensorflow.keras as keras

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
    # from keras.layers import Dense
    # from keras.layers import LSTM, RNN, GRU
    # from keras.layers import Embedding
    # from keras.layers import SpatialDropout1D

import kerastuner


# In[2]:


labeled_names = ([(name, 1) for name in names.words('male.txt')] +
                 [(name, 0) for name in names.words('female.txt')])
random.shuffle(labeled_names)


# ## Train-Test Split

# In[3]:


train_set, test_set = train_test_split(labeled_names,
                                       test_size=0.2,
                                       random_state=42)
print(len(train_set), len(test_set))


# In[4]:


names = [n for (n, l) in train_set] ## X
labels = [l for (n, l) in train_set] ## y


# In[5]:


len(names)


# ## Tokenizer

# - `keras.preprocessing.text.Tokenizer` is a very useful tokenizer for text processing in deep learning.
# - `Tokenizer` assumes that the **word tokens** of the input texts have been delimited by whitespaces.
# - `Tokenizer` provides the following functions:
#     - It will first create a dictionary for the entire corpus (a mapping of each word token and its unique integer index index) (`Tokenizer.fit_on_text()`)
#     - It can then use the corpus dictionary to convert **words** in each corpus text into **integer sequences** (`Tokenizer.texts_to_sequences()`)
#     - The dictionary is in `Tokenizer.word_index`

# - Notes on `Tokenizer`:
#     - By default, the token index 0 is reserved for **padding token**.
#     - If `oov_token` is specified, it is default to index 1. (Default: `oov_token=False`)
#     - Specify `num_words=N` for `Tokenizer` to include only top N words in converting texts to sequences.
#     - `Tokenizer` will automatically remove punctuations.
#     - `Tokenizer` use the **whitespace** as word-token delimiter.
#     - If every character is treated as a token, specify `char_level=True`.
#     - Please read `Tokenizer` documentation very carefully. Very important!!

# In[6]:


tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(names) ## similar to CountVectorizer.fit_transform()


# ### Vocabulary
# 
# After we `Tokenizer.fit_on_texts()`, we can take a look at the corpus dictionary, i.e., the mapping of words and their unique integer indices.

# In[7]:


# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)


# In[8]:


tokenizer.word_index


# ## Prepare Input and Output Tensors

# - Like in feature-based machine learning, a computational model only accepts numeric values. It is necessary to convert raw texts to numeric tensors for neural network.
# - After we create the `Tokenizer`, we use the `Tokenizer` to perform text vectorization, i.e., converting texts into tensors.
# - In deep learning, words or characters are automatically converted into numeric representations. In other words, the feature engineering step is fully automatic.

# ### Two Ways of Text Vectorization

# - Texts to Sequences:
#     - For each text, we convert all word tokens into **integer sequences**.
#     - These integer sequences will then be transformed into **embeddings** in the deep learning network.
#     - These embeddings are usually the basis for deep learning sequence models (i.e., RNN).
#     

# - Texts to Matrix: 
#     - For each text, we vectorize the entire text into a vector of bag-of-words representation.
#     - A **One-hot encoding** of the entire text would have all values of the text vector to be either 0 or 1, indicating the occurrence of all the words in the dictionary.
#     - We can of course use the frequency-based bag-of-words representation (similar to `CountVectorizer()`).

# ## Method 1: Text to Sequences

# ### From Texts and Sequences
# 
# - We can convert our corpus texts into integer sequences using `Tokenizer.texts_to_sequences()`.
# - Because texts vary in lengths, we use `keras.preprocessing.sequence.pad_sequence()` to pad all texts into a uniform length.
# - This step is important. This ensures that every text, when pushed into the network, has exactly the same **tensor shape**.

# In[9]:


names_ints = tokenizer.texts_to_sequences(names)


# In[10]:


print(names[:5])
print(names_ints[:5])
print(labels[:5])


# ### Padding
# 
# - When padding the all texts into uniform lengths, consider whether to pad or remove values from the beginning of the sequence (i.e., `pre`) or the other way (`post`).
# - Check `padding` and `truncating` parameters in `pad_sequences`
# - In this tutorial, we first identify the longest name, and use its length as the `max_len` and pad all names into the `max_len`.

# In[11]:


## We can check the length distribution of texts in corpus

names_lens = [len(n) for n in names_ints]
names_lens

sns.displot(names_lens)
print(names[np.argmax(names_lens)])  # longest name


# In[12]:


max_len = names_lens[np.argmax(names_lens)]
max_len


# In[13]:


names_ints_pad = sequence.pad_sequences(names_ints, maxlen=max_len)
names_ints_pad[:10]


# ### Define X and y
# 
# - So for names, we convert names into integer sequences, and pad them into the uniform length.
# - We perform exactly the same processing to the names in testing data.
# - We convert both the names (X) and labels (y) into `numpy.array`.

# In[14]:


## training data
X_train = np.array(names_ints_pad).astype('int32')
y_train = np.array(labels)


## testing data
X_test_texts = [n for (n, l) in test_set]
X_test = np.array(
    sequence.pad_sequences(tokenizer.texts_to_sequences(X_test_texts),
                           maxlen=max_len)).astype('int32')
y_test = np.array([l for (n, l) in test_set])


# In[15]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# ## Method 2: Text to Matrix

# ### One-Hot Encoding (Bag-of-Words)

# - We can convert each text to a bag-of-words vector using `Tokenzier.texts_to_matrix()`.
# - In particular, we can specify the parameter `mode`: `binary`, `count`, or `tfidf`.
# - When the `mode="binary"`, the text vector is a **one-hot encoding** vector, indicating whether a character occurs in the text or not.

# In[16]:


names_matrix = tokenizer.texts_to_matrix(names, mode="binary")
print(names_matrix.shape)


# In[17]:


print(names[2])
print(names_matrix[2,:])


# In[18]:


tokenizer.word_index


# ### Define X and Y

# In[19]:


X_train2 = np.array(names_matrix).astype('int32')
y_train2 = np.array(labels)

X_test2 = tokenizer.texts_to_matrix(X_test_texts,
                                    mode="binary").astype('int32')
y_test2 = np.array([l for (n, l) in test_set])


# In[20]:


print(X_train2.shape)
print(y_train2.shape)
print(X_test2.shape)
print(y_test2.shape)


# ## Model Definition

# - Three important steps for building a deep neural network:
#     - **Define** the model structure 
#     - **Compile** the model
#     - **Fit** the model

# - After we have defined our input and output tensors (X and y), we can define the **architecture** of our neural network model.
# - For the two ways of *name* vectorized representations, we try two different types of networks.
#     - **Text to Matrix**: Fully connected Dense Layers
#     - **Text to Sequences**: Embedding + RNN
# 

# In[21]:


# Two Versions of Plotting Functions for `history` from `model.fit()`
def plot1(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
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
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    #plt.gca().set_ylim(0,1)
    plt.show()


# ### Model 1: Fully Connected Dense Layers
# 
# - Let's try a simple neural network with two fully-connected dense layers with the Text-to-Matrix inputs.
# - That is, the input of this model is the bag-of-words representation of the entire name.

# ![](../images/name-gender-classifier-dl/name-gender-classifier-dl.002.jpeg)

# #### Dense Layer Operation
# 
# - The transformation of each Dense layer will transform the input tensor into a tensor whose dimension size is the same as the node number of the Dense layer.
# 
# ![](../images/name-gender-classifier-dl/name-gender-classifier-dl.003.jpeg)

# In[22]:


## Define Model
model1 = keras.Sequential()
model1.add(keras.Input(shape=(vocab_size, ), name="one_hot_input"))
model1.add(layers.Dense(16, activation="relu", name="dense_layer_1"))
model1.add(layers.Dense(16, activation="relu", name="dense_layer_2"))
model1.add(layers.Dense(1, activation="sigmoid", name="output"))


# In[23]:


## Compile Model
model1.compile(loss='binary_crossentropy',
               optimizer='adam',
               metrics=["accuracy"])


# In[24]:


plot_model(model1, show_shapes=True)


# #### A few hyperparameters for network training
# 
# - Batch Size: The number of inputs needed per update of the model parameter (gradient descent)
# - Epoch: How many iterations needed for training
# - Validation Split Ratio: Proportion of validation and training data split

# In[25]:


## Hyperparameters
BATCH_SIZE = 128
EPOCHS = 20
VALIDATION_SPLIT = 0.2


# In[26]:


## Fit the model
history1 = model1.fit(X_train2,
                      y_train2,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      verbose=2,
                      validation_split=VALIDATION_SPLIT)


# In[27]:


plot1(history1)


# In[28]:


model1.evaluate(X_test2, y_test2, batch_size=BATCH_SIZE, verbose=2)


# ### Model 2: Embedding + RNN

# - Another possibility is to introduce an embedding layer in the network, which transforms each **character** of the name into a tensor (i.e., embeddings), and then we add a Recurrent Neural Network (RNN) layer to process each character sequentially.
# - The strength of the RNN is that it iterates over the **timesteps** of a sequence, while maintaining an internal state that encodes information about the **timesteps** it has seen so far.
# - It is posited that after the RNN iterates through the entire sequence, it keeps important information of all previously iterated tokens for further operation.
# - The input of this network is a padded sequence of the original text (name).

# ![](../images/name-gender-classifier-dl/name-gender-classifier-dl.004.jpeg)

# In[29]:


## Define the embedding dimension
EMBEDDING_DIM = 128

## Define model
model2 = Sequential()
model2.add(
    layers.Embedding(input_dim=vocab_size,
                     output_dim=EMBEDDING_DIM,
                     input_length=max_len,
                     mask_zero=True))
model2.add(layers.SimpleRNN(16, activation="relu", name="RNN_layer"))
model2.add(layers.Dense(16, activation="relu", name="dense_layer"))
model2.add(layers.Dense(1, activation="sigmoid", name="output"))

model2.compile(loss='binary_crossentropy',
               optimizer='adam',
               metrics=["accuracy"])


# #### Embedding Layer Operation
# 
# ![](../images/name-gender-classifier-dl/name-gender-classifier-dl.005.jpeg)

# #### RNN Layer Operation
# 
# ![](../images/name-gender-classifier-dl/name-gender-classifier-dl.006.jpeg)

# #### RNN Layer Operation
# 
# ![](../images/name-gender-classifier-dl/name-gender-classifier-dl.007.jpeg)

# #### Unrolled Version of RNN Operation
# 
# ![](../images/name-gender-classifier-dl/name-gender-classifier-dl.008.jpeg)

# #### Unrolled Version of RNN Operation
# 
# ![](../images/name-gender-classifier-dl/name-gender-classifier-dl.009.jpeg)

# In[30]:


plot_model(model2, show_shapes=True)


# In[31]:


history2 = model2.fit(X_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      verbose=2,
                      validation_split=VALIDATION_SPLIT)


# In[32]:


plot1(history2)


# In[33]:


model2.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=2)


# ### Model 3: Regularization and Dropout

# - Based on the validation results of the previous two models (esp. the RNN-based model), we can see that they are probably a bit overfit because the model performance on the validation set starts to **stall** after the first few epochs.
# - We can add **regularization** and **dropouts** in our network definition to avoid overfitting.

# In[34]:


## Define embedding dimension
EMBEDDING_DIM = 128

## Define model
model3 = Sequential()
model3.add(
    layers.Embedding(input_dim=vocab_size,
                     output_dim=EMBEDDING_DIM,
                     input_length=max_len,
                     mask_zero=True))
model3.add(
    layers.SimpleRNN(16,
                     activation="relu",
                     name="RNN_layer",
                     dropout=0.2, ## dropout for input character
                     recurrent_dropout=0.2))  ## dropout for previous state
model3.add(layers.Dense(16, activation="relu", name="dense_layer"))
model3.add(layers.Dense(1, activation="sigmoid", name="output"))

model3.compile(loss='binary_crossentropy',
               optimizer='adam',
               metrics=["accuracy"])


# In[35]:


plot_model(model3, show_shapes=True)


# In[36]:


history3 = model3.fit(X_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      verbose=2,
                      validation_split=VALIDATION_SPLIT)


# In[37]:


plot1(history3)


# In[38]:


model3.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=2)


# ### Model 4: Improve the Models

# - In addition to regularization and dropouts, we can further improve the model by increasing the model **complexity**.
# - In particular, we can increase the **depths** and **widths** of the network layers.
# - Let's try stacking two RNN layers.

# :::{tip}
# 
# When we stack two sequence layers (e.g., RNN), we need to make sure that the hidden states (outputs) of the first sequence layer at **all timesteps** are properly passed onto the next sequence layer, not just the hidden state (output) of the last timestep.
# 
# In keras, this usually means that we need to set the argument `return_sequences=True` in a sequence layer (e.g., `SimpleRNN`, `LSTM`, `GRU` etc).
# 
# :::

# In[39]:


## Define embedding dimension
MBEDDING_DIM = 128

## Define model
model4 = Sequential()
model4.add(
    layers.Embedding(input_dim=vocab_size,
                     output_dim=EMBEDDING_DIM,
                     input_length=max_len,
                     mask_zero=True))
model4.add(
    layers.SimpleRNN(16,
                     activation="relu",
                     name="RNN_layer_1",
                     dropout=0.2,
                     recurrent_dropout=0.2,
                     return_sequences=True) 
)  ## To ensure the hidden states of all timesteps are pased down to next layer
model4.add(
    layers.SimpleRNN(16,
                     activation="relu",
                     name="RNN_layer_2",
                     dropout=0.2,
                     recurrent_dropout=0.2))
model4.add(layers.Dense(1, activation="sigmoid", name="output"))

## Compile model
model4.compile(loss='binary_crossentropy',
               optimizer='adam',
               metrics=["accuracy"])


# In[40]:


plot_model(model4, show_shapes=True)


# In[41]:


history4 = model4.fit(X_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      verbose=2,
                      validation_split=VALIDATION_SPLIT)


# In[42]:


plot1(history4)


# In[43]:


model4.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=2)


# ### Model 5: Bidirectional

# - We can also increase the model complexity in at least two possible ways:
#     - Use more advanced RNNs, such as LSTM or GRU
#     - Process the sequence in two directions
#     - Increase the hidden nodes of the RNN/LSTM
# - Now let's try the more sophisticated RNN, LSTM, and with bidirectional sequence processing and add more nodes to the LSTM layer.

# In[44]:


## Define embedding dimension
EMBEDDING_DIM = 128

## Define model
model5 = Sequential()
model5.add(
    layers.Embedding(input_dim=vocab_size,
                      output_dim=EMBEDDING_DIM,
                      input_length=max_len,
                      mask_zero=True))
model5.add(
    layers.Bidirectional(  ## Bidirectional sequence processing
        layers.LSTM(32,
                    activation="relu",
                    name="lstm_layer_1",
                    dropout=0.2,
                    recurrent_dropout=0.5,
                    return_sequences=True)))
model5.add(
    layers.Bidirectional(  ## Bidirectional sequence processing
        layers.LSTM(32,
                    activation="relu",
                    name="lstm_layer_2",
                    dropout=0.2,
                    recurrent_dropout=0.5)))
model5.add(layers.Dense(1, activation="sigmoid", name="output"))

model5.compile(loss='binary_crossentropy',
               optimizer='adam',
               metrics=["accuracy"])


# In[45]:


plot_model(model5, show_shapes=True)


# In[46]:


history5 = model5.fit(X_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      verbose=2,
                      validation_split=VALIDATION_SPLIT)


# In[47]:


plot1(history5)


# In[48]:


model5.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=2)


# ## Check Embeddings

# - Compared to one-hot encodings of characters, embeddings may include more information relating to the characteristics (semantics?) of the characters.
# - We can extract the embedding layer and apply dimensional reduction techniques (i.e., TSNE) to see how embeddings capture the relationships in-between characters.

# In[49]:


## A name in sequence from test set
print(X_test_texts[10])
print(X_test[10])


# In[50]:


## Extract Corpus Dictionary (mapping of chars and integer indices)
ind2char = tokenizer.index_word
[ind2char.get(i) for i in X_test[10] if ind2char.get(i) != None]


# In[51]:


## Extract the embedding layer (its weights matrix)
char_vectors = model5.layers[0].get_weights()[0]
print(char_vectors.shape) ## embedding shape (vocab_size, embed_dim)
print(char_vectors[1,:]) ## first char embeddings


# In[52]:


labels = [char for (ind, char) in tokenizer.index_word.items()]
labels.insert(0, None)
labels


# In[53]:


## Visulizing char embeddings via dimensional reduction techniques

tsne = TSNE(n_components=2, random_state=0, n_iter=5000, perplexity=3)
np.set_printoptions(suppress=True)

T = tsne.fit_transform(char_vectors)
labels = labels

plt.figure(figsize=(10, 7), dpi=150)
plt.scatter(T[:, 0], T[:, 1], c='orange', edgecolors='r')
for label, x, y in zip(labels, T[:, 0], T[:, 1]):
    plt.annotate(label,
                 xy=(x + 1, y + 1),
                 xytext=(0, 0),
                 textcoords='offset points')


# ## Issues of Word/Character Representations

# - One-hot encoding does not indicate semantic relationships between characters.
# - For deep learning NLP, it is preferred to convert one-hot encodings of words/characters into **embeddings**, which are argued to include more semantic information of the tokens.
# - Now the question is how to train and create better word embeddings. 
# - There are at least two alternatives:
#     - We can train the embeddings along with our current NLP task.
#     - We can use pre-trained embeddings from other unsupervised learning (transfer learning).
# - We will come back to this issue later.

# ## Hyperparameter Tuning

# :::{note}
# 
# Please install keras tuner module in your current conda:
# 
# ```
# pip install -U keras-tuner
# ```
# 
# or 
# 
# ```
# conda install -c conda-forge keras-tuner
# ```
# 
# :::

# - Like feature-based ML methods, neural networks also come with many **hyperparameters**, which require default values.
# - Typical hyperparameters include:
#     - Number of nodes for the layer
#     - Learning Rates
# - We can utilize the module, [`keras-tuner`](https://keras-team.github.io/keras-tuner/documentation/tuners/), to fine-tune these hyperparameters (i.e., to find the values that optimize the model performance).

# - Steps for Keras Tuner
#     - First, wrap the model definition in a function, which takes a single `hp` argument. 
#     - Inside this function, replace any value we want to tune with a call to hyperparameter sampling methods, e.g. `hp.Int()` or `hp.Choice()`. The function should return a compiled model.
#     - Next, instantiate a `tuner` object specifying our optimization objective and other search parameters.
#     - Finally, start the search with the `search()` method, which takes the same arguments as `Model.fit()` in keras.
#     - When the search is over, we can retrieve the best model and a summary of the results from the `tunner`.
# 

# In[54]:


## confirm if the right kernel is being used
# import sys
# sys.executable


# In[55]:


## Wrap model definition in a function
## and specify the parameters needed for tuning
# def build_model(hp):
#     model1 = keras.Sequential()
#     model1.add(keras.Input(shape=(max_len,)))
#     model1.add(layers.Dense(hp.Int('units', min_value=32, max_value=128, step=32), activation="relu", name="dense_layer_1"))
#     model1.add(layers.Dense(hp.Int('units', min_value=32, max_value=128, step=32), activation="relu", name="dense_layer_2"))
#     model1.add(layers.Dense(2, activation="softmax", name="output"))
#     model1.compile(
#         optimizer=keras.optimizers.Adam(
#             hp.Choice('learning_rate',
#                       values=[1e-2, 1e-3, 1e-4])),
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy'])
#     return model1


# In[56]:


## wrap model definition and compiling

def build_model(hp):
    m = Sequential()
    m.add(
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=hp.Int(
                'output_dim',  ## tuning 2
                min_value=32,
                max_value=128,
                step=32),
            input_length=max_len,
            mask_zero=True))
    m.add(
        layers.Bidirectional(
            layers.LSTM(
                hp.Int('units', min_value=16, max_value=64,
                       step=16),  ## tuning 1
                activation="relu",
                dropout=0.2,
                recurrent_dropout=0.2)))
    m.add(layers.Dense(1, activation="sigmoid", name="output"))

    m.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=["accuracy"])
    return m


# In[57]:


## This is to clean up the temp dir from the tuner
## Every time we re-start the tunner, it's better to keep the temp dir clean

if os.path.isdir('my_dir'):
    shutil.rmtree('my_dir')


# - The `max_trials` variable represents the maximum number of the hyperparameter combinations that will be tested by the tuner (because sometimes the combinations are too many and the max number can limit the time in fine-tuning).
# - The `execution_per_trial` variable is the number of models that should be built and fit for each trial for robustness purposes (i.e., consistent results).

# In[58]:


## Instantiate the tunner

tuner = kerastuner.tuners.RandomSearch(build_model,
                                       objective='val_accuracy',
                                       max_trials=10,
                                       executions_per_trial=2,
                                       directory='my_dir')


# In[59]:


## Check the tuner's search space
tuner.search_space_summary()


# In[60]:


get_ipython().run_cell_magic('time', '', '## Start tuning with the tuner\ntuner.search(X_train, y_train, validation_split=VALIDATION_SPLIT, batch_size=BATCH_SIZE)')


# In[61]:


## Retrieve the best models from the tuner
models = tuner.get_best_models(num_models=2)


# In[62]:


plot_model(models[0], show_shapes=True)


# In[63]:


## Retrieve the summary of results from the tuner
tuner.results_summary()


# ## Explanation

# ### Train Model with the Tuned Hyperparameters

# In[64]:


EMBEDDING_DIM = 128
HIDDEN_STATE=32
model6 = Sequential()
model6.add(
    layers.Embedding(input_dim=vocab_size,
                     output_dim=EMBEDDING_DIM,
                     input_length=max_len,
                     mask_zero=True))
model6.add(
    layers.Bidirectional(
        layers.LSTM(HIDDEN_STATE,
                    activation="relu",
                    name="lstm_layer",
                    dropout=0.2,
                    recurrent_dropout=0.5)))
model6.add(layers.Dense(1, activation="sigmoid", name="output"))

model6.compile(loss='binary_crossentropy',
               optimizer='adam',
               metrics=["accuracy"])
plot_model(model6)


# In[65]:


history6 = model6.fit(X_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      verbose=2,
                      validation_split=VALIDATION_SPLIT)


# In[66]:


plot2(history6)


# In[67]:


explainer = LimeTextExplainer(class_names=['female','male'], char_level=True)


# In[68]:


def model_predict_pipeline(text):
    _seq = tokenizer.texts_to_sequences(text)
    _seq_pad = keras.preprocessing.sequence.pad_sequences(_seq, maxlen=max_len)
    return np.array([[float(1-x), float(x)] for x in model6.predict(np.array(_seq_pad))])
    #return model6.predict(np.array(_seq_pad))


# In[69]:


text_id = 12
print(X_test_texts[text_id])
model_predict_pipeline([X_test_texts[text_id]])


# In[70]:


exp = explainer.explain_instance(X_test_texts[text_id],
                                 model_predict_pipeline,
                                 num_features=10,
                                 top_labels=1)


# In[71]:


exp.show_in_notebook(text=True)


# In[72]:


y_test[text_id]


# In[73]:


exp = explainer.explain_instance('Tim',
                                 model_predict_pipeline,
                                 num_features=10,
                                 top_labels=1)
exp.show_in_notebook(text=True)


# In[74]:


exp = explainer.explain_instance('Michaelis',
                                 model_predict_pipeline,
                                 num_features=10,
                                 top_labels=1)
exp.show_in_notebook(text=True)


# In[75]:


exp = explainer.explain_instance('Sidney',
                                 model_predict_pipeline,
                                 num_features=10,
                                 top_labels=1)
exp.show_in_notebook(text=True)


# In[76]:


exp = explainer.explain_instance('Timber',
                                 model_predict_pipeline,
                                 num_features=10,
                                 top_labels=1)
exp.show_in_notebook(text=True)


# In[77]:


exp = explainer.explain_instance('Alvin',
                                 model_predict_pipeline,
                                 num_features=10,
                                 top_labels=1)
exp.show_in_notebook(text=True)


# ## References
# 
# - Chollet (2017), Ch 3 and  Ch 4
