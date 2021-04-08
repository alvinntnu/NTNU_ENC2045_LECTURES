# Deep Learning: A Simple Example

- Let's get back to the Name Gender Classifier.

## Prepare Data

import numpy as np
import nltk
from nltk.corpus import names
import random

labeled_names = ([(name, 1) for name in names.words('male.txt')] +
                 [(name, 0) for name in names.words('female.txt')])
random.shuffle(labeled_names)

## Train-Test Split

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(labeled_names, test_size = 0.2, random_state=42)
print(len(train_set), len(test_set))

## Feature Engineering

- In deep learning, words or characters are automatically converted into numeric representations.
- In other words, the feature engineering step is fully automatic.

- Steps:
    - Text to Integers
    - Padding each instance to be of same lengths
    

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

names = [n for (n, l) in train_set]
labels = [l for (n, l) in train_set] 

len(names)

### Tokenizer

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(names)

### Text to Sequences

names_ints = tokenizer.texts_to_sequences(names)

print(names[:10])
print(names_ints[:10])
print(labels[:10])

### Vocabulary

# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

tokenizer.word_index

### Padding

names_lens=[len(n) for n in names_ints]
names_lens
import seaborn as sns
sns.displot(names_lens)
print(names[np.argmax(names_lens)]) # longest name

max_len = names_lens[np.argmax(names_lens)]
max_len

names_ints_pad = sequence.pad_sequences(names_ints, maxlen = max_len)
names_ints_pad[:10]

## Define X and Y

X_train = np.array(names_ints_pad).astype('float32')
y_train = np.array(labels)

X_test = np.array(sequence.pad_sequences(
    tokenizer.texts_to_sequences([n for (n,l) in test_set]),
    maxlen = max_len)).astype('float32')
y_test = np.array([l for (n,l) in test_set])

X_test_texts = [n for (n,l) in test_set]

X_train.shape

X_train[2,]

## Model Definition

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
## Plotting results
# def plot(history):

#     matplotlib.rcParams['figure.dpi'] = 100
#     acc = history.history['accuracy']
#     val_acc = history.history['val_accuracy']
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']

#     epochs = range(1, len(acc)+1)
#     ## Accuracy plot
#     plt.plot(epochs, acc, 'bo', label='Training acc')
#     plt.plot(epochs, val_acc, 'b', label='Validation acc')
#     plt.title('Training and validation accuracy')
#     plt.legend()
#     ## Loss plot
#     plt.figure()

#     plt.plot(epochs, loss, 'bo', label='Training loss')
#     plt.plot(epochs, val_loss, 'b', label='Validation loss')
#     plt.title('Training and validation loss')
#     plt.legend()
#     plt.show()

    
def plot(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    #plt.gca().set_ylim(0,1)
    plt.show()

### Model 1

- Two layers of fully-connected dense layers

from keras import layers
model1 = keras.Sequential()
model1.add(keras.Input(shape=(max_len,)))
model1.add(layers.Dense(128, activation="relu", name="dense_layer_1"))
model1.add(layers.Dense(128, activation="relu", name="dense_layer_2"))
model1.add(layers.Dense(2, activation="softmax", name="output"))

model1.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)


plot_model(model1, show_shapes=True )

history1 = model1.fit(X_train, y_train, 
                    batch_size=128, 
                    epochs=50, verbose=2,
                   validation_split = 0.2)

plot(history1)

model1.evaluate(X_test, y_test, batch_size=128, verbose=2)

### Model 2

- One Embedding Layer + Two layers of fully-connected dense layers

EMBEDDING_DIM = 128
model2 = Sequential()
model2.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_len, mask_zero=True))
model2.add(layers.GlobalAveragePooling1D()) ## The GlobalAveragePooling1D layer returns a fixed-length output vector for each example by averaging over the sequence dimension. This allows the model to handle input of variable length, in the simplest way possible.
model2.add(layers.Dense(128, activation="relu", name="dense_layer_1"))
model2.add(layers.Dense(128, activation="relu", name="dense_layer_2"))
model2.add(layers.Dense(2, activation="softmax", name="output"))

model2.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)

plot_model(model2, show_shapes=True)

history2 = model2.fit(X_train, y_train, 
                    batch_size=128, 
                    epochs=50, verbose=2,
                    validation_split = 0.2)

plot(history2)

model2.evaluate(X_test, y_test, batch_size=128, verbose=2)

## Check Embeddings

- Compared to one-hot encodings of characters, embeddings may include more information relating to the characteristics of the characters.
- We can extract the embedding layer and apply dimensional reduction techniques (i.e., TSNE) to see how embeddings capture the relationships in-between characters.

ind2char = tokenizer.index_word
[ind2char.get(i) for i in X_test[10]]

char_vectors = model2.layers[0].get_weights()[0]
char_vectors.shape

labels = [char for (ind, char) in tokenizer.index_word.items()]
labels.insert(0,None)
labels

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=0, n_iter=5000, perplexity=2)
np.set_printoptions(suppress=True)
T = tsne.fit_transform(char_vectors)
labels = labels

plt.figure(figsize=(10, 7), dpi=150)
plt.scatter(T[:, 0], T[:, 1], c='orange', edgecolors='r')
for label, x, y in zip(labels, T[:, 0], T[:, 1]):
    plt.annotate(label, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points')

## Issues of Word/Character Representations

- One-hot encoding does not indicate semantic relationships between characters.
- For deep learning NLP, it is preferred to convert one-hot encodings of words/characters into embeddings, which are argued to include more semantic information of the tokens.
- Now the question is how to train and create better word embeddings. We will come back to this issue later.

## Hyperparameter Tuning

- Like feature-based ML methods, neural networks also come with many hyperparameters, which require default values.
- Typical hyperparameters include:
    - Number of nodes for the layer
    - Learning Rates
- We can utilize the module, [`kerastuner`](https://keras-team.github.io/keras-tuner/documentation/tuners/), to fine-tune the hyperparameters.

- Steps for Keras Tuner
    - First, wrap the model definition in a function, which takes a single `hp` argument. 
    - Inside this function, replace any value we want to tune with a call to hyperparameter sampling methods, e.g. `hp.Int()` or `hp.Choice()`. The function should return a compiled model.
    - Next, instantiate a tuner object specifying your optimization objective and other search parameters.
    - Finally, start the search with the `search()` method, which takes the same arguments as `Model.fit()` in keras.
    - When search is over, we can retrieve the best model and a summary of the results from the `tunner`.


import kerastuner

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

## This is to clean up the temp dir from the tuner
## Every time we re-start the tunner, it's better to keep the temp dir clean

import os
import shutil

if os.path.isdir('my_dir'):
    shutil.rmtree('my_dir')
    

## Instantiate the tunner

tuner = kerastuner.tuners.RandomSearch(
  build_model,
  objective='val_accuracy',
  max_trials=10,
  executions_per_trial=3,
  directory='my_dir')

## Check the tuner's search space
tuner.search_space_summary()

## Start tuning with the tuner
tuner.search(X_train, y_train, validation_split=0.2, batch_size=128)

## Retrieve the best models from the tuner
models = tuner.get_best_models(num_models=2)

## Retrieve the summary of results from the tuner
tuner.results_summary()

## Sequence Models

### Model 3

- One Embedding Layer + LSTM + Dense Layer

EMBEDDING_DIM = 128
model3 = Sequential()
model3.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_len, mask_zero=True))
#model3.add(SpatialDropout1D(0.2))
model3.add(LSTM(64))# , dropout=0.2, recurrent_dropout=0.2))
model3.add(Dense(2, activation="softmax"))

model3.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)

plot_model(model3, show_shapes=True)

history3 = model3.fit(X_train, y_train, 
                    batch_size=128, 
                    epochs=50, verbose=2,
                   validation_split = 0.2)

plot(history3)

### Model 4

- One Embedding Layer + Two Stacked LSTM + Dense Layer

EMBEDDING_DIM = 128
model4 = Sequential()
model4.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_len, mask_zero=True))
#model.add(SpatialDropout1D(0.2))
model4.add(LSTM(64, return_sequences=True)) #, dropout=0.2, recurrent_dropout=0.2))
model4.add(LSTM(64))
model4.add(Dense(2, activation="softmax"))

model4.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)

plot_model(model4,show_shapes=True)

history4 = model4.fit(X_train, y_train, 
                    batch_size=128, 
                    epochs=50, verbose=2,
                   validation_split = 0.2)

plot(history4)

### Model 5

- One Embedding Layer + LSTM [hidden state of last time step + cell state of last time step] + Dense Layer

EMBEDDING_DIM = 128

inputs = keras.Input(shape=(max_len,))
x=layers.Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_len, mask_zero=True)(inputs)
#x=layers.SpatialDropout1D(0.2)(x)
x_all_h,x_last_h, x_c = layers.LSTM(64, dropout=0.2, 
                               recurrent_dropout=0.2, 
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
model5 = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")

plot_model(model5, show_shapes=True)

model5.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)
history5 = model5.fit(X_train, y_train, 
                    batch_size=128, 
                    epochs=50, verbose=2,
                   validation_split = 0.2)

plot(history5)

model5.evaluate(X_test, y_test, batch_size=128, verbose=2)

### Model 6

- Adding AttentionLayer
    - Use the hidden state h of the last time step and the cell state c of the last time step
    - Check their attention
    - And use [attention out + hidden state h of the last time step] for decision

EMBEDDING_DIM = 128

inputs = keras.Input(shape=(max_len,))
x=layers.Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_len)(inputs)
#x=layers.SpatialDropout1D(0.2)(x)
x_all_hs, x_last_h, x_last_c = layers.LSTM(64, dropout=0.2, 
                               recurrent_dropout=0.2, 
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
model6 = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")

plot_model(model6, show_shapes=True)

model6.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)
history6 = model6.fit(X_train, y_train, 
                    batch_size=128, 
                    epochs=50, verbose=2,
                   validation_split = 0.2)

plot(history6)

## Explanation

from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=['female','male'], char_level=True)

def model_predict_pipeline(text):
    _seq = tokenizer.texts_to_sequences(text)
    _seq_pad = keras.preprocessing.sequence.pad_sequences(_seq, maxlen=max_len)
    #return np.array([[float(1-x), float(x)] for x in model.predict(np.array(_seq_pad))])
    return model6.predict(np.array(_seq_pad))



# np.array(sequence.pad_sequences(
#     tokenizer.texts_to_sequences([n for (n,l) in test_set]),
#     maxlen = max_len)).astype('float32')

reversed_word_index = dict([(index, word) for (word, index) in tokenizer.word_index.items()])

text_id =305

X_test[text_id]

X_test_texts[text_id]

' '.join([reversed_word_index.get(i, '?') for i in X_test[text_id]])

print(X_test[22])
print(X_test_texts[22])

X_test_texts[text_id]

model_predict_pipeline([X_test_texts[text_id]])

exp = explainer.explain_instance(
X_test_texts[text_id], model_predict_pipeline, num_features=100, top_labels=1)

exp.show_in_notebook(text=True)

y_test[text_id]

exp = explainer.explain_instance(
'Alvin', model_predict_pipeline, num_features=100, top_labels=1)
exp.show_in_notebook(text=True)

exp = explainer.explain_instance(
'Michaelis', model_predict_pipeline, num_features=100, top_labels=1)
exp.show_in_notebook(text=True)

exp = explainer.explain_instance(
'Sidney', model_predict_pipeline, num_features=100, top_labels=1)
exp.show_in_notebook(text=True)

exp = explainer.explain_instance(
'Timber', model_predict_pipeline, num_features=100, top_labels=1)
exp.show_in_notebook(text=True)