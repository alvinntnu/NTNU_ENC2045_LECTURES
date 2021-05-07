# Sequence Models Intuition

- In deep learning NLP, sequence models are often the most widely adopted methods.
- In this tutorial, we will go over the intuitions of sequence models.

## Why Sequence Models?

- Humans process texts in a sequential order (from left to right)
- When we process texts and make classification, we utilize our reasoning about previous sequences to inform later decision making.

- **Recurrent neural network** (RNN) addresses this issue by implementing networks with loops that allow information to persist.
- The loop allows information to be passed from previous time step to the next time step.

![](../images/s2s-rnn.jpeg)

## Neural Network

- Neural network expects an numeric input, i.e., a numeric representation of the input text.
- So the first step in deep learning is the same as traditional ML, which is text vectorization.
- And because a sequence model eats in one word as the input, word representation/vectorization is crucial.


## Word Representations in Deep Learning

- Like in traditional machine learning, feature engineering is crucial to the success of the computational modeling.
- Of particular importance is the transformation of each text into numeric representation that has a significant portion of **textual semantics**.

- One-hot encoding is the most intuitive way to represent lexical words numerically.
- If the language vocabulary size is *V*, each word can be represented as a vector of size *V*, with its correpsonding dimension to be the value of **1** and the rest being **0**'s.

![](../images/seq-1hot.png)

- The main problem with is one-hot encoding is that the semantic distances in-between words are all the same, i.e., $D(mice,rats)=D(mice, horses)= 0$.

## Word Embeddings

- Now via neural network, we can learn **word embeddings** automatically. (See Word Embeddings notes).
- These word embeddings allows us to perform computation of lexical semantics.

![](../images/seq-embeddings.png)

import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

mice = np.array([0.2,0.0, 0.8, 0.1, 0.0])
rats = np.array([0.2,0.1,0.9, 0.3,0.0])
horses = np.array([0.0,0.0,0.1,0.9,0.8])
cosine_similarity([mice, rats, horses])

- While each dimension of the word embeddings may not have very transparent **semantic fields**, the results out of the semantic computation do reflect a lot the lexical semantic relations between words.


## From One-hot to Embeddings

- Therefore, we often add **Embedding Layer** as the first layer of a sequence model to render all lexical items in a text into semantically informative embeddings (vectors).
- Weights of the Embedding Layer can be trained along with the sequence model.
- Or alternatively, we can use pre-trained word embeddings, which were trained in a different task based on a much larger corpus.

![](../images/seq-1hot-to-embeddings.png)

- This is how an Embedding Layer works in `keras.layers.Embedding`:


![](../images/name-gender-classifier-dl/name-gender-classifier-dl.005.jpeg)

## Recurrent Neural Network (RNN) Language Model

- Words, as semantically represented by embeddings, can now be the input tensors for a RNN.

![](../images/s2s-rnn.jpeg)

- Moreover, we can create a RNN-based language model.
- A language model has two main objectives in its NLP applications:
    - To estimate the probability of a given sentence (or any other meaningful linguistic units)
    - To predict the upcoming word given limited linguistic inputs

![](../images/s2s-rnn-lm.jpeg)

- The RNN Langage model takes one word at each time step as the input and output the predicted next word.
- And the output at time step *i* becomes the input of the RNN at time step *i+1*.

- The loss function of the RNN Language Model is the distance between the predicted word, *y*, and the correct next-word (in its one-hot form).
- The training of the RNN language model is thus to minimize the sum of the **cross-entropy** at all time steps.

![](../images/s2s-rnn-lm-loss.jpeg)

- For example, if the target next word is *dog*, it's one-hot representation is `[0, 1, 0, 0 ,0, 0]`, and the RNN LM predicted *y* is `[0.2, 0.4, 0.1, 0.1, 0.2, 0.0]`, we can compute the **cross-entropy** at this time step as follows.

$$
E = - \sum_{k}t_k \log{y_k}
$$

    - k: refers to the dimensions of the one-hot vectors
    - t: refers to the target next word
    - y: refers to the predicted y from the RNN LM


def cross_entropy(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


t = [0, 1, 0, 0 ,0, 0]
y = [0.2, 0.4, 0.1, 0.1, 0.2, 0.0]

cross_entropy(np.array(y), np.array(t))

- And we compute the average cross-entropy error across all time steps for a particular sample (i.e., for the entire sequence).

$$
E = - \frac{1}{N}\sum_n\sum_{k}t_{nk} \log{y_{nk}}
$$

    - N: the number of words in the input text
- We can also compute the average cross-entropy error for samples of a batch size.
- We can also compute the average cross-entropy error for the entire training set.

## Back Propogation

- With the defined **loss function**, we can learn how good our current model is in the training process (i.e., the distance between the true target and the predicted label).
- In deep learning, we can use **back propogation** to find out:
    - how each parameter of the RNN LM is connected to the loss function
    - or, which parameter of the RNN LM contributes to the change of the loss function more
    - And therefore, we can **adjust** the parameters of the RNN LM accordingly.
    - The algorithm often used is called **gradient descent**

## Types of Gradient Descent

As we can compute the cross entropy in different ways, we can perform the gradient descent in different ways as well.

- **Batch Gradient Descent**: Update the model weights after we get the average cross entropy of all the sequences in the entire training set (as one epoch).

- **Stochastic Gradient Descent**(SGD): Update the model weights after we get the cross entropy of every sequence of the training set (across all time steps of course) (online).

- **Mini-batch Gradient Descent**: Update the model weights after we get the average cross entropy of a subset of the sequences in the training set. (Recommended!)



## From Vanilla RNN to LSTM and GRU

### Issues with Vanilla RNN

- An RNN on a sequence of words can be taken as a very deep neural network, where the depths of the network are the number of time steps of the sequence.
- In back propagation, for longer sequences, we would run into the **vanishing gradient problems**. To simply put, the gradients would become smaller and smaller as we back propagate further back to the previous contexts of the sequence.
- The further back the weights are, the more likely their gradients would approach zero. 


### Why is vanishing gradient an issue?

If the gradient becomes vanishingly small over longer distances:

- it is more difficult for RNN to learn the long-distance dependency relations between different time steps in the sequence.
- it is less likely that RNN would learn to preserve information over many timesteps.
- it is more likely that RNN would pay more attention to the effects of the recent time steps (i.e., biasing the RNN towards learning from **sequential recency**).


### LSTM (Long Short-Term Memory)

- A type of RNN proposed by [Hochreiter and Schmidhuber in 1997](https://www.bioinf.jku.at/publications/older/2604.pdf) as a solution to the vanishing gradients problem.
- For every time step, LSTM keeps track of two states: a **hidden** state and a **cell** state
    - Both are of the vector length $n$ same as the node/neuron number of the LSTM.
    - The **cell** state stores long-term information in the sequence.
    - The LSTM can erase, write and read information from the **cell**.
- The selection of which information is erased/written/read is controlled by three corresponding **gates**.
    - There are three **gates** in LSTM: **output**, **input**, and **forget** gates.
    - The gates are also of the vector length $n$ same as the node/neuron number of the LSTM.
    - On each time step, each element of the gates can be open(1),closed(0),or somewhere in-between.
    - The gates are dynamic: their value is computed based on the current **cell** state.

- Vanilla RNN

![](../images/chris-olah/LSTM3-SimpleRNN.png)
(Source: http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

- LSTM

![](../images/chris-olah/LSTM3-chain-annotated.jpeg)
(Source: http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

- LSTM gates:
    - **Input** gate controls how much of the input ($X_t$) is used in computing the new **cell** state ($C_t$)
    - **Output** gate determines how much of the new **cell** state ($C_t$) is used in the output **hidden** state 
    - **Forget** gate determines how much of the old **cell** state ($C_{t-1}$) is used in the new **cell** state ($C_t$).

![](../images/s2s-lstm.png)

### GRU (Gated Recurrent Unit)

- Proposed by [Cho et al. in 2014](https://arxiv.org/pdf/1406.1078v3.pdf) as a simpler alternative to the LSTM.
- On each time step, GRU keeps track of only the **hidden** state (no cell state).

- GRU

![](../images/chris-olah/LSTM3-var-GRU.png)
(Source: http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

![](../images/s2s-gru.png)

### Heuristics

- LSTM is a good default choice (especially if our data has particularly long dependencies, or we have lots of training data)
- We may switch to GRUs for **speed** and fewer parameters.

## Variations of Sequence Models

- There are four major variants of sequence models depending on the types of input and output of the RNN
    - Many to Many (Same lengths for input and output sequences)
        - Most of the tagging problems fall into this category.
    - Many to One
        - Most of the classification problems fall into this category.
    - One to Many
        - Image Captioning
    - Many to Many (Variable lengths for input and output sequences)
        - Machine Translation
        - Chatbot Q&A
        - Text Summarization

-----
![](../images/seq2seq-m2m-s.jpeg)

-----
![](../images/seq2seq-m21.jpeg)

-----
![](../images/seq2seq-12m.jpeg)

-----
![](../images/seq2seq-m2m-d.jpeg)

## References

- A must-read article on LSTM: Chris Olah's blog post on [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- Check Ch 5 and 6 in [Deep Learning 2｜用Python進行自然語言處理的基礎理論實作](https://www.books.com.tw/products/0010817138)
- These lecture notes are based on a talk presented by Ananth Sankar: [Sequence to Sequence Learning with Encoder-Decoder Neural Network Models](https://confengine.com/conferences/odsc-india-2019/proposal/10176/sequence-to-sequence-learning-with-encoder-decoder-neural-network-models). Some of the graphs used here are taken from Dr. Sankar's slides. His talk is highly recommended!
- [Sutskever et al. (2014). Sequence to sequence learning with neural networks.](https://papers.nips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf)
- New York Times: [The Great A.I. Awakening](https://www.nytimes.com/2016/12/14/magazine/the-great-ai-awakening.html)