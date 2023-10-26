#!/usr/bin/env python
# coding: utf-8

# # Assignment XII: Encoder-Decoder Sequence Models

# ## Question 1
# 
# Please download the dataset, `demo_data/date-student-version.csv`, which includes a two-column data frame. The first column, `INPUT`, includes dates representations in variable ways, and the second column, `OUTPUT`, includes their uniform representations.
# 
# Please create a Sequence Model using Encoder-Decoder architecture as shown in the Number Addition examples in our lecture, whose objective is to perform date conversion, i.e., to convert the dates in variable formats (INPUT) to dates in a consistent format (OUTPUT).
# 
# In particular, please compare the effectiveness of different network architectures, including:
# 
# - Simple RNN based Model
# - GRU/LSRM based Moel
# - Bi-directional Sequence Model
# - Peeky Sequence Model
# - Attention-based Sequence Model
#  
# In your report, please present:
# 
# - (a) the training histories of each model in one graph for comparison of their respective effectiveness.
# - (b) translations of a few sequences for quick model evaluation
# - (c) the attention plot from the attention-based model on one random input sequence

# In[58]:


for seq_index in range(20):
    # Take one sequence (part of the training set)
    # for trying out decoding.

    decoded_sentence, _ = decode_sequence(
        encoder_input_onehot[seq_index:seq_index + 1, :, :])
    print('-')
    print('Input sentence:', tr_input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)


# ![](../images/exercise/enc-dec-1.png)
# ![](../images/exercise/enc-dec-2.png)
# ![](../images/exercise/enc-dec-3.png)
