#!/usr/bin/env python
# coding: utf-8

# # Machine Learning: A Simple Example

# ## A Quick Example: Name Gender Prediction

# Let's assume that we have collected a list of personal names and we have their corresponding gender labels, i.e., whether the name is a male or female one.
# 
# The goal of this example is to create a classifier that would automatically classify a given name into either male or female.

# ### Prepare Data

# - We use the data provided in NLTK. Please download the corpus data if necessary.
# - We load the corpus, `nltk.corpus.names` and randomize it before we proceed.

# In[1]:


import numpy as np
import nltk
from nltk.corpus import names
import random


# In[2]:


labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
                 [(name, 'female') for name in names.words('female.txt')])
random.shuffle(labeled_names)


# ### Feature Engineering

# - Now our unit for classification is a name. 
# - In **feature engineering**, our goal is to transform the texts (i.e., names) into vectorized representations.
# - To start with, let's represent each text (name) by using its last character as the features.

# In[3]:


def text_vectorizer(word):
    return {'last_letter': word[-1]}


text_vectorizer('Shrek')


# ### Train-Test Split

# - We then apply the feature engineering method to every text in the data and split the data into **training** and **testing** sets.

# In[4]:


featuresets = [(text_vectorizer(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets[500:], featuresets[:500]


# ### Train the Model

# - A good start is to try the simple Naive Bayes Classifier.

# In[5]:


classifier = nltk.NaiveBayesClassifier.train(train_set)


# ### Model Prediction

# In[6]:


print(classifier.classify(text_vectorizer('Neo')))
print(classifier.classify(text_vectorizer('Trinity')))
print(classifier.classify(text_vectorizer('Alvin')))


# In[7]:


print(nltk.classify.accuracy(classifier, test_set))


# ### Post-hoc Analysis

# - One of the most important steps after model training is to examine which features contribute the most to the classifier prediction of the class.

# In[8]:


classifier.show_most_informative_features(5)


# - Please note that in `NLTK`, we can use the `apply_features` to create training and testing datasets.
# - When you have a very large feature set, this can be more effective in terms of memory management.

# - This is our earlier method of creating training and testing sets:
# 
# ```
# featuresets = [(text_vectorizer(n), gender) for (n, gender) in labeled_names]
# train_set, test_set = featuresets[500:], featuresets[:500]
# ```

# In[9]:


from nltk.classify import apply_features
train_set = apply_features(text_vectorizer, labeled_names[500:])
test_set = apply_features(text_vectorizer, labeled_names[:500])


# ## How can we improve the model/classifier?

# In the following, we will talk about methods that we may consider to further improve the model training.
# 
# - Feature Engineering
# - Error Analysis
# - Cross Validation
# - Try Different Machine-Learning Algorithms
# - (Ensemble Methods)

# ## More Sophisticated Feature Engineering

# - We can extract more features from the names.
# - Use the following features for vectorized representations of names:
#     - The first/last letter
#     - Frequencies of all 26 alphabets in the names

# In[10]:


def text_vectorizer2(name):
    features = {}
    features["first_letter"] = name[0].lower()
    features["last_letter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count({})".format(letter)] = name.lower().count(letter)
        features["has({})".format(letter)] = (letter in name.lower())
    return features


text_vectorizer2('Alvin')


# In[11]:


text_vectorizer2('John')


# In[12]:


train_set = apply_features(text_vectorizer2, labeled_names[500:])
test_set = apply_features(text_vectorizer2, labeled_names[:500])
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))


# In[13]:


classifier.show_most_informative_features(n=20)


# ## Train-Development-Test Data Splits for Error Analysis

# - Normally we have **training**-**testing** splits of data
# - Sometimes we can use **development (dev)** set for error analysis and feature engineering.
# - This dev set should be independent of training and testing sets.

# - Now let's train the model on the **training set** and first check the classifier's performance on the **dev** set.
# - We then identify the errors the classifier made in the **dev** set.
# - We perform error analysis for further improvement.
# - We only test our **final model** on the testing set. (Note: Testing set can only be used **once**.)

# In[14]:


train_names = labeled_names[1500:]
devtest_names = labeled_names[500:1500]
test_names = labeled_names[:500]

train_set = [(text_vectorizer2(n), gender) for (n, gender) in train_names]
devtest_set = [(text_vectorizer2(n), gender) for (n, gender) in devtest_names]
test_set = [(text_vectorizer2(n), gender) for (n, gender) in test_names]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, devtest_set))


# In[15]:


errors = []
for (name, tag) in devtest_names:
    guess = classifier.classify(text_vectorizer2(name))
    if guess != tag:
        errors.append((tag, guess, name))


# In[16]:


import csv

with open('error-analysis.csv', 'w') as f:

    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(['tag', 'guess', 'name'])
    write.writerows(errors)


# - Ideally, we can inspect the errors in a spreadsheet and come up with better rules (features) that could help improve the classifier.

# In[17]:


import pandas as pd
## check first and last N rows
pd.read_csv('error-analysis.csv').iloc[[*range(10), *range(-10, 0)],]


# ## Evaluation

# ![](../images/confusion-matrix.jpeg)

# - Confusion Matrix:
#     - **True positives** are relevant items that we correctly identified as relevant.
#     - **True negatives** are irrelevant items that we correctly identified as irrelevant.
#     - **False positives** (or Type I errors) are irrelevant items that we incorrectly identified as relevant.
#     - **False negatives** (or Type II errors) are relevant items that we incorrectly identified as irrelevant.
#     

# Given these four numbers, we can define the following model evaluation metrics:
# - **Accuracy**: How many items were correctly classified, i.e., $\frac{TP + TN}{N}$
# - **Precision**: How many of the items identified by the classifier as relevant are indeed relevant, i.e., $\frac{TP}{TP+FP}$.
# - **Recall**: How many of the true relevant items were successfully identified by the classifier, i.e., $\frac{TP}{TP+FN}$.
# - **F-Measure (or F-Score)**: the harmonic mean of the precision and recall,i.e.:
#     
# 
# $$ 
# F= \frac{(2 × Precision × Recall)}{(Precision + Recall)} 
# $$

# :::{note}
# 
# When dealing with imbalanced class distributions, we need to take into account the baseline performance in our model evaluation. For example. if the distribution of `Class 0` and `Class 1` is 9:1, then a naive classifier might as well classify all cases as `Class 0`, yielding a high-**precision** performance (i.e., Precision = 90%).
# 
# Given this baseline, to better evaluate the classifier on imbalanced dataset, probably the classifier's **recall rates** are more important.
# 
# :::

# In[18]:


print('Accuracy: {:4.2f}'.format(nltk.classify.accuracy(classifier, test_set)))


# In[19]:


## Compute the Confusion Matrix
t_f = [feature for (feature, label) in test_set]  # features of test set
t_l = [label for (feature, label) in test_set]  # labels of test set
t_l_pr = [classifier.classify(f) for f in t_f]  # predicted labels of test set
cm = nltk.ConfusionMatrix(t_l, t_l_pr)


# In[20]:


cm = nltk.ConfusionMatrix(t_l, t_l_pr)
print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))


# In[21]:


def createCM(classifier, test_set):
    t_f = [feature for (feature, label) in test_set]
    t_l = [label for (feature, label) in test_set]
    t_l_pr = [classifier.classify(f) for f in t_f]
    cm = nltk.ConfusionMatrix(t_l, t_l_pr)
    print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))


# In[22]:


createCM(classifier, test_set)


# ## Cross Validation

# - We can also check our average model performance using the cross-validation method.

# ---
# 
# ![](../images/ml-kfold.png)
# (Source: https://scikit-learn.org/stable/modules/cross_validation.html)
# 
# ---

# In[23]:


import sklearn.model_selection
kf = sklearn.model_selection.KFold(n_splits=10)
acc_kf = []  ## accuracy holder

## Cross-validation
for train_index, test_index in kf.split(train_set):
    #print("TRAIN:", train_index, "TEST:", test_index)
    classifier = nltk.NaiveBayesClassifier.train(
        train_set[train_index[0]:train_index[len(train_index) - 1]])
    cur_fold_acc = nltk.classify.util.accuracy(
        classifier, train_set[test_index[0]:test_index[len(test_index) - 1]])
    acc_kf.append(cur_fold_acc)
    print('accuracy:', np.round(cur_fold_acc, 2))


# In[24]:


np.mean(acc_kf)


# ## Try Different Machine Learning Algorithms

# - There are many ML algorithms for classification tasks.
# - Here we will demonstrate a few more classifiers implemented in NLTK, including:
#     - Maximum Entropy Classifier (Logistic Regression)
#     - Decision Tree Classifier
# - Also, in NLTK, we can use the classification methods provided in `sklearn` as well, including:
#     - Naive Bayes
#     - Logistic Regression
#     - Support Vector Machine

# - When we try another ML algorithm, we do the following:
#     - train the model
#     - check model performance (accuracy and confusion matrix)
#     - check the most informative features
#     - obtain average performance using *k*-fold cross validation

# ### Try Maxent Classifier

# - Maxent is memory hungry, slower, and it requires `numpy`.
# 

# In[25]:


get_ipython().run_cell_magic('time', '', "from nltk.classify import MaxentClassifier\nclassifier_maxent = MaxentClassifier.train(train_set,\n                                           algorithm='iis',\n                                           trace=0,\n                                           max_iter=10000,\n                                           min_lldelta=0.001)")


# ```{note}
# The default algorithm for training is `iis` (Improved Iterative Scaling). Another alternative is `gis` (General Iterative Scaling), which is faster.
# ```

# In[26]:


nltk.classify.accuracy(classifier_maxent, test_set)


# In[27]:


classifier_maxent.show_most_informative_features(n=20)


# In[28]:


createCM(classifier_maxent, test_set)


# In[29]:


get_ipython().run_cell_magic('time', '', 'for train_index, test_index in kf.split(train_set):\n    #print("TRAIN:", train_index, "TEST:", test_index)\n    classifier = MaxentClassifier.train(\n        train_set[train_index[0]:train_index[len(train_index) - 1]],\n        algorithm=\'gis\',\n        trace=0,\n        max_iter=100,\n        min_lldelta=0.01) ## set smaller value for `min_lldelta`\n    print(\n        \'accuracy:\',\n        nltk.classify.util.accuracy(\n            classifier,\n            train_set[test_index[0]:test_index[len(test_index) - 1]]))')


# ### Try Decision Tree

# - Parameters:
#     - `binary`: whether the features are binary
#     - `entropy_cutoff`: a value used during tree refinement process
#         - entropy = 1 -> high-level uncertainty
#         - entropy = 0 -> perfect model prediction
#     - `depth_cutoff`: to control the depth of the tree
#     - `support_cutoff`: the minimum number of instances that are required to make a decision about a feature.

# In[30]:


get_ipython().run_cell_magic('time', '', 'from nltk.classify import DecisionTreeClassifier\nclassifier_dt = DecisionTreeClassifier.train(train_set,\n                                             binary=True,\n                                             entropy_cutoff=0.7,\n                                             depth_cutoff=5,\n                                             support_cutoff=5)')


# In[31]:


nltk.classify.accuracy(classifier_dt, test_set)


# In[32]:


createCM(classifier_dt, test_set)


# In[33]:


get_ipython().run_cell_magic('time', '', '\nfor train_index, test_index in kf.split(train_set):\n    #print("TRAIN:", train_index, "TEST:", test_index)\n    classifier = DecisionTreeClassifier.train(\n        train_set[train_index[0]:train_index[len(train_index) - 1]],\n        binary=True,\n        entropy_cutoff=0.7,\n        depth_cutoff=5,\n        support_cutoff=5)\n    print(\n        \'accuracy:\',\n        nltk.classify.util.accuracy(\n            classifier,\n            train_set[test_index[0]:test_index[len(test_index) - 1]]))')


# ### Try `sklearn` Classifiers

# - `sklearn` is a very useful module for machine learning. We will talk more about this module in our later lectures.
# - This package provides a lot more ML algorithms for classification tasks.

# #### Naive Bayes in `sklearn`

# In[34]:


from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB

sk_classifier = SklearnClassifier(MultinomialNB())
sk_classifier.train(train_set)


# In[35]:


nltk.classify.accuracy(sk_classifier, test_set)


# #### Logistic Regression in `sklearn`

# In[36]:


from sklearn.linear_model import LogisticRegression
sk_classifier = SklearnClassifier(LogisticRegression(max_iter=500))
sk_classifier.train(train_set)
nltk.classify.accuracy(sk_classifier, test_set)


# #### Support Vector Machine in `sklearn`

# - `sklearn` provides several implementations for Support Vector Machines.
# - Please see its documentation for more detail: [Support Vector Machine](https://scikit-learn.org/stable/modules/svm.html)

# In[37]:


from sklearn.svm import SVC
sk_classifier = SklearnClassifier(SVC())
sk_classifier.train(train_set)
nltk.classify.accuracy(sk_classifier, test_set)


# In[38]:


from sklearn.svm import LinearSVC
sk_classifier = SklearnClassifier(LinearSVC(max_iter=2000))
sk_classifier.train(train_set)
nltk.classify.accuracy(sk_classifier, test_set)


# In[39]:


from sklearn.svm import NuSVC
sk_classifier = SklearnClassifier(NuSVC())
sk_classifier.train(train_set)
nltk.classify.accuracy(sk_classifier, test_set)


# ## Remaining Issues

# - Feature engineering is crucial to the process of machine learning.
# - The quality of the text vectorization almost determines the classifier's performance to a great deal.
# - Every ML algorithm requires a lot of **hyperparameter** settings, which can have substantial impact on the model performances.
# - We need a more **systematic** way to find the optimal combinations of hyperparameters for a given ML algorithm.
# - We will come back to issue when we talk about doing ML with `sklearn`.

# ## References
# 
# - NLTK Book, [Chapter 6 Learning to Classify Texts](https://www.nltk.org/book/ch06.html)
# - Géron (2019), Chapter 3 Classification
