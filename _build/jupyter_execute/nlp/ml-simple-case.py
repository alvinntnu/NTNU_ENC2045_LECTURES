# Machine Learning: A Simple Example

Let's assume that we have collected a list of personal names and we have their corresponding gender labels, i.e., whether the name is a male or female one.

The goal of this example is to create a classifier that would automatically classify a given name into either male or female.

## A Quick Example: Name Gender

import nltk

from nltk.corpus import names
import random

labeled_names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
random.shuffle(labeled_names)

def gender_features(word):
    return {'last_letter': word[-1]}
gender_features('Shrek')

featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets[500:], featuresets[:500]

classifier = nltk.NaiveBayesClassifier.train(train_set)

classifier.classify(gender_features('Neo'))
classifier.classify(gender_features('Trinity'))

print(nltk.classify.accuracy(classifier, test_set))

classifier.show_most_informative_features(5)

from nltk.classify import apply_features
train_set = apply_features(gender_features, labeled_names[500:])
test_set = apply_features(gender_features, labeled_names[:500])


## Features and Training

def gender_features2(name):
    features = {}
    features["first_letter"] = name[0].lower()
    features["last_letter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count({})".format(letter)] = name.lower().count(letter)
        features["has({})".format(letter)] = (letter in name.lower())
    return features

gender_features2('John') 

train_set = apply_features(gender_features2, labeled_names[500:])
test_set = apply_features(gender_features2, labeled_names[:500])
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))

classifier.show_most_informative_features(n = 20)

## Train-Development-Test Data Splits for Error Analysis

train_names = labeled_names[1500:]
devtest_names = labeled_names[500:1500]
test_names = labeled_names[:500]

train_set = [(gender_features2(n), gender) for (n, gender) in train_names]
devtest_set = [(gender_features2(n), gender) for (n, gender) in devtest_names]
test_set = [(gender_features(n), gender) for (n, gender) in test_names]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, devtest_set))


errors = []
for (name, tag) in devtest_names:
    guess = classifier.classify(gender_features(name))
    if guess != tag:
        errors.append((tag, guess, name))

import csv

with open('error-analysis.csv', 'w') as f: 
      
    # using csv.writer method from CSV package 
    write = csv.writer(f) 
    write.writerow(['tag','guess','name']) 
    write.writerows(errors) 

## Evaluation

![](../images/confusion-matrix.png)

- Confusion Matrix:
    - **True positives** are relevant items that we correctly identified as relevant.
    - **True negatives** are irrelevant items that we correctly identified as irrelevant.
    - **False positives** (or Type I errors) are irrelevant items that we incorrectly identified as relevant.
    - **False negatives** (or Type II errors) are relevant items that we incorrectly identified as irrelevant.
    Given these four numbers, we can define the following metrics:

- Evaluation Metrics:
    - **Precision**: how many of the items that we identified were relevant, is TP/(TP+FP).
    - **Recall**: how many of the relevant items that we identified, is TP/(TP+FN).
    - **F-Measure (or F-Score)**: the harmonic mean of the precision and recall,i.e.:
    

$$ 
\frac{(2 × Precision × Recall)}{(Precision + Recall)} 
$$


print('Accuracy: {:4.2f}'.format(nltk.classify.accuracy(classifier, test_set))) 

t_f = [feature for (feature, label) in test_set] # features of test set
t_l = [label for (feature, label) in test_set] # labels of test set
t_l_pr = [classifier.classify(f) for f in t_f] # predicted labels of test set
cm = nltk.ConfusionMatrix(t_l, t_l_pr)

cm=nltk.ConfusionMatrix(t_l, t_l_pr)
print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))

def createCM(classifier, test_set):
    t_f = [feature for (feature, label) in test_set]
    t_l = [label for (feature, label) in test_set]
    t_l_pr = [classifier.classify(f) for f in t_f]
    cm = nltk.ConfusionMatrix(t_l, t_l_pr)
    print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))

createCM(classifier, test_set)

## Try Maxent Classifier

- Maxent is memory hungry, slower, and it requires `numpy`.


%%time
from nltk.classify import MaxentClassifier
classifier_maxent = MaxentClassifier.train(train_set, algorithm = 'gis', trace = 0, max_iter=10, min_lldelta=0.5)

```{note}
The default algorithm for training is `iis` (Improved Iterative Scaling). Another alternative is `gis` (General Iterative Scaling), which is faster.
```

nltk.classify.accuracy(classifier_maxent, test_set)

classifier_maxent.show_most_informative_features(n=20)

createCM(classifier_maxent, test_set)

## Try Decision Tree

- Parameters:
    - `binary`: whether the features are binary
    - `entropy_cutoff`: a value used during tree refinement process (entropy=1 -> high-level uncertainty; entropy = 0 -> perfect model prediction)
    - `depth_cutoff`: to control the depth of the tree
    - `support_cutoff`: the mimimum number of instances that are required to make a decision about a feature.

%%time
from nltk.classify import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier.train(train_set, binary=True, 
                                             entropy_cutoff=0.8, depth_cutoff=5, support_cutoff =5)

nltk.classify.accuracy(classifier_dt, test_set)

createCM(classifier_dt, test_set)

## Reference

- NLTK Book, Chapter 6.