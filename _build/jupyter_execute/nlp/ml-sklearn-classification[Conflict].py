# Sentiment Analysis Using Bag-of-Words

- Sentiment analysis is to analyze the textual documents and extract information that is related to the author's sentiment or opinion. It is sometimes referred to as "opinion mining".
- It is popular and widely used in industry, e.g., corporate surveys, feedback surveys, social media data, reviews for movies, places, hotels, commodities, etc..
- The sentiment information from texts can be crucial to further decision making in the industry.

- Output of Sentiment Analysis
    - Qualitative: overall sentiment scale (positive/negative)
    - Quantitative: sentiment polarity scores 

- In this tutorial, we will use movie reviews as an example.
- And the classification task is simple: to classify the document into one of the two classes, i.e., positive or negative.
- Most importantly, we will be using mainly the module `sklearn` for creating and training the classifier.

import nltk, random
from nltk.corpus import movie_reviews
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

## Data Loading

- We will use the corpus `nlkt.corpus.movie_reviews` as our data.
- Please download the corpus if you haven't.

print(len(movie_reviews.fileids()))
print(movie_reviews.categories())
print(movie_reviews.words()[:100])
print(movie_reviews.fileids()[:10])

- Rearrange the corpus data as a list of tuple, where the first element is the word tokens of the documents, and the second element is the label of the documents (i.e., sentiment labels).

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

- Important descriptive statistics:
    - Corpus Size (Number of Documents)
    - Corpus Size (Number of Words)
    - Distribution of the Two Classes
- It is important to always report the above necessary descriptive statistics when presenting your classifiers.

print('Number of Reviews/Documents: {}'.format(len(documents)))
print('Corpus Size (words): {}'.format(np.sum([len(d) for (d,l) in documents])))
print('Sample Text of Doc 1:')
print('-'*30)
print(' '.join(documents[0][0][:50])) # first 50 words of the first document

## Check Sentiment Distribution of the Current Dataset
from collections import Counter
sentiment_distr = Counter([label for (words, label) in documents])
print(sentiment_distr)

## Train-Test Split

- We split the entire dataset into two parts: training set and testing set.
- The proportion of training and testing sets may depend on the corpus size.
- In the train-test split, make sure the the distribution of the classe is proportional.

from sklearn.model_selection import train_test_split
train, test = train_test_split(documents, test_size = 0.33, random_state=42)

## Sentiment Distrubtion for Train and Test
print(Counter([label for (words, label) in train]))
print(Counter([label for (words, label) in test]))

- Because in most of the ML steps, the feature sets and the labels are often separated as two units, we split our training data into `X_train` and `y_train` as the features and labels in training.
- Likewise, we split our testing data into `X_test` and `y_test` as the features and labels in testing.

X_train = [' '.join(words) for (words, label) in train]
X_test = [' '.join(words) for (words, label) in test]
y_train = [label for (words, label) in train]
y_test = [label for (words, label) in test]

## Text Vectorization

- In feature-based machine learning, we need to vectorize texts into feature sets (i.e., feature engineering on texts).
- We use the naive bag-of-words text vectorization. In particular, we use the weighted version of BOW.

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

tfidf_vec = TfidfVectorizer(min_df = 10, token_pattern = r'[a-zA-Z]+')
X_train_bow = tfidf_vec.fit_transform(X_train) # fit train
X_test_bow = tfidf_vec.transform(X_test) # transform test

- Important Notes:
    - Always split the data into train and test first before vectorizing the texts
    - Otherwise, you would leak information to the training process, which may lead to over-fitting
    - When vectorizing the texts, `fit_transform()` on the training set and `transform()` on the testing set.
    - Always report the number of features used in the model.

print(X_train_bow.shape)
print(X_test_bow.shape)

## Models

- For our current binary sentiment classifier, we will try a few common classification algorithms:
    - Support Vector Machine
    - Decision Tree
    - Naive Bayes
    - Logistic Regression
    

- The common steps include:
    - We fit the model with our training data.
    - We use the fitted model to make prediction.
    - We evaluate the model prediction by comparing the predicted classes and the true labels.

### SVM

from sklearn import svm

model_svm = svm.SVC(C=8.0, kernel='linear')
model_svm.fit(X_train_bow, y_train)

model_svm.predict(X_test_bow[:10])
#print(model_svm.score(test_text_bow, test_label))

### Decision Tree

from sklearn.tree import DecisionTreeClassifier

model_dec = DecisionTreeClassifier(max_depth=20, random_state=0)
model_dec.fit(X_train_bow, y_train)

model_dec.predict(X_test_bow[:10])

### Naive Bayes

from sklearn.naive_bayes import GaussianNB
model_gnb = GaussianNB()
model_gnb.fit(X_train_bow.toarray(), y_train)

model_gnb.predict(X_test_bow[:10].toarray())

### Logistic Regression

from sklearn.linear_model import LogisticRegression

model_lg = LogisticRegression()
model_lg.fit(X_train_bow, y_train)

model_lg.predict(X_test_bow[:10].toarray())

## Evaluation

- To evaluate each model's performance, there are several common metrics in use:
    - precision
    - recall
    - F-score
    - Accuracy
    - Confusion Matrix

#Mean Accuracy
print(model_svm.score(X_test_bow, y_test))
print(model_dec.score(X_test_bow, y_test))
print(model_gnb.score(X_test_bow.toarray(), y_test))
print(model_lg.score(X_test_bow, y_test))

# F1
from sklearn.metrics import f1_score

f1_score(y_test, model_svm.predict(X_test_bow), 
         average=None, 
         labels = movie_reviews.categories())

from sklearn.metrics import confusion_matrix, plot_confusion_matrix

plot_confusion_matrix(model_svm, X_test_bow, y_test, normalize=None)

plot_confusion_matrix(model_lg, X_test_bow.toarray(), y_test, normalize=None)

## try a whole new self-created review:)
new_review =['This book looks soso like the content but the cover is weird',
             'This book looks soso like the content and the cover is weird'
            ]
new_review_bow = tfidf_vec.transform(new_review)
model_svm.predict(new_review_bow)

## Tuning Model Hyperparameters - Grid Search

- For each model, we have not optimized it in terms of its hyperparameter setting.
- Now that SVM seems to perform the best among all, we take this as our base model and further fine-tune its hyperparameter using **cross-validation** and **Grid Search**.

%%time

from sklearn.model_selection import GridSearchCV

parameters = {'kernel': ('linear', 'rbf'), 'C': (1,4,8,16,32)}

svc = svm.SVC()
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(X_train_bow, y_train)

- We can check the parameters that yield the most optimal results in the Grid Search:

#print(sorted(clf.cv_results_.keys()))
print(clf.best_params_)

print(clf.score(X_test_bow, y_test))

## Post-hoc Analysis

- After we find the optimal classifier, next comes the most important step: interpret the classifier.
- A good classifier may not always perform as we have expected. Chances are that the classifier may have used cues that are unexpected for decision marking.
- In the post-hoc analysis, we are interested in:
    - Which features contribute to the classifier's prediction the most?
    - Which features are more relevant to each class prediction?

- We will introduce three methods for post-hoc analysis:
    - LIME
    - Model coefficients and feature importance
    - Permutation importances

### LIME

- Using LIME (Local Interpretable Model-agnostic Explainations) to interpret the importance of the features in relatio to the model prediction.
- LIME was introduced in 2016 by Marco Ribeiro and his collaborators in a paper called [“Why Should I Trust You?” Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938) 
- Its objective is to explain a model prediction for a specific text sample in a human-interpretable way.
- What we have done so far tells us that the model accuracy is good, but **we have no idea whether the classifier has learned features that are useful and meaningful**.
- Identify important words that may have great contribution to the model prediction

import lime
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline, TransformerMixin, Pipeline

## Refit model based on optimal parameter settings
pipeline = Pipeline([
  ('vectorizer',tfidf_vec), 
  ('clf', svm.SVC(C=4, kernel='rbf', probability=True))])
pipeline.fit(X_train, y_train)

import textwrap
reviews_test = X_test
sentiments_test = y_test



# We choose a sample from test set
idx = 210
text_sample = reviews_test[idx]
class_names = ['negative', 'positive']

print('Review ID-{}:'.format(idx))
print('-'*50)
print('Review Text:\n', textwrap.fill(text_sample,400))
print('-'*50)
print('Probability(positive) =', pipeline.predict_proba([text_sample])[0,1])
print('Probability(negative) =', pipeline.predict_proba([text_sample])[0,0])
print('Predicted class: %s' % pipeline.predict([text_sample]))
print('True class: %s' % sentiments_test[idx])

import matplotlib
matplotlib.rcParams['figure.dpi']=300
%matplotlib inline


explainer = LimeTextExplainer(class_names=class_names)
explanation = explainer.explain_instance(text_sample, 
                                         pipeline.predict_proba, 
                                         num_features=20)
explanation.show_in_notebook(text=True)

### Model Coefficients and Feature Importance

importances = pipeline.named_steps['clf'].coef_.toarray().flatten()
top_indices_pos = np.argsort(importances)[::-1][:10] 
top_indices_neg = np.argsort(importances)[:10]
feature_names = np.array(tfidf_vec.get_feature_names()) # List indexing is different from array

feature_importance_df = pd.DataFrame({'FEATURE': feature_names[np.concatenate((top_indices_pos, top_indices_neg))],
                                     'IMPORTANCE': importances[np.concatenate((top_indices_pos, top_indices_neg))],
                                     'SENTIMENT': ['pos' for _ in range(len(top_indices_pos))]+['neg' for _ in range(len(top_indices_neg))]})
feature_importance_df

plt.figure(figsize=(7,5), dpi=100)
plt.style.use('fivethirtyeight')
#print(plt.style.available)
plt.bar(x = feature_importance_df['FEATURE'], height=feature_importance_df['IMPORTANCE'])
plt.xlabel("Most Important Features")
plt.ylabel("Feature Importances")
plt.xticks(rotation=90)
plt.title("Feature Importance: Top Words", color="black")

%load_ext rpy2.ipython

%%R -i feature_importance_df -w 6 -h 6 --units in -r 150
library(ggplot2)
library(dplyr)

head(feature_importance_df)

feature_importance_df %>%
    ggplot(aes(reorder(FEATURE,IMPORTANCE), IMPORTANCE, fill=IMPORTANCE, color=SENTIMENT)) +
     geom_bar(stat="identity") +
     coord_flip() +
     scale_fill_gradient2(guide=FALSE) +
     labs(x='FEATURE', x="IMPORTANCE", title="Most Important Features")

### Permutation Importances

- Permutation feature importance is a model inspection technique that can be applied to any fitted classifier using feature-based ML.
- Permutation importance is defined as the decrease in a model score when a single feature value is randomly shuffled.
    - The drop in the model score is indicative of how much the model depends on the feature.
    - The increase in the model score is indicative of how redundant the feature is.
- Permutation importance reports the importance of the feature as the difference between target model accuracy - shuffled model accuracy.
    - Positive permutation importance score, Higher importance.
    - Negative permutation importance score, lower importance.
- See `sklearn` documentation: [4.2 Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-importance)

:::{tip}
On which dataset (training, held-out, or testing sets) should be perform the permutation importance computation?

It is suggested to use a held-out set, which makes it possible to highlight which features contribute the most to the generalization power of the classifier (i.e., to avoid overfitting problems).
:::

%%time
from sklearn.inspection import permutation_importance
r = permutation_importance(model_lg, X_test_bow.toarray(), y_test,
                           n_repeats=5,
                           random_state=0)

for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{feature_names[i]:<8}"
              f"{r.importances_mean[i]:.3f}"
              f" +/- {r.importances_std[i]:.3f}")

### Other Variants

- Some of the online code snippets try to implement the `show_most_informative_features()` in `nltk` classifier.
- Here the codes only work with linear classifiers (e.g., Logistic models) in sklearn.
- Need more updates. See this [SO post](https://stackoverflow.com/questions/11116697/how-to-get-most-informative-features-for-scikit-learn-classifiers)

def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))

show_most_informative_features(tfidf_vec, model_lg, n=20)

- For tree-based classifiers, visualization is better.

import sklearn
from sklearn.tree import plot_tree
text_representation = sklearn.tree.export_text(model_dec, feature_names = tfidf_vec.get_feature_names())
print(text_representation)




plt.style.use('classic')
from matplotlib import pyplot as plt

fig = plt.figure(figsize=(80,50))
_ = sklearn.tree.plot_tree(model_dec, max_depth=3,
                   feature_names=tfidf_vec.get_feature_names(),  
                   class_names=model_dec.classes_,
                   filled=True, fontsize=36)

#fig.savefig("decistion_tree.png")

## Saving Model


#  import pickle

# with open('../ml-sent-svm.pkl', 'wb') as f:
#     pickle.dump(clf, f)
# with open('../ml-sent-svm.pkl' 'rb') as f:
#     loaded_svm = pickle.load(f)

## Dictionary-based Sentiment Classifier

- Without machine learning, we can still build a sentiment classifier using a dictionary-based approach.
- Words can be manually annotated with sentiment polarity scores.
- Based on the sentiment dictionary, we can then compute the sentiment scores for a text.

### TextBlob Lexicon

from textblob import TextBlob

doc_sent = TextBlob(X_train[0])
print(doc_sent.sentiment)
print(doc_sent.sentiment.polarity)

doc_sents = [TextBlob(doc).sentiment.polarity for doc in X_train]

y_train[:10]

doc_sents_prediction = ['pos' if score >= 0.1 else 'neg' for score in doc_sents]
doc_sents_prediction[:10]

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
print(accuracy_score(y_train, doc_sents_prediction))
print(f1_score(y_train, doc_sents_prediction, average=None, labels=['neg','pos']))

ConfusionMatrixDisplay(confusion_matrix(y_train, doc_sents_prediction, normalize="all")).plot()

### AFINN Lexicon

from afinn import Afinn

afn = Afinn(emoticons=True)

afn.score("This movie, not good. worth it :( But can give it a try!! worth it")

doc_sents_afn = [afn.score(d) for d in X_train]

doc_sents_afn_prediction = ['pos' if score >= 1.0 else 'neg' for score in doc_sents_afn]

print(accuracy_score(y_train, doc_sents_afn_prediction))
print(f1_score(y_train, doc_sents_afn_prediction, average=None, labels=['neg','pos']))

ConfusionMatrixDisplay(confusion_matrix(y_train, doc_sents_afn_prediction, normalize="all")).plot()

- Disadvantage of Dictionary-based Approach
    - Constructing the sentiment dictionary is time-consuming.
    - The sentiment dictionary can be topic-dependent.


- English Resources for Sentiment Analysis
    - See Sarkar (2019) Chapter 9 Senitment Analysis (on Unsupervised Lexicon-based Models)
    - [`pattern`](https://github.com/clips/pattern)
    - [`textblob`](https://github.com/sloria/TextBlob)
    - [`affin`](https://github.com/fnielsen/afinn)
- Chinese Resources for Sentiment Analysis
    - [`polyglot`](https://polyglot.readthedocs.io/en/latest/Sentiment.html)
    - [`snownlp`](https://github.com/isnowfy/snownlp) (for simplified Chinese)
    - [ANTUSD (The Augmented NTU Sentiment Dictionary)](http://academiasinicanlplab.github.io/#resources)

## Package Requirement

import pkg_resources
import types
def get_imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            # Split ensures you get root package, 
            # not just imported function
            name = val.__name__.split(".")[0]

        elif isinstance(val, type):
            name = val.__module__.split(".")[0]

        # Some packages are weird and have different
        # imported names vs. system/pip names. Unfortunately,
        # there is no systematic way to get pip names from
        # a package's imported name. You'll have to add
        # exceptions to this list manually!
        poorly_named_packages = {
            "PIL": "Pillow",
            "sklearn": "scikit-learn"
        }
        if name in poorly_named_packages.keys():
            name = poorly_named_packages[name]

        yield name

imports = list(set(get_imports()))

# The only way I found to get the version of the root package
# from only the name of the package is to cross-check the names 
# of installed packages vs. imported packages
requirements = []
for m in pkg_resources.working_set:
    if m.project_name in imports and m.project_name!="pip":
        requirements.append((m.project_name, m.version))

for r in requirements:
    print("{}=={}".format(*r))

## References

- Geron (2019), Ch 2 and 3
- Sarkar (2019), Ch 9
- See [Keith Galli's sklearn tutorial](https://github.com/KeithGalli/sklearn)
- See a blog post, [LIME of words: Interpreting RNN Predictions](https://data4thought.com/deep-lime.html).
- See a blog post, [LIME of words: how to interpret your machine learning model predictions](https://data4thought.com/lime-of-words.html)