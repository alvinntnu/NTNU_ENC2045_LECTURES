# Midterm Exam

## Instructions

- The midterm consists of **THREE** main questions. For each question, you need to accomplish one ultimate goal by using the datasets from `demo_data/midterm`. To help you reach the goal, each question is further divided into two sub-questions for you to accomplish the goal step by step.
- If you fail to produce the output from the previous question and usually its output is the input of the subsequent question, you can use the sample output files provided in `demo_data/midterm` to start the second question. For example, you can use the sample output of Question 1-1 for the task of Question 1-2.
- Same as previous assignments, please submit both your **notebook** (`*.ipynb`) and the **HTML** output (`*.html`).
- Deadline for Midterm Submission: **Noon, Sunday, April 18** via Moodle.

:::{tip}

If you have any questions regarding the descriptions of the tasks, please raise your questions in the Discussion forum on Moodle by 11:00am, April 16. I will get back to you ASAP.

:::

------

## Question 1-1

Please download the dataset from `demo_data/midterm/jay/`, which is a directory including lyric text files of songs by Jay Chou (周杰倫). Please load the entire corpus into a data frame, preprocess the raw lyrics, and save them in another column of the data frame. A sample data frame is provided below.

When preprocesssing the raw lyrics, please pay attention to the following issues:
   - Remove symbols and punctuations in the lyrics
   - Remove (English) alphabetic characters (including full-width alphabets, e.g. `ｔ`)
   - Remove digits (e.g., `01234`)

A complete output csv file is also available in `demo_data/midterm/question1-1-output-jay.csv`. You can compare your result with this sample csv.

------

- A data frame including both the title (filename), raw lyrics, and preprocessed lyrics of each song:

jay.head()

- When removing symbols, please make sure that the characters before and after the symbol are still properly separated (as shown below):

print("Song Title:", jay.title[100])
print("[Raw Lyrics]:")
print(jay.lyric[100])
print("="*50)
print("[Preprocessed Version]:")
print(jay.lyric_pre[100])

- Also, when removing the alphabets, make sure that the alphabets in full-width forms are removed as well, as shown below (e.g., `ｔｏｎｅ`):

print("Song Title:", jay.title[200])
print("[Raw Lyrics]:")
print(jay.lyric[200])
print("="*50)
print("[Preprocessed Version]:")
print(jay.lyric_pre[200])

## Question 1-2

Following the previous question, create a cluster analysis on all Jay's songs and find out the similarities in-between Jay's songs. Please pay attention to the following issues:

- Use `ckip-transformer` to word-seg the lyrics into word tokens. 
- Please use TF-IDF weighted version of the bag-of-words representations for clustering.
- Please include in the bag-of-word vectorization:
    - (a) words whose minimum document frequency = 2; 
    - (b) words which have at least two characters (i.e., removing all one-character word tokens);
    - (c) words whose parts-of-speech tags indicate they are either NOUNS or VERBS. However, for nouns, please EXCLUDE words that are pronouns （e.g., 你 我 她） or numerals (e.g., 一 二 三). Specifically, include words whose POS tags start with `N` or `V`, but exclude words tagged as `Nh` (i.e., pronouns) or `Neu` (i.e., numerals).

Your output should be a dendrogram as shown below. A complete `jpeg` file of the dendrogram is also available in `demo_data/midterm/question1-2-output-dendrogram.jpeg`.

------


- The Shape of the `CountVectorizer` Matrix After Filtering: ( `Number_of_Songs`,  `Number_of_Features`)

jay_bow_df.shape

- Sample of the `CountVectorizer` Matrix After Filtering:

jay_bow_df

- The Shape of the `TfidfVectorizer` Matrix After Filtering: ( `Number_of_Songs`,  `Number_of_Features`)

tv_matrix.shape

- Sample of the `TfidfVectorizer` Matrix After Filtering (Please use this weighted TF-IDF matrix for clustering):

jay_tv_df.round(2)

- The Pairwise Similarity Matrix of All Songs

similarity_doc_df.round(2)

- The Dendrogram of Songs

![](../exercise-ans/midterm/question1-2-output-dendrogram.jpeg)

## Question 2-1

Use the datasets, `demo_data/midterm/chinese_name_gender_train.txt` (training set) and `demo_data/midterm/chinese_name_gender_test.txt` (testing set), to build a classifier to determine the gender of a Chinese name based on the bag-of-words model. The training set text file includes around 480,000 Chinese names and their gender labels (around 240,000 for each gender). All names have exactly three characters and they have been randomized.

The first step to the building of the classifier is text/name vectorization. Please create a NAME-by-FEATURE matrix using bag-of-words model. However, do not include all characters. Please include in the bag-of-words model only the following features:
   - Any Chinese characters that appear in the second position of the name (e.g., the `英` in 蔡英文)
   - Any Chinese characters that appear in the third position of the name (e.g., the `文` in 蔡英文)
   - Any Chinese character bigrams that appear in the second and the third characters of the name (i.e., the given name, e.g., `英文` in 蔡英文)

For all the above features, they will be included as classifying features only when they appear in at least **100 different names** (i.e., the minimum document frequency threshold).

The expected output of Question 2-1 is the bag-of-word representation of all the names in the training set following the above filtering guidelines. A sample has been provided below.

A complete sample output of the name-by-feature matrix for the training set is also available in `demo_data/midterm/question2-1-output-tv-matrix.csv`. (It is stored as a data frame with the Chinese names as the index and feature names as the columns.)

-----


- For training data, the shape of the NOUN-by-FEATURE matrix is as follows: ( `Number_of_Names_in_the_Training_Set`,  `Number_of_Features`)

X_train_bow.shape

- A Sample of the NOUN-by-FEATURE matrix (Training Set):

X_train_bow_df.head()

- In particular, bigrams that passed the minimum document frequency include (there are 287 bigrams):

X_train_bow_df[[col for col in X_train_bow_df.columns if len(col)>1]].head()

- For testing data, the shape of the NOUN-by-FEATURE matrix is as follows : ( `Number_of_Names_in_the_Testing_Set`,  `Number_of_Features`)
- Please note that the feature number should be exactly the same as the number of the vectorized matrix of the training set.

X_test_bow.shape

## Question 2-2

Following the previous question, please use the NAME-by-FEATURE matrix for classifier training (I used the Count-based version, i.e., `CountVectorizer()`). In order to find the best-performing classifier, please work on the following steps:

- Try two ML algorithms, `sklearn.naive_bayes.GaussianNB` and `sklearn.linear_model.LogisticRegression` and determine which algorithm performs better using *k*-fold cross validation (k = 10). Report the average accuracies of cross-validation for each ML method.
- After cross-validation, you would see that Logistic Regression performs a lot better. In Logistic Regression, there is one hyperparameter `C` and different initial values of C may yield different performances as well. Use Grid Search to fine-tune this parameter from these values: C = [1, 5, 10]. (You may refer to [sklearn's Logistic Regression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) for more detail on `C`.)
- After determining the ML algorithm and hyperparameter tuning, evaluate your final model with the testing set, i.e., `demo_data/midterm/chinese_name_gender_test.txt`. Report the confusion matrix plot of the results as shown below.
- Present LIME explanations of your model on the gender prediction of the following four names: `'王貴瑜','林育恩','張純映','陳英雲'`.
- Finally, perform a post-hoc analysis of the feature importances by looking at the top 10 features of the largest coefficient values for each gender prediction (see below).


------------------


- Cross Validation Results

print("Mean Accuracy of Naive Bayes Model: ", model_gnb_acc.mean())
print("Mean Accuracy of Logistic Regression Model:", model_lg_acc.mean())

- Best Hyperparameter for Logistic Regression from Grid Search:

clf.best_params_

- Confusion Matrix of the Final Model on Testing Set (Normalized):

plot_confusion_matrix(clf, X_test_bow, y_test, normalize='all')
plt.title("Confusion Matrix (Normalized %)")

plot_confusion_matrix(clf, X_test_bow, y_test, normalize=None)
plt.title("Confusion Matrix (Frequencies)")

- LIME Explanations of Names:

explanations[0].show_in_notebook(text=True)

explanations[1].show_in_notebook(text=True)

explanations[2].show_in_notebook(text=True)

explanations[3].show_in_notebook(text=True)

- Feature Coefficients Analysis of Logistic Regression Model (Note: The number of coefficients should be the same as the number of features used in training.)

![](../exercise-ans/midterm/_question2-2-output-featimportance.jpeg)

## Question 3-1

This exercise requires the dataset, `demo_data/midterm/apple5000.csv`, which includes 5000 news articles from Apple Daily. Please use `spacy` and its pre-trained language model `zh-core-web-lg`, to extract word pairs of the dependency relation of `amod`. For example, in the following sequence:

```
"陸軍542旅下士洪仲丘關禁閉被操死，該旅副旅長何江忠昨遭軍高檢向最高軍事法院聲押獲准。何江忠的前同事說：「他（何江忠）只能用『陰險』兩字形容，得罪他都沒好下場。」還說他常用官威逼部下，「仗勢欺人、人神共憤，大家都不喜歡他。」被他帶過的阿兵哥說，懲處到了何手上都會加重，簡直是「大魔頭」。"
```

`spacy` identifies three token pairs showing a `amod` dependency relation, namely:

```
amod dep:  高 head: 軍事
amod dep:  前 head: 同事
amod dep:  大 head: 魔頭
```

Please note that the head and the dependent are NOT necessarily adjacent to each other. For example, in a sentence like:

```
"這是一個漂亮且美麗的作品，明亮的窗戶，房子很大。"
```

`spacy` identifies two token pairs showing a `amod` dependency relation, namely:

```
amod dep:  漂亮 head: 作品
amod dep:  明亮 head: 窗戶
```

With the `apple5000.csv` corpus, your job is to extract all word-pairs that show a `amod` dependency relation using `spacy` dependency parser. (These two word tokens may or may not be adjacent to each other.)

Please follow the following instructions for the analysis.

1. Preprocess each news article by removing symbols, punctuations, digits, and English alphabets (see the sample data frame below).
2. Parse all the articles using `zh-core-web-lg` language model and extract word pairs showing the `amod` dependency relation. 
3. In your final report, please include only word pairs where the nouns are of AT LEAST two syllables/characters. Your final report is a frequency list of these word pairs (see the sample data frame below for the top 50 frequent pairs).
4. A sample output csv is provided in `demo_data/midterm/question3-1-output-modnounfreq.csv`.

---------


- Examples of Raw Texts and Preprocessed Texts

apple_df.head()

- Number of MOD-NOUN Types:

len(mod_head_df)

- Top 50 Frequent MOD-NOUN Showing `amod` dependency relation in Apple News:

mod_head_df.sort_values(['Frequency'],ascending=[False]).head(50)

## Question 3-2

Following the previous question, with the extracted MODIFIER-NOUN word pairs, please create a NOUN-by-MODIFIER co-occurrence table, showing the co-occurring frequencies of a particular noun (i.e., the row) and a particular modifier (i.e., the column) (see the sample data frame below).

In addition, with a co-occurrence matrix like this, we can cluster the NOUNS according their co-occurring patterns with different modifiers. That is, please perform a cluster analysis on the NOUNS, using their co-occurring frequencies with the MODIFIERS as the features. In particular, among all these modifier-noun pairs:
   - please include nouns whose total frequencies are > 70 (i.e., given the NOUN-by-MODIFIER matrix, you need to include only rows whose row sums are > 70)
   - please include modifiers whose total frequencies are > 10 (i.e., given the NOUN-by-MODIFIER matrix, you need to include only columns whose column sums are > 10)
   - perform the cluster analysis using the default settings used in the lecture notes (i.e., cosine similarity, linkage of ward's method).
   
   
:::{important}

In case you fail to create the output from Question 3-1, you can use the sample output csv, `demo_data/midterm/question3-1-output-modnounfreq.csv`, as your starting point for this exercise. 

The csv file is the expected output from Question 3-1, including all the MODIFIER-NOUN pairs identified by `spacy` and their frequency counts in the corpus. (As specified in Question 3-1, bigrams with one-syllable nouns have been removed from the list.)

:::

:::{attention}

My current analysis shows that the closest neighbors for 總統 are 男友 and 女友!!!

:::

-------

- Before filtering, the shape of the NOUN-by-MODIFIER co-occurrence matrix should be as follows: ( `Number_of_Noun_Types`,  `Number_of_Modifier_Types`)

print(noun_by_mod.shape)

- Noun-by-Modifier Co-occurrence Matrix After Filtering

noun_by_mod_filtered_df

- After filtering, the shape of the NOUN-by-MODIFIER co-occurrence matrix should be as follows: (`Number_of_Noun_Types`, `Number_of_Modifier_Types`)

noun_by_mod_filtered_df.shape

- Pairwise Cosine Similarity Matrix for Nouns Whose Frequency > 70

![](../exercise-ans/midterm/_question3-2-output-heatmap.jpeg)

- The Cluster Result, the Dengrogram:

![](../exercise-ans/midterm/question3-2-output-dendrogram.jpeg)