# Classification Models

- Before the era of deep learning, probability-based classifiers are commonly used in many classfic NLP tasks.
- There are two types of probability-based models:
    - **Generative**:
        - Training is based on the **joint** probability of the data and the class, i.e., $P(c,d)$
        - Examples: n-gram models, Naive Bayes, HMM, probablistic context-free grammars.
    
    - **Distriminative**: 
        - Training is based on the **conditional** Probablity of the class given the data, i.e., $P(c|d)$.
        - Examples: Logistic regression, maximum entropy models, CRF, SVM, perceptron.
    
- Strengths of discriminative models:
    - They give high accuracy performance.
    - They make it easier to include linguistically relevant features.
- Other classification models:
    - Tree-based methods (Decision tree)
    - Neural Network

## Naive Bayes

![](../images/naive-bayes.png)

````{margin}
```{note}
In Bayes, the $P(Y)$ is referred to as the **prior probability** of Y and the $P(Y|X)$ is referred to as the **posterior probability** of Y.
```
````

- Naive Bayes features the Bayes Theorem:

$$
P(Y|X) = P(Y) \frac{P(X|Y)}{P(X)} 
$$ (naive_bayes1)

- In naive bayes, given a document $d$ and a class $c$, the goal is to find the maximum joint probability $P(c,d)$. And according bayes rule, the goal is simiplied to find the maximum of $P(c|d)$:

$$
P(c,d) = P(d|c)\times P(c) = P(c|d) \times P(d)\\
P(c|d) = \frac{P(d|c)\times P(c)}{P(d)}
$$ (navie_bayes2)

- Because the $P(d)$ is a constant for all classes estimation, we can drop the denominator. And now the goal is to find out the class $c$ that maximizes the **posterior** probability of the class, i.e., $P(c|d)$.

$$
C_{MAP} = \underset{c \in C}{\arg\max}P(c|d)\\
C_{MAP}= \underset{c \in C}{\arg\max}\frac{P(d|c)\times P(c)}{P(d)} \\
= \underset{c \in C}{\arg\max}P(d|c)\times P(c) \\
$$ (naive_bayes3)


````{margin}
```{note}
In naive Bayes, the probabilities $𝑃(C=c_i)$ and $𝑃(𝑋𝑖|C=c_i)$ are parameters. The standard, maximum likelihood, approach is to calculate the probabilities using MLE estimators. For example, for $𝑃(C=c_i)$ we  count the cases where $C=c_i$ and divide by the sample size, same for conditional probabilities $𝑃(𝑋𝑖|C=c_i)$. Bayesian maximum a posteriori estimate $P(C=c_i|Xi)$ would need from you to assume prior distributions for the parameters. 
```
````

- A document needs to be vectorized through feature engineering into a numeric representation, i.e., a set of **n** features from the document, $\{x_1, x_2, ..., x_n\}$
- Given a class $c$ and a set of **n** features from the document, $\{x_1, x_2, ..., x_n\}$, we can denote the posterior probability of each class $c$ as follows:

$$
C_{MAP}= \underset{c \in C}{\arg\max}  P(d|c) \times P(c) \\
= \underset{c \in C}{\arg\max} P(x_1,x_2,...,x_n|c) \times P(c) \\
$$ (naive_bayes4)

- Important assumptions in Naive Bayes:
    - Bag of Words Assumption: Assume position doesn't matter
    - Conditional Independence: Assume the feature probabilities $P(x_i|c)$ are independent given the label $c$.

$$
P(x_1,x_2,...,x_n|c)=P(x_1|c)\times P(x_2|c)\times P(x_3|c) \times ... \times P(x_n|c)
$$ (naive_bayes5)

- Then we can simply the equation:

$$
C_{MAP} = \underset{c \in C}{\arg\max} P(x_1,x_2,...,x_n|c) \times P(c) \\
= \underset{c \in C}{\arg\max} P(c) \prod_{x\in X}P(x|c)
$$ (naive_bayes6)

- Multinomial Naive Bayes is an extenstion of the Naive Bayes for predicting and classifying data points, where the number of distinct labels or outcomes are more than two.

- In Naive Bayes, each feature $x_i$ of the document is in the form of a probability value of an observed event given the class, $P(x_i|c)$

- Issues of Smoothing
    - For unseen features in a new document, e.g., $P(x|c)$, this would render the entire posterior probability of the class to be zero. 
    - Laplace smoothing (Add-one)
    - Lidstone smooting (Self-defined $\alpha$ < 1)

## Logistic Regression

- Logistic Regression is also a form of probabilistic statistical classification model.
- The model is trained by maximizing directly the probability of the class given the observed data, i.e., $P(c|d)$.
- Logistic Regression is similar to linear regression, whose predicted values are numeric.

- A document needs to be vectorized through feature engineering into a numeric representation, i.e., a set of **n** features from the document, $\{x_1, x_2, ..., x_n\}$

- Then the logistic regression models the probability of the class given these observed values:

$$
y = \beta_0 + \beta_1x_1 + \beta_2 x_2 + ... + \beta_ix_i
= \sum_{i}\beta_ix_i
$$

````{margin}
```{note}
For example, we would hope that our model can give us values between 0 and 1, which represent the probability of the class given the observed data.
```
````

- But the $y$ is not a probability value yet. It could be any number ($-\infty$ to $\infty$) based on the current model. In order to apply a linear modeling approach, the dependent variable $y$ often needs to be transformed with a so-called **link function**.
- Link functions tranform the predicted $y$ into a range more appropriate for linear modeling.
- For binary logistic regression, the **inverse logit transformation** is used, transforming y from the range of $-\infty$ to $\infty$ to into probabily values ranging from 0 to 1.

$$
\frac{1}{1+e^y}
$$

- And we can re-formulate the model equation as:

$$
P(c|d) = P(c|x_1,x_2,...,x_i)\\= \frac{1}{1+e^{\sum_{i}\beta_ix_i}}
$$
- The model will then determine the class of the document by choosing the class that maximizes the conditional probability of $P(c|d)$.

- Each feature $x_i$ of the document is a function that chracterizes the relevant linguistic properties of the document. These features can be all manually annotated or created automatically.
- These features are often in simple forms that are of binary values or numeric values within a range:
    $$ 
    x_i = f_i(c,x_i) = [C = c_i \wedge contains(w_i) ] = \{0, 1\}
    $$
   

## Other Discriminative Models

There are many more ways to learn the weights (i.e., $\beta_i$) in the discriminative models. Different algorithms may prioritize different link functions to train the model parameters.

In this course, we will not cover all the details of these variants. Please refer to machine learning courses for more details on these algorithms.

- **Maximum Entropy Model (Maxent)**
    - Maxent is very similar to logistic regression, and differs in mainly its way of parameterization (e.g., feature functions, link functions)
    - Maxent uses a *softmax* function to transform values into probabilities.
    
$$
p(c|d,\beta) = \frac{e^{\sum_{i}\beta_ix_i}}{\sum_{c'}{e^{\sum_{i}\beta_ix_i}}}\\
= \frac{y_c}{\sum_{c'}{y_{c'}}}
$$

- **Support Vector Machine (SVM)**
    - finds optimal model parameters that can maximize the margins of the classes on the hyperplane.
    - SVM has a **cost function**, which controls the compromise of the training errors and margins. A smaller cost creates a softer margin, allowing more misclassifcations; a larger cost creates a narrower margin, allowing fewer misclassifications.

- **Perceptron**
    - finds a currently misclassified example, and nudge weights in the direction of its correct classification.

## Other Classification Models

- There are other classification models that are not probabilistic
    - **Decision Tree**
        - It uses a split condition to predict class labels based on one or multiple input features.
        - The classification process starts from the root node of the tree; at each node, the classifier will check which input feature will most effectively split the data into sub-groups. 


## Emsemble Methods

- Bagging 
    - Take subsets of data (e.g., bootstrap samples) and train a model on each subset in parallel
    - The subsets are allowed to simultaneously vote on the outcome
    - The final outcome is often an average aggregation.
    - Example: **Random Forests**
- Boosting
    - Utilize sequential modeling by improving one model from the errors of the previous model
    - Often use the output of one model as an input into the next in a form of sequential processing
    - Example: **Gradient boosting machines**

## Conclusion

- Feature engineering is a lot more important than algorithm selection.
- With good feature representations of the data, usually it does not really mather which machine learning algorithm we choose.
- Generally, discriminative models out-perform generative models.
- Maybe one day the optimal algorithm can be determined *automatically* via machine learning as well!!
- Please see [Google's AutoML effort: Google AI Introduces Model Search, An Open Source Platform for Finding Optimal Machine Learning Models](https://www.marktechpost.com/2021/02/28/google-ai-introduces-model-search-an-open-source-platform-for-finding-optimal-machine-learning-ml-models/?fbclid=IwAR3FkqJzXGIqG-X25BXslDZ077F9s4onOaQzlosc9u7qeto9LNd7prt_PNE).

## References

- NLTK Book, Chapter 6.
- Sarkar (2020), Chapter 5.