#!/usr/bin/env python
# coding: utf-8

# # ENC2045 Computational Linguistics
# 
# ```{warning}
# These lecture notes are still work-in-progress. There will always be last-minute changes to the files and notebooks. Please ALWAYS download the most recent version of the notebooks before the class of each week.
# ```
# 
# ## Introduction
# 
# Computational Linguistics (CL) is now a very active sub-discipline in applied linguistics. Its main focus is on the computational text analytics, which is essentially about leveraging computational tools, techniques, and algorithms to process and understand natural language data (in spoken or textual formats). Therefore, this course aims to introduce useful strategies and common workflows that have been widely adopted by data scientists to extract useful insights from natural language data. In this course, we will focus on textual data.
# 
# ## Course Objectives
# 
# A selective collection of potential topics may include:
# 
# <i class="fa fa-check fa-1x" style="color:DarkTurquoise;margin-right:5px"></i> A Pipeline for Natural Language Processing
# 
# -   Text Normalization
# -   Text Tokenization
# -   Parsing and Chunking
# 
# <i class="fa fa-check fa-1x" style="color:DarkTurquoise;margin-right:5px"></i> Chinese Processing
# 
# -   Issues for Chinese Language Processing (Word Segmentation)
# 
# <i class="fa fa-check fa-1x" style="color:DarkTurquoise;margin-right:5px"></i> Machine Learning Basics
# 
# -   Feature Engineering and Text Vectorization
# -   Traditional Machine Learning
# -   Classification Models (Naive Bayes, SVM, Logistic Regression)
# 
# <i class="fa fa-check fa-1x" style="color:DarkTurquoise;margin-right:5px"></i>
# Common Computational Tasks
# 
# -   Sentiment Analysis
# -   Tex Clustering and Topic Modeling
# 
# <i class="fa fa-check fa-1x" style="color:DarkTurquoise;margin-right:5px"></i>
# Deep Learning NLP
# 
# -   Neural Language Model
# -   Sequence Models
# -   RNN
# -   LSTM/GRU
# -   Sequence-to-sequence Model
# -   Attention-based Models
# -   Explainable Artificial Intelligence and Computational Linguistics
# 
# This course is extremely hands-on and will guide the students through classic examples of many task-oriented implementations via in-class theme-based tutorial sessions. The main coding language used in this course is Python. We will make extensive use of the language. It is assumed that you know or will quickly learn how to code in **Python**. In fact, this course assumes that every enrolled student has working knowledge of Python. (If you are not sure whether you fulfill the prerequisite, please contact the instructor first.)
# 
# A test on Python Basics will be conducted on the second week of the class to ensure that every enrolled student fulfills the prerequisite. (To be more specific, you are assumed to have already had working knowledge of all the concepts included in the book, Lean Python: Learn Just Enough Python to Build Useful Tools). Those who fail on the Python basics test are NOT advised to take this course.
# 
# Please note that this course is designed specifically for linguistics majors in humanities. For computer science majors, this course will not feature a thorough description of the mathematical operations behind the algorithms. We focus more on the practical implementation.

# ## Course Website
# 
# All the course materials will be available on the [course website ENC2045](https://alvinntnu.github.io/NTNU_ENC2045/). You may need a password to access the course materials/data. If you are an officially enrolled student, please ask the instructor for the passcode.
# 
# We will also have a Moodle course site for assignment submission.
# 
# Please read the FAQ of the course website before course registration.

# ## Major Readings
# 
# The course materials are mainly based on the following readings.
# 
# ```{image} images/book-nltk.jpg 
# :alt: ntlk 
# :width: 200px 
# :align: center
# ```
# 
# ```{image} images/book-pnlp.jpg 
# :alt: ntlk 
# :width: 200px 
# :align: center
# ```
# 
# ```{image} images/book-text-analytics.jpg 
# :alt: text-analytics 
# :width: 200px 
# :align: center
# ```
# 
# ```{image} images/book-sklearn.jpg 
# :alt: sklearn 
# :width: 200px 
# :align: center
# ```
# 
# ```{image} images/book-dl.jpg 
# :alt: dl 
# :width: 200px 
# :align: center
# ```

# ## Recommended Books
# 
# Also, there are many other useful reference books, which are listed as follows in terms of three categories:
# 
# -   Python Basics {cite}`gerrard2016lean`
# -   Data Analysis with Python {cite}`mckinney2012python,geron2019hands`
# -   Natural Language Processing with Python {cite}`sarkar2019text,bird2009natural,vajjala2020,perkins2014python,srinivasa2018natural`
# -   Deep Learning {cite}`francois2017deep`

# ## Online Resources
# 
# In addition to books, there are many wonderful on-line resources, esp. professional blogs, providing useful tutorials and intuitive understanding of many complex ideas in NLP and AI development. Among them, here is a list of my favorites:
# 
# -   [Toward Data Science](https://towardsdatascience.com/)
# -   [LeeMeng](https://leemeng.tw/)
# -   [Dipanzan Sarkar’s articles](https://towardsdatascience.com/@dipanzan.sarkar)
# -   [Python Graph Libraries](https://python-graph-gallery.com/)
# -   [KGPTalkie NLP](https://kgptalkie.com/category/natural-language-processing-nlp/)
# -   [Jason Brownlee’s Blog: Machine Learning Mastery](https://machinelearningmastery.com/)
# -   [Jay Alammar’s Blog](https://jalammar.github.io/)
# -   [Chris McCormich’s Blog](https://mccormickml.com/)
# -   [GLUE: General Language Understanding Evaluation Benchmark](https://gluebenchmark.com/)
# -   [SuperGLUE](https://super.gluebenchmark.com/)

# ## YouTube Channels <i class="fa fa-youtube"></i>
# 
# -   [Chris McCormick AI](https://www.youtube.com/channel/UCoRX98PLOsaN8PtekB9kWrw/videos)
# -   [Corey Schafer](https://www.youtube.com/channel/UCCezIgC97PvUuR4_gbFUs5g)
# -   [Edureka!](https://www.youtube.com/channel/UCkw4JCwteGrDHIsyIIKo4tQ)
# -   [Deeplearning.ai](https://www.youtube.com/channel/UCcIXc5mJsHVYTZR1maL5l9w)
# -   [Free Code Camp](https://www.youtube.com/c/Freecodecamp/videos)
# -   [Giant Neural Network](https://www.youtube.com/watch?v=ZzWaow1Rvho&list=PLxt59R_fWVzT9bDxA76AHm3ig0Gg9S3So)
# -   [Tech with Tim](https://www.youtube.com/channel/UC4JX40jDee_tINbkjycV4Sg)
# -   [PyData](https://www.youtube.com/user/PyDataTV/featured)
# -   [Python Programmer](https://www.youtube.com/channel/UC68KSmHePPePCjW4v57VPQg)
# -   [Keith Galli](https://www.youtube.com/channel/UCq6XkhO5SZ66N04IcPbqNcw)
# -   [Data Science Dojo](https://www.youtube.com/c/Datasciencedojo/featured)
# -   [Calculus: Single Variable by Professor Robert Ghrist](https://www.youtube.com/playlist?list=PLKc2XOQp0dMwj9zAXD5LlWpriIXIrGaNb)

# ## Environment Setup
# 
# 1.  We assume that you have created and set up your python environment as follows:
#     -   Install the python with [Anaconda](https://www.anaconda.com/products/individual)
#     -   Create a conda environment named `python-notes` with `python 3.7`
#     -   Run the notebooks provided in the course materials in this self-defined conda environment
# 2.  We will use `jupyter notebook` for python scripting in this course. All the assignments have to be submitted in the format of jupyter notebooks. (See [Jupyter Notebook installation documentation](https://jupyter.org/install)).
# 3.  We assume you have created a conda virtual environment, named, `python-notes`, for all the scripting, and you are able to run the notebooks in the conda environment kernel in Jupyter.
# 4.  You can also run the notebooks directly in Google Colab, or alternatively download the `.ipynb` notebook files onto your hard drive or Google Drive for further changes.
# 
# ``` {.tip}
# If you cannot find the conda environment kernal in your Jupyter Notebook. Please instal the module `ipykernel`. For more details on how to use specific conda environments in Jupyter, please see [this article](https://medium.com/@nrk25693/how-to-add-your-conda-environment-to-your-jupyter-notebook-in-just-4-steps-abeab8b8d084).
# ```
# 
# When you run the installation of the python modules and other packages in the terminal, please remember to change the arguments of the parameters when you follow the instructions provided online.
# 
# For example, when you create the conda environment:
# 
# `conda create --name XXX`
# 
# You need to specify the conda environment name `XXX` on your own.
# 
# Similarly, when you add the self-defined conda environment to the notebook kernel list:
# 
# `python -m ipykernel install --user --name=XXX`
# 
# You need to specify the conda environment name `XXX`.
# 
# :::{warning}
# 
# There are several important things here:
# 
# - You need to install the relevant modules AFTER you activate the conda environment in the terminal.
# - You need to add the kernel name with `python -m ipykernel install --user --name=XXX` within the conda enviroment as well.
# - In other words, you need to install the module `ipykernel` in the target conda environment as well.
# - After a few trial-and-errors, I think the best environment setting is that you **only** add the kernel name (conda environment) to `ipykernel` within the conda environment. **Do not** add the conda environment **again** in your base python environment.
# - What's even better is to install `jupyter` in your conda environment (`python-notes`) and run your notebook from this `python-notes` as well.
# 
# :::
# 
# For Windows users, it is highly recommended to run the installation of python-related modules in `Anaconda Prompt` instead of `cmd`.

# ## Windows Users
# 
# Because some of the window's users are suffering from the problem that the notebook cannot find the correct path of your `python-notes` environment. Could you please try the following?
# 
# ```{tip}
# We assume you run the following steps in Anaconda Powershell Prompt.
# ```
# 
# - Create the new conda environment, `python-notes` (if you have done this, ignore)
# 
# ```
# $ conda create --name python-notes python=3.7
# ```
# 
# - Activate the newly-created `python-notes` conda environment
# 
# ```
# $ conda activate python-notes
# ```
# 
# - Within the `python-notes` environment, install the module `ipykernel`
# 
# ```
# $ conda install ipykernel
# ```
# 
# - Deactivate the conda environment
# 
# ```
# $ conda deactivate
# ```
# 
# - Install the module `nb_conda_kernels` in your Anaconda base python (This is crucial to the use of self-defined conda environments in notebook.)
# 
# ```
# $ conda install nb_conda_kernels
# ```
# 
# - Then check if the system detects your `python-notes` environment
# 
# ```
# $ python -m nb_conda_kernels list
# ```
# 
# :::{note}
# Ideally, you should see your `python-notes` popping up in the results like:
# ```
# conda-env-python-notes-py    c:\users\alvinchen\anaconda3\envs\python-notes\share\jupyter\kernels\python3
# ```
# :::
# 
# 
# - If you see the new conda environment name, then it should work. Next, initiate your jupyter notebook
# 
# ```
# $ jupyter notebook
# ```
# 
# - Now you should be able to see you conda environment `python-notes` in your notebook, but the kernel name would be something like `python [conda evn: python-notes]`. 
# 
# 

# ## Module Requirements
# 
# Not all modules used in this course come with the default installation of the python environment. Please remember to install these packages in use if you get a module-not-found type of errors.
# 
# The following is a list of packages we will use (non-exhaustive):
# 
# -   `beautifulsoup4`
# -   `ckip-transformers`
# -   `gensim`
# -   `jieba`
# -   `jupyter`
# -   `jupyter-nbextensions-configurator`
# -   `Keras`
# -   `matplotlib`
# -   `nltk`
# -   `numpy`
# -   `pandas`
# -   `scikit-learn`
# -   `seaborn`
# -   `spacy`
# -   `scipy`
# -   `tensorflow`
# -   `transformers`
# 
# Please google these packages for more details on their installation.

# ## Coding Assignments Reminders
# 
# 1.  You have to submit your assignments via Moodle.
# 2.  Please name your files in the following format: `Assignment-X-NAME.ipynb` and `Assignment-X-NAME.html`.
# 3.  Please always submit both the Jupyter notebook file and its HTML version.
# 
# ``` {.{.{warning}}}
# Unless otherwise specified in class, all assignments will be due on the date/time given on Moodle. Late work within **7 calendar days** of the original due date will be accepted by the instructor at the instructor's discretion. After that, no late work will not be accepted.
# ```

# ## References
# 
# ``` {bibliography} ./book.bib
# :filter: docname in docnames
# :style: unsrt
# ```

# 
# ```{toctree}
# :hidden:
# :titlesonly:
# :caption: INTRODUCTION
# 
# nlp/nlp-primer
# nlp/nlp-pipeline
# ```
# 
# 
# ```{toctree}
# :hidden:
# :titlesonly:
# :caption: Preprocessing
# 
# nlp/text-preprocessing
# ```
# 
# 
# ```{toctree}
# :hidden:
# :titlesonly:
# :caption: Text Vectorization
# 
# nlp/text-vec-traditional
# ```
# 
# 
# ```{toctree}
# :hidden:
# :titlesonly:
# :numbered: 2
# :caption: Machine Learning Basics
# 
# nlp/ml-overview
# nlp/ml-simple-case
# nlp/ml-algorithm
# ```
# 
# 
# ```{toctree}
# :hidden:
# :titlesonly:
# :numbered: 2
# :caption: Machine-Learning NLP
# 
# nlp/ml-sklearn-classification
# nlp/topic-modeling-naive
# ```
# 
# 
# ```{toctree}
# :hidden:
# :titlesonly:
# :numbered: 2
# :caption: Deep Learning NLP
# 
# nlp/dl-neural-network-from-scratch
# nlp/dl-simple-case
# nlp/dl-sentiment-case
# ```
# 
# 
# ```{toctree}
# :hidden:
# :titlesonly:
# :numbered: 2
# :caption: Neural Language Model and Embeddings
# 
# nlp/dl-sequence-models-intuition
# nlp/dl-neural-language-model-primer
# nlp/text-vec-embedding
# ```
# 
# 
# ```{toctree}
# :hidden:
# :titlesonly:
# :caption: Seq2Seq, Attention, Transformers, and Transfer Learning
# 
# temp/dl-seq-to-seq-types
# temp/dl-seq-to-seq-attention-addition
# temp/dl-transformers-intuition
# temp/dl-transformers-keras
# ```
# 
# 
# ```{toctree}
# :hidden:
# :titlesonly:
# :caption: Exercises
# 
# exercise/1-python-basics
# exercise/2-journal-review
# exercise/3-preprocessing
# exercise/4-chinese-nlp
# exercise/5-text-vectorization
# exercise/6-machine-learning
# exercise/midterm-exam
# exercise/7-topic-modeling
# exercise/8-dl-chinese-name-gender
# exercise/9-sentiment-analysis-dl
# exercise/10-neural-language-model
# exercise/11-word2vec
# ```
# 
