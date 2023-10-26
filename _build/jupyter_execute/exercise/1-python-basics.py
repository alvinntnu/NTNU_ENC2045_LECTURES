#!/usr/bin/env python
# coding: utf-8

# # Assignment I: Python Basics

# In this course, we assume you have basic working knowledge of the language python. Please finish the exercises in this assignment by writing programs in python. All these exercises cover the necessary knowledge needed for more advanced tasks in our course.
# 

# ```{note}
# Please use default modules provided by python and avoid using non-default modules which require additional installation.
# ```

# 
# ## 
# What are the core default modules available in Python? List down a few of them and explain their functionalities.
# 

# ```{admonition} Ans
# :class: tip
# Answers may vary.
# ```
# 

# ## 
# Assuming that you have not installed the modulel `numpy` in your system. How to install the module?

# ## 
# In Jupyter Notebook, how to get the documentation for the python objects you are working with?

# ## 
# Demonstrate how to subset and slice a `list` in Python with examples.

# ## 
# What are the different ways to generate random numbers? Demonstrate at least the following two ways of random number generation.
# 
# - Generate a random number between 0 and 1
# - Generate a random integer between 0 and 100

# ## 
# A `dict` is provided below. Please demonstrate how to sort the dictionary by (a) its key, and (b) its values in an descending order.
# 
# ```
# example = {'e':10, 'a':22, 'h':17}
# ```
# 
# The expected results:
# 
# ```
# ## According to key:
# [('h', 17), ('e', 10), ('a', 22)]
# ## According to value:
# [('a', 22), ('h', 17), ('e', 10)]
# ```

# ## 
# How do we remove the duplicate elements from the `list` given below?
# 
# ```
# words = [‘one’, ‘one’, ‘two’, ‘three’, ‘three’, ‘two’]
# ```
# 
# The expected output:
# 
# ```
# ['two', 'one', 'three']
# ```

# ## 
# How to convert the strings `123` and `456.6` into  numbers?

# ## 
# How to get the full path to the current working directory of your project in Python?
# 

# ## 
# How to get all the names of the files in the current working directory that have the file extension of `.txt`?

# (question-control)=
# ## 
# Use control structures of `for-loop` and `if-condition` to identify the elements from the `list` below whose values are divisible by 13 but are not a multiple of 5.
# 
# ```
# numbers = range(100,200)
# ```
# 
# The expected results:
# 
# ```
# 104
# 117
# 143
# 156
# 169
# 182
# ```
# 

# ## 
# To achieve the same goal as specified in the previous [question](question-control), please re-write the script with only one line utilizing the **list comprehension** in python.
# 
# For example:
# 
# ```
# numbers = range(100,200)
# [XXXXXX number XXXXX] # list comprehension
# ```
# 
# The expected results are as follows:
# 
# ```
# [104, 117, 143, 156, 169, 182]
# ```

# ## 
# Given a `list` of `tuples` as shown below, please sort the `list` according to the second value of the tuples by **descending** order. If the second values are the same, the next sorting criterion is based on the third values of the tuples. Please note that the first value is a string while the rest should be converted to numbers before sorting.
# 
# The input `list`:
# 
# ```
# data= [('John', '20', '90'), ('Jony', '17', '91'), ('Jony', '17', '93'), ('Json', '21', '85'), ('Tom', '19', '80')]
# ```
# 
# The expected results:
# 
# ```
# [('Json', '21', '85'),
#  ('John', '20', '90'),
#  ('Tom', '19', '80'),
#  ('Jony', '17', '93'),
#  ('Jony', '17', '91')]
# ```

# ## 
# Define a function which can iterate the numbers, which are divisible by 13, between a given range 0 and n, and print all numbers in one line, separated by commas.
# 
# Expected outputs of `generateNumbers(100)`:
# 
# `0,13,26,39,52,65,78,91`
# 

# ## 
# Write a program which accepts the following sequence of words separated by whitespace as input and print the words composed of digits only.
# 
# ```
# s = '2 cats and 3 dogs.'
# ```
# 
# The expected output:
# 
# ```
# ['2', '3']
# ```

# ## 
# Define a function which can compute the sum of two numbers.
# 
# For example, you define a function `addNums()`:
# 
# 
# The expected results of `addNums(1,4)`:
# 
# ```
# 5
# ```

# ## 
# Write a program that accepts as input a long string, where a series of words are separated by commas and prints the words in a comma-separated sequence after sorting them alphabetically.
# 
# The input long string:
# 
# ```
# s = "without,hello,bag,world"
# ```
# 
# The expected output:
# 
# ```
# bag,hello,without,world
# ```

# ## 
# Write a program that accepts an input string and computes the total number of upper case letters and lower case letters in the entire string.
# 
# The input string:
# 
# ```
# 
# s = """
# Hello world!
# Then, the output should be:
# UPPER CASE 1
# LOWER CASE 9
# """
# ```
# 
# The expected Outputs:
# 
# ```
# UPPER CASE: 20
# LOWER CASE: 29
# ```

# ## 
# Explain what the following code chunk is doing in your own words step by step (i.e., line by line). Please paraphrase the scripts as clearly as possible.

# In[36]:


li = [1,2,3,4,5,6,7,8,9,10]
evenNumbers = map(lambda x: x**2, filter(lambda x: x%2==0, li))
[n for n in evenNumbers]


# ## 
# Write a program that accepts a long string and identifies strings that are possibly emails.
# 
# The input string:
# 
# ```
# s = """
# This is a short text with example@ntnu.edu.tw emails that we will use as@gmail.com examples. Some emails are complex like alvin-chen@my_domain.org and some sentences do not have any email address, like the emoticon here --> @.@.
# """
# ```
# 
# The expected output:
# 
# ```
# ['example@ntnu.edu.tw', 'as@gmail.com', 'alvin-chen@my_domain.org']
# ```
