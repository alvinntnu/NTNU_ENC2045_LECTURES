# Assignment I: Python Basics

In this course, we assume you have basic background knowledge with the language python. 



## 
What are the core default modules available in Python? List down a few of them.


```{admonition} Ans
:class: tip
Answers may vary.
```


## 
Assuming that you have not installed the modulel `numpy` in your system. How to install the module?

```{admonition} Ans
:class: tip

`pip install numpy`

`conda install numpy`
```



## 
Demonstrate how to subset and slice a `list` in Python with examples.

l = [1, 2, 3, 4, 5]
l[1:3]
l[1:-2]
l[-3:-1]  # negative indexes in slicing
s = "Hello World"
s[1:3]
s[:-5]
s[-5:]

## 
What are the different ways to generate random numbers?

    - Generate a random number between 0 and 1
    - Generate a random integer between 0 and 100

import random

print(random.random())
print(random.uniform(0, 1))
print(random.randint(0, 100))

## 
A `dict` is provided below. Please demonstrate how to sort the dictionary by (a) its key, and (b) its values in an descending order.

```
example = {'e':10, 'a':22, 'h':17}
```

The expected results:

```
## According to key:
[('a', 22), ('e', 10), ('h', 17)]
## According to value:
[('a', 8), ('e', 10), ('h', 17)]
```

example = {'e': 10, 'a': 22, 'h': 17}
print(sorted(example.items(), key=lambda x: x[0]))
print(sorted(example.items(), key=lambda x: x[1]))

## 
How do we remove the duplicate elements from the given list?

```
words = [‘one’, ‘one’, ‘two’, ‘three’, ‘three’, ‘two’]
```


words = ["one", "one", "two", "three", "three", "two"]
list(set(words))

## 
How to convert the strings `123` and `456.6` into  numbers?

s1 = '123'
s2 = '456.6'
print(type(s1))
print(type(s2))
print(type(int(s1)))
print(type(float(s2)))

## 
How to get the current working directory of your project in Python?


import os
os.getcwd()

(question-control)=
## 
Use control structures of for-loop and if-condition to identify the elements from the `list` below whose values are the multiples of 3.

```
numbers = [0,4,8,9,15,7,29,33]
```

numbers = [0,4,8,9,15,7,29,33]
for n in numbers:
    if n % 3 ==0:
        print(n)

## 
To achive the same goal as specified in the previous [question](question-control), please re-write the script with one line utilizing the **list comprehension** in python.

For example:

```
numbers = [0,4,8,9,15,7,29,33]
[XXXXXX number XXXXX]
```

And then you get the following output:

```
[0, 9, 15, 33]
```

numbers = [0,4,8,9,15,7,29,33]

[n for n in numbers if n % 3 == 0]

## 
Given a `list` of `tuples` as shown below, please sort the `list` according to the second value of the tuples by **descending** order. If the second values are the same, the second criterion for sorting is the third values. Please note that the first value is string while the rest should be converted to numbers before sorting.

```
data= [('John', '20', '90'), ('Jony', '17', '91'), ('Jony', '17', '93'), ('Json', '21', '85'), ('Tom', '19', '80')]
```

The expected results:

```
[('Json', '21', '85'),
 ('John', '20', '90'),
 ('Tom', '19', '80'),
 ('Jony', '17', '93'),
 ('Jony', '17', '91')]
```

data= [('John', '20', '90'), ('Jony', '17', '91'), ('Jony', '17', '93'), ('Json', '21', '85'), ('Tom', '19', '80')]

sorted(data, key=lambda x:(x[1],x[2]), reverse= True)

## 
Define a function which can iterate the numbers, which are divisible by 13, between a given range 0 and n, and print all line by line to the console.


def generateNumbers(n):
    i = 0
    while i<n:
        j=i
        i=i+1
        if j%13==0:
            print(j)
            
print("Expected outputs of `generateNumbers(100)`:\n")            
generateNumbers(100)

