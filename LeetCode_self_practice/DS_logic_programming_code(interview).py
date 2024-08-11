# ------zzz--------
from collections import defaultdict
from email.policy import default
from heapq import merge
from itertools import count
import numbers
from pydoc import cli
from re import M
from turtle import width
from unicodedata import digit

from numpy import array
from sympy import isprime
# ----------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pydataset import dataset
import seaborn as sns
# Get the list of available example dataset names
dataset_names = sns.get_dataset_names()
# Print the list of dataset names
for name in dataset_names:
    print(name)

# Choose a dataset name from the available list
dataset_name = 'mpg'  # For example, you can choose 'iris'
# Load the chosen dataset
df = sns.load_dataset(dataset_name)
# Display the first few rows of the dataset
print(df.head())
# --------------------------

\\=-------------------------------------------------------
# Top 15 Python Programming Questions & Solutions -asked in Amazon, Facebook, Microsoft, Tesla Interviews
# link:https://www.youtube.com/watch?v=BCZWQTY9xPE

# !1. Write a Python Program to print Prime Numbers between 2 numbers
# !*2. Write a Sort function to sort the elements in a list
# !3. Write a sorting function without using the list.sort function
#! 4. Write a Python program to print Fibonacci Series
# 5. Write a Python program to print a list in reverse (same as qn: 22)
# 6: Write a matrix 10x10 code or generalized code (Addition, Multiplication, etc)
# 6. Write a Python program to check whether a string is a Palindrome or not
# 6.Write a Python Program to Count the Number of Digits in a Number?
# 6. Write a Python Program to Find the Second Largest Number in a List?
# 6.Write a Python Program to Swap the First and Last Value of a List?
# 6.Write a Python Program to Check if a String is a Palindrome or Not?
# 7. Write a Python program to print set of duplicates in a list
# 8. Write a Python program to print number of words in a given sentence                                    ***
# 9. Given an array arr[] of n elements, write a Python function to search a given element x in arr[].
# 10. Write a Python program to implement a Binary Search
# 11. Write a Python program to plot a simple bar chart
# 12. Write a Python program to join two strings (Hint: using join())                                         **
# 13. Write a Python program to extract digits from re import search                                       **
# 14. Write a Python program to split strings using newline delimiter
# 15. Given a string as your input, delete any reoccurring character, and return the new string.
# 16. Reversed number in Python
# 17. Reversed string in Python
# 18. How do you calculate the number of vowels and consonant peresent in a string
# 19. How do you get a matching element in an integer array
# 20. Code the bubble sorts algorithm
# 21. how do you reverse an array (same as qn:22)
# 22. How do you reverse a list   (same question as asked in qn:5 and 21)
# 23. Swap the two numbers without third variable.
# 24. How do you implement a binary search
# 25. find the second largest number in the array
# 26. Write a code to print a Table of 5 upto 20.
# 27. Critical path Method/ analysis for project management
# 28. write a factorial function code for n natural number
# 29. Write codes for  Job Sequencing Problem
# 30. Python Program for Sum of squares of first n natural numbers
# 31. Python Program for cube sum of first n natural numbers
# 32. Python Program for simple interest
# 33. Binary search tree algorithm
# Q34: Python Program for Linear Search
# Q35: Python Program for Insertion Sort
# Q36: Python Program for Recursive Insertion Sort
# Q37: Python Program for QuickSort
# Q38: Python Program for Iterative Quick Sort
# Q39: Python Program for Selection Sort
# Q40: Python Program for Heap Sort
# 41: -Binary tree and Binary search tree algorithm
# 42. Recursion programming
# 43  Convert a sorted list into a binary search tree
# 44. Bubble sort algorithm  (shorting list array: numeric & string)
# 45. Merg sort algorithm  (shorting list array: numeric & string)
# 46. Quick sort algorithm  (shorting list array: numeric & string)
# 47. Heap sort algorithm  (shorting list array: numeric & string)

# 48. write Dijkstra's Algorithm (Shortest path method: Greegy Algorithm)
# 49. Greedy algorithm (Activity  selection problem)
# 50. Greedy algorithm (Coin Change Problem)
# 51. Greedy algorithm (FractionalKnapsack problrm: item associates weights and values)
# 52. BFS (Breadth First Search)  Graph Traversal algorithm.
# 53. DFS (Deapth First Search)  Graph Traversal algorithm.
# 54: Capacity To Ship Packages Within D Days , A conveyor belt has packages that must be shipped from one port to another within  D days days
# 55: Minimize the maximum difference between the heights of tower either by increasing or decreasing with k (only once)
#  56: Finding missing number
# 57: Count the total prime number less than a given number N
# 58: Finding the maximum sum of the 'k' consequtive elements
# 59: Boats to save people
# 60: Containers with most waters (using towers heigh)
# 61: Finding single number (only once time occurence)
# 62: Summing multiple values of each key in a dict?
# 63: Two digit sum numbers to get a target value
# 64: Finding duplicate value using hashmap
# 65: Mejority of the element in an array
# 66: Group Anangrams using Hashmap
# 67: Recursively sum of the digits
# 68: Recursively fabonacci numbers
# 69: Recursively factorial numbers
# 70: Recursively powers number calculation
# 71: Recursively find greatest commom diviser (GCD)
# 72: Recursively find palindrom
# 73: Recursively print the star pattern
# 74:  Recursively solving Binary Search problem
# 75: Python Calender of months
# 76: Stock maximum profit
\\=----------------------------------------------------------------

#!Q1: Write a python program to check prime number

# 1st method : It is 100% accurate.


def prime(n):
    count = 0
    for i in range(1, n+1):
        if n % i == 0:
            count = count+1
            # print( 'Count number is ={:.1f} at {}'.format(count,i))
    if (count == 2):
            return 'It is a prime number'
    else:
        return 'It is not a prime number'
print(prime(5))

# 2nd method: it is not 100% accurate result:
n = 25
if n > 1:
    for i in range(2, n+1):
        if n % i == 0:
            print("It is not a prime number")
            break
        else:
            print("It is a prime number")
            break


# 3rd method: It is not 100% accurate result

number = int(input("Enter any number: "))
if number > 1:
    for i in range(2, number+1):
        if (number % i == 0):
            print('It is not a prime number')
            break
        else:
            print('It is a prime number')
            break
else:
    print('Please enter any number greater than 1 in order to check the prime number')

# 3rd method by defining a defination.


def primeCheck(X):
        if X > 1:
            for i in range(2, X+1):
                if X % i == 0:
                    return 'It is not a prime number'
                    # print('It is not a prime number')
                    break
                else:
                    return 'It is a prime number'
                    # print('It is a prime number')
        else:
            return 'Plase enter any number greater than 1 in  order to check prime number'
            # print('Plase enter any number greater than 1 in  order to check prime number')


# Write a programm to print the list of the 100 prime numbers or n prime numbers.
# It is not %100 accurate
lower = 1
upper = 100

print("Prime numbers between", lower, "and", upper, "are:")

for num in range(lower, upper + 1):
   # all prime numbers are greater than 1
   if num > 1:
       for i in range(2, num):
           if (num % i) == 0:
               break
       else:
           print(num, end=',')


# --------2nd methods------------------------
lower = 1
upper = 100

prime_number = []

print("Prime numbers between", lower, "and", upper, "are:")

for num in range(lower, upper + 1):
   # all prime numbers are greater than 1
   if num > 1:
       for i in range(2, num):
           if (num % i) == 0:
               break
       else:
           prime_number.append(num)
print(prime_number)

# Q2: write a code for odd/even number in python
# 1st methods

num = int(input("Enter any number: "))

if (num % 2 == 0):
    print('It is an even number')
else:
    print('Enter value greater than 1')

# 2nd methods


def OddEven(X):
    if (X % 2 == 0):
        print('It is an even number')
    else:
        print('It is an odd number')


# 3rd methods (under construction):
while True:
    try:
        num = int(input("Enter any number: ")
        if (num % 2 == 0):
            print("{} is an even number".format(num))
        else:
            print("{} is an odd number".format(num))
            break
    except ValueError:
        print("Oops!  That was no valid number.  Try again...")
print('You succesfully run your Odd even number problem')

# Q3: Write a code to swap (interchange) two variable value like x into y and y into.

# fisrt method
x=input("Enter X value : ")
y=input("Enter Y value : ")
# Swapping function
temp=x
x=y
y=temp

print("The value of x after swapped : {}".format(x))
print("The value of y after swapped: {}".format(y))

# 2nd method

def swap(x, y):
    x, y=y, x
    print("The value of x after swapped : {}".format(x))
    print('The value of y after swapped:{}'.format(y))

# Q4: Write a programm code for even list

import numpy as np

# ---------for loop:-------------------

import numpy as np
l=np.linspace(1, 20, 20, dtype='int')
l_even=[]
for item in l:
    if item % 2 == 0:
        l_even.append(item)
print("You have list of all even numbers :", l_even)

# -------2nd methos:----------

n=20
l_even=[]
for item in range(1, n+1):
    if item % 2 == 0:
        l_even.append(item)
print("You have list of all even numbers :", l_even)

# ------3rd method:----using while loop:------------------=--
n=20
l_even=[]
while (n > 1):
    if n % 2 == 0:
        l_even.append(n)
    n=n-1
print("You have list of all even numbers:", l_even)

# --------------list comprehension:---------

l=np.linspace(1, 20, 20, dtype='int')
l_even_comp=[item for item in l if item % 2 == 0]
print("You have list of all even numbers :", l_even_comp)

!- -------------------OR-------------------

l=np.linspace(1, 20, 20, dtype='int')
l_even_comp=[x for x in l if x % 2 == 0]
print("You have list of all even numbers :", l_even_comp)


# ---------------------------------------
# Q5: Write a program to filter out the number divided by 3 from the tuple
# using lambda function

# 1st method

l=np.linspace(1, 20, 20, dtype='int')

l_filter=tuple(filter(lambda x: (x % 3 == 0), l))
print(l_filter)

# 2nd Method
l=np.linspace(1, 20, 20, dtype='int')
l_filter=list(filter(lambda x: (x % 3 == 0), l))
print(l_filter)

# 3rd method
l=np.linspace(1, 20, 20, dtype='int')
list_every_3rd=[]
for item in l:
    if item % 3 == 0:
        list_every_3rd.append(item)
print("You have list of all every 3rd element :", list_every_3rd)

# Q6: Write a matrix 10x10 code or generalized code
#!link:https://www.youtube.com/watch?v=66hIDupiJjo
# 1st method
matrix=np.arange(1, 101).reshape(10, 10)
# 2nd method
row=int(input("Enter the number of rows :"))
col=int(input("Enter the number of columns"))

Matrix=[]

for i in range(row):
    a=[]
    for j in range(col):
        a.append(int(input()))
    Matrix.append(a)
print(Matrix)

# -----------

# Matrix Multiplication problem Related:
# !3x3 matrix
X=[[12, 7, 3],
   [4, 5, 6],
   [7, 8, 9]]
#! 3x4 matrix
Y=[[5, 8, 1, 2],
   [6, 7, 3, 0],
   [4, 5, 9, 1]]

# result
# !3x4 matrix
result=[[0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]]

# !iterate through rows of X
for i in range(len(X)):
    # !iterate through columns of Y
    for j in range(len(Y[0])):
        # !iterate through rows of Y
        for k in range(len(Y)):
            result[i][j] += X[i][k]*Y[k][j]
print(result)

# Matrix summation problem Related:

# !3x3 matrix
X=[[12, 7, 3],
   [4, 5, 6],
   [7, 8, 9]]
#! 3x4 matrix
Y=[[5, 8, 1, 2],
   [6, 7, 3, 0],
   [4, 5, 9, 1]]

# Result:
result=[[sum(a*b for a, b in zip(X_rows, Y_rows))
             for Y_col in zip(*Y) for X_rows in X]]

# ------------------------------------------------------
# Q6:Write a Python Program to Check if a Number is a Palindrome or not?
n=int(input("Enter number:"))
temp=n
rev=0
while (n > 0):
    dig=n % 10
    rev=rev*10+dig
    n=n//10

if (temp == rev):
    print("The number is a palindrome!")
else:
    print("The number isn't a palindrome!")


# Q6:Write a Python Program to Count the Number of Digits in a Number?
n=int(input("Enter number:"))
count=0
while (n > 0):
    count=count+1
    n=n//10

print("The number of digits in the number is:", count)

# Q6: Write a Python Program to Find the Second Largest Number in a List?
a=[]
n=int(input("Enter number of elements:"))
for i in range(1, n+1):
b=int(input("Enter element:"))
a.append(b)
a.sort()
print("Second largest element is:", a[n-2])

# Q6:Write a Python Program to Swap the First and Last Value of a List?

a=[]
n=int(input("Enter the number of elements in list:"))
for x in range(0, n):
element=int(input("Enter element" + str(x+1) + ":"))
a.append(element)
temp=a[0]
a[0]=a[n-1]
a[n-1]=temp
print("New list is:")
print(a)

# Q6.Write a Python Program to Check if a String is a Palindrome or Not?
string=input("Enter string:")
if (string == string[::-1]):
    print("The string is a palindrome")
else:
     print("The string isn't a palindrome")

# Q7: Write a code for finding average of n numbers

def avg_num(x):
    sum_num=0
    for i in x:
        sum_num=sum_num+i
    avg=sum_num/len(x)
    return avg
# note: x could be a list


# Q7:How to write a code for list count
list1=['red', 'green', 'blue', 'orange', 'green', 'gray', 'green']
color_count=list1.count('green')
print('The count of color: green is ', color_count)

# Q8: Write a Python program to print number of words in a given sentence
# !1st method
a_string='one two three'
word_list=a_string.split()
number_of_words=len(word_list)
print(number_of_words)

# Q8: How can you count duplicate elements in a given list?

list1=[3, 3, 3, 3, 5, 6, 7, 8, 9, 10]
num_count=list1.count(3)
print('The count of duplicate count: 3 is ', num_count)

# Q9: Write a code to get index of an element in a list using for loop

my_list=['Guru', 'Siya', 'Tiya', 'Guru', 'Daksh', 'Riya', 'Guru']
all_indexes=[]
for i in range(0, len(my_list)):
    if my_list[i] == 'Guru':
        all_indexes.append(i)
print("Original_list ", my_list)
print("Indexes for element Guru : ", all_indexes)

# Q10: Write a programme for any dataset using any classifier to find the accuracy of the model

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pydataset import data

# dataset: IRIS
df1=sns.load_dataset('IRIS')
df2=data('iris')

# ---------Descptive statistic:----
df1.isnull().sum()
df1.shape
df1.describe()
df1.info()

# -------Selecting the target variable:---------

X=df1.iloc[:, :4]
X.info()
X.shape

Y=df1.iloc[:, -1]
Y.shape

# -----Importing sklearn libraries

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---importing Decision tree classifier:----------
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# ------- Now spliiting the dataset into training and testing:----

X_train, X_test, Y_train, Y_test=train_test_split(
    X, Y, test_size=0.20, random_state=42)

X_train.shape
Y_train.shape
X_test.shape
Y_test.shape
# -----------Now training our model using training datset:------
model=DecisionTreeClassifier()
model.fit(X_train, Y_train)

# ----Now predicting : X-test data

predictions=model.predict(X_test)

# -----Checking model accuracy:----------
print(accuracy_score(Y_test, predictions))
print(classification_report(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))


# --------------------------------------------------------------
# Q11: Write a programme for Linear regression model and check their accuracy report as well

df=pd.read_csv('E:/Z-Jupyter/USA_Housing.csv')
df.head()
df.info()
df.shape

sns.pairplot(df)
sns.displot(df['Price'])

df.isnull().sum()

df.corr()
sns.heatmap(df.corr(), annot=True)

# ----Now selcting the target variable:-----
df.info()

X=df.iloc[:, :5]
X.info()

Y=df.loc[:, 'Price']
Y.info()

X.shape
Y.shape

# ---Now splitting the datset into training and testing :-----
X_train, X_test, Y_train, Y_test=train_test_split(
    X, Y, test_size=0.33, random_state=42)

X_train.shape
Y_train.shape

# ----Now importing linear regression models
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

model=LinearRegression()
model.fit(X_train, Y_train)


predictions=model.predict(X_test)

predictions
Y_test

# -----
print(model.intercept_)
print(model.coef_)

# -------scatter plot:-----
sns.scatterplot(Y_test, predictions)
plt.legend(True)
# -----Distplot:----
sns.distplot(Y_test-predictions)
# ----------------------------------------------------------------
# ----model accuracy checking:----

mse=mean_absolute_error(Y_test, predictions)

rmse=np.sqrt(mse)
rmse




# ---------------------------------------------------
import pandas as pd
import numpy as np

df=pd.read_excel('Selectiondata.xlsx')
df.isna().sum()

df1=df.fillna(0)
df1.isna().sum()

df1=df1.astype(int)

df1['D']=np.where(df1['ST'] == df1['AD'], '1', '0')
df1

# duplicate_value=[]

# for i in range(df.shape[0]):
#     if df1['ST']=df1['AD']:
#         print(df['ST'])

# print(duplicate_value)

df1.shape[0]
# -------------------------------

# Q: Write the code for finding the maximum number of the given two
def maxfind(a, b):
    if a > b:
        return a
    else:
        return b

a=-10
b=-20
print(maxfind(a, b))

# Q: Write the programming for the finding circle area

def circleArea(r):
    area=(22/7)*r**2
    # area=(22/7)*(r*r)
    return area
r=10
print("Area of the circle is: %.2f" % circleArea(r))

# Q: Write the programming code for finding the sum of the square of first n natural number

def squar_n_number(n):
            sm=0
            for i in range(1, n+1):
                sm=sm+(i*i)
            return sm

print(squar_n_number(64))

# Q: Write a programme code to the cube sum of the first n natural numbers;

def cube_sum_n_number(n):
    sm=0
    for i in range(1, n+1):
        sm=sm+(i**3)
    return sm

cube_sum_n_number(4)


# Q: Sum of the all n natural numbers' cube with alternative sign.

def cub_alt_sign(n):
    sm=0
    for i in range(1, n+1):
        if (i % 2) == 1:                 # to get the all odd number
            sm=sm+(i*i*i)  # sum the all odd numbers
        else:
            sm=sm-(i*i*i)            # to get the all even number
    return sm                        # sum the all even numbers

cub_alt_sign(5)

# Q: Sum of cube of n-even number

# https://www.geeksforgeeks.org/sum-of-cubes-of-first-n-even-numbers/?ref=lbp

def cube_sum_even(n):
    sm=0
    for i in range(1, n+1):
        sm=sm + (2*i)*(2*i)*(2*i)
    return sm

print(cube_sum_even(8))

# Q: Sum of cube of n-odd number

# https://www.geeksforgeeks.org/sum-of-cubes-of-first-n-odd-natural-numbers/?ref=lbp
def cube_sum_odd(n):
    sm=0
    for i in range(0, n):
        sm=sm+(2*i+1)*(2*i+1)*(2*i+1)
    return sm
print(cube_sum_odd(2))

# Q: write a factorial function code for n natural number

def factorial_n(n):
    if n < 0:
        return 0
    elif n == 0 or n == 1:
        return 1
    else:
        fact=1
        while (n > 1):
            fact=fact*n  # fact*=n
            n=n-1  # n-=1
        return fact
print(factorial_n(2))

# ----2nd methos: for factorial-----------------------------------
n=4
fact=1
if n < 0:
    print("factorial is not possible")
while (n > 0):
         fact=fact*n
         n=n-1

print("The factorial is:", fact)


# Q: Write a program to generate fibonacci series

def febonacci_series(n):
            # First two term
            a, b=0, 1
            count=0
            # Check if the number of terms is valid
            if n <= 0:
                print("The enter a positive integer")
            # If there is only one term, return n1
            elif n == 1:
                print("Fibonacci sequence upto", n, ':')
                print(a)
                # generate fibonacci sequence
            else:
                print("Fibonacci sequence:")
                while (count <= n):
                    print(a)
                    sum=a+b
                    a=b
                    b=sum
                    count=count+1

print(febonacci_series(5))

# Q: Write a program to create a stars pattern of numbers using for loop
# 1.Square patter
n=4
for i in range(n):
    for j in range(n):
        print(" * ", end='')
    print()

# 2.Left Triangle patter
n=4
for i in range(n):
    for j in range(i+1):
        print(" * ", end='')
    print()


# 2.Revers Left Triangle patter
# (1st method)
n=4
for i in range(n):
    for j in range(4-i):
        print(" * ", end='')
    print()
# (2nd method)
n=4
for i in range(n):
    for j in range(i, n):
        print(" * ", end='')
    print()

# 2.lower Right Triangle patter
n=5
for i in range(n):
    for j in range(n-i-1):
        print(" ", end=" ")
    for j in range(i+1):
        print("*", end=" ")
    print()

# Reverse left triangle
n=5
for i in range(n):
    for j in range(n-i):
        print('*', end=' ')
    for j in range(i):
        print(' ', end=' ')
    print()

# Reverse right triangle
n=5
for i in range(n):
    for j in range(5-i-1):
        print('*', end=' ')
    for j in range(i):
        print(' ', end=' ')
    print()

# Q: Printing pyramid stars using while loop

# link:https://www.youtube.com/watch?v=PTHSTjBfXmY

# n=int(input("Enter the number of rows :"))
n=10
k=1
i=1
while i <= n:
    b=1
    while b <= n-i:
        print(" ", end="")
        b=b+1
    j=1
    while j <= k:
        print("*", end="")
        j=j+1
    print()
    i=i+1
    k=k+2



# Q: Printing reverse pyramid stars using while loop function

# n=input("Enter the no of rows: ")
n=10
i=1
while (n > 0):
    b=1
    while (b < i):
        print(" ", end=" ")
        b=b+1
    j=1
    while (j <= n*2-1):         # we could also put 1 inplace of i
        print("*", end=" ")
        j=j+1
    print()
    n=n-1
    i=i+1

# Q: Write the code to printing the stars in the shape of hallow heart

for row in range(6):
    for col in range(7):
        if (row == 0 and col % 3 != 0) or (row == 1 and col % 3 == 0) or (row-col == 2) or (row+col == 8):
            print("*", end="")
        else:
            print(" ", end="")
    print()


# Q: Write the code to printing the stars in the shape of solid heart

for i in range(4):
    for j in range(4-i-1):
        print(" ", end="")
    for j in range(i+1):
        print("* ", end="")
    for j in range(2*(4-i-1)):
         print(" ", end="")
    for j in range(i+1):
        print("* ", end="")
    print()
for i in range(8, 0, -1):
    for j in range(8-i):
        print(" ", end="")
    for j in range(i, 0, -1):
        print("* ", end="")
    print()

# Q: Write the code to printing the stars in the shape of solid heart with writing text
# link:https://www.youtube.com/watch?v=6lJqSEvE1Rw

num=int(input("Enter the number :"))
n=num//2

for i in range(n):
    for j in range(n-i-1):
        print(" ", end="")
    for j in range(i+1):
        print("* ", end="")
    for j in range(2*(n-i-1)):
         print(" ", end="")
    for j in range(i+1):
        print("* ", end="")
    print()

for i in range(2*n, 0, -1):
    for j in range(2*n-i):
        print(" ", end="")
    for j in range(i, 0, -1):
        print("* ", end="")
    print()

# second method

num=int(input("Enter the number :"))
msg=input("Enter the message")
l=len(msg)
n=num//2

for i in range(n):
    print(" "*(n-i-1)+"* "*(i+1), end="")
    if num % 2 == 0:
        for j in range(n-i-1):
            print(" ", end="")
        else:
            for j in range(2*(n-i-1)):
                print(" ", end="")
        else:
            for j in range(2*(n-i-1)):
                print(" ", end="")
        for j in range(i+1):
            print("* ", end="")
        print()
for i in range(num, 0, -1):
    print(" "*(num-i)+"* "*(i))
# ----------------------------------------------------------------
# printing starts
n=4
for i in range(n):
    for j in range(n):
        print("*", end="")
    print()
# ----------------------------------------------
n=8
for i in range(n):  # ith row
    for j in range(i):  # jth column
        print(" * ", end="")
    print()

# -------------------------------------
n=8
for i in range(n, 1, -1):  # ith row
    for j in range(i-1):  # jth column
        print(" * ", end="")
    print()
# -------------------------------
n=8
for i in range(n, 0, -1):
    for k in range(0, n-i):
        print(" ", end="")
    for j in range(1, i+1):
        print("*", end="")
    print("\n")

# Printing hollow square

n=18
for i in range(n):
    for j in range(n):
        if (i == 0 or i == n-1 or j == 0 or j == n-1):
            print("*", end="")
        else:
            print(" ", end="")
    print('\n')


# Printing a letter 'A'

n=5
for i in range(n+2):
    for j in range(n):
        if (((j == 0 or j == n-1) and i != 0) or (i == 0 or i == 3) and (j > 0 and j < 4)):
            print("*", end="")
        else:
            print(" ", end="")
    print('\n')


#  -------------Rough work  ----------------------------------------
count=0
sum=0
print('before :', "Count=", count, "Sum =", sum)
for value in [9, 41, 12, 3, 74, 15]:
    count=count+1
    sum=sum+value
    print(count, sum, value)

print("After :", "Count=", count, "Sum =", sum, "Average =", sum/count)

# ----------ending rough work----------------------------------

# Q16: Write a code to print the reverse order of any digits (at least two digits)

Number=int(input("Please Enter any Number: "))
Number1=Number
Reverse=0
while (Number > 0):
    Reminder=Number % 10
    Reverse=(Reverse*10)+Reminder
    Number=Number//10
# print("Reverse of the entered number is =%d" %Reverse)
print("Reverse of the entered number ={} is ={}".format(Number1, Reverse))

# -------------exp---------------------------------------
# Number=int(input("Please Enter any Number: "))
Number=1234
Number1=Number
Reverse=0
while (Number > 0):
    Reminder=Number % 10
    print("Reminder =%d" % Reminder)
    Reverse=(Reverse*10) + Reminder
    print("Revsersed =%d" % Reverse)
    Number=Number//10
    print("Numbrrs for which reminder is taken =%d" % Number)
# print("Reverse of the entered number is =%d" %Reverse)
print("Reverse of the entered number ={} is ={}".format(Number1, Reverse))

# -------2nd methods:-----
num=123456
print(str(num)[::-1])

# --------------------------
# Q17. Reversed string in Python
#!-----1st method: using for loop----
str='Hello'
reversed_str=''
for i in str:
     reversed_str=i+reversed_str

print(reversed_str)

#!--2nd Method: using while loop-----

str='Zeeshan'
reversed_str=''

n=len(str)
while (n > 0):
    reversed_str=reversed_str+str[n-1]
    # reversed_str+=str[n-1]
    n=n-1
print(reversed_str)

#! --3rd Method: using for loop
name='Zeeshan Haleem'
n=len(name)
rd=''
for i in range(n):
    # print(n-i-1)         #It is used to get the reversed order numbering
    rd=rd+name[n-i-1]
print(rd)

# ----------------------------------------------
# Q18:How do you calculate the number of vowels and consonant peresent in a string

vowels=['a', 'e', 'i', 'o', 'u']  # fisrt defining a list of vowels
# Input a string and convert it into lower classifier
str=input('Enter a string :').lower()

# define a counter variable
v_count=0
c_count=0

# Iterate through the character of the input variable
for x in str:
    if x in vowels:
        v_count=v_count+1

    elif x not in vowels:
        c_count=c_count+1

print('Character', str)
print('Total vowels :', v_count)
print('Total consonant :', c_count)

# Q19. How do you get a matching element in an integer array
# !1st method
a=[1, 2, 3, 4, 5]
b=[9, 8, 7, 6, 5]
matched_element=set(a) & set(b)   # set method
print(matched_element)

# !2nd method
a=[1, 2, 3, 4, 5]
b=[9, 8, 7, 6, 5]

# list comprehension method
matched_element=[i for i, j in zip(a, b) if i == j]
print(matched_element)

# !3rd method

a=[1, 2, 3, 4, 5, 12]
b=[9, 8, 7, 6, 5, 12]
mat_ls=[]

for i in a:
    for j in b:
        if i == j:
            mat_ls.append(i)
print(mat_ls)


# Q19: find matching elements counts in array of numbers

ls1=[5, 4, 1, 3, 2, 5, 6, 7, 8]
ls2=[1, 2, 3, 2]

res=len([ls1.index(i) for i in ls2])

print('The original list1', ls1)
print('The original list2', ls2)
print('The match indices list count is :', res)

# Q22: Write a Python program to print a list in reverse order

# !1st method
studentNames=['Hannah', 'Imogen', 'Lewis', 'Peter']
print(studentNames[::-1])
# !2nd method

studentNames=['Hannah', 'Imogen', 'Lewis', 'Peter']
studentNames.reverse()
print(studentNames)

# !3rd method
studentNames=['Hannah', 'Imogen', 'Lewis', 'Peter']
for i in reversed(studentNames):
    print(i)
# !4th method : from scratch
lst=['Hannah', 'Imogen', 'Lewis', 'Peter']
new_list=[]
for i in range(len(lst)):
    new_list.append(lst[-i-1])
print(new_list)

# !-----------OR-----------------
lst=['Hannah', 'Imogen', 'Lewis', 'Peter']
new_list=[]
for i in range(1, len(lst)+1):
    new_list.append(lst[-i])
print(new_list)

!- ----------------------------------------
x=[1, 2, 3, 4, 5]
newx=[]
for i in range(1, len(x)+1):
  newx.append(x[-i])
#   print (x[-i])
print(newx)
# !-----------------------------------
# !-----------------------------------
# Q25:- find the second largest number in the array

#!1st method (sorting method)
ls=[10, 20, 4, 45, 99]
ls.sort()
# we took 2nd position element from the last
print("Second largest element is :", ls[-2])

#!2nd method (sorting method)

#!link: https://www.tutorialspoint.com/python-program-to-find-the-second-largest-number-in-a-list
#!link:https://www.studytonight.com/python-programs/python-program-to-find-second-largest-number-in-a-list

ls=[11, 22, 1, 2, 5, 67, 21, 32]

max_=max(ls[0], ls[1])
secondmax=min(ls[0], ls[1])

for i in range(2, len(ls)):
    # if found element is greater than max_
    if ls[i] > max_:
        secondmax=max_
        max_=ls[i]
        # if found element is greator than secondmax
    else:
        if ls[i] > secondmax:
            secondmax=ls[i]
print("Second largest element in the list is:", secondmax)

# Q26:- Write a code to print a Table of 5 upto 20.

n=5
for i in range(1, 21):
    print(n, 'X', i, '=', n*i)

# 2nd Method-Write a code to print a Table of 5 upto 20.

n=int(input("Enter any number you want table"))
for i in range(1, 21):
    print(n, 'X', i, '=', n*i)

# ----------------------------------------------
# Q27: Creating bins (for goodness of fit of normal distribution)
# Q27: Creating CPM ( Critical Path Method/ Analysis) for Project managemet (Operations Research)

import pandas as pd
import datetime
import matplotlib.pyplot as plt
import networkx as nx
from criticalpath import Node
import plotly.express as px
from IPython.display import Image


# set up the tasks:
tasks=[("A", {"Duration": 3}),
         ("B", {"Duration": 5}),
         ("C", {"Duration": 2}),
         ("D", {"Duration": 3}),
         ("E", {"Duration": 5})]

# set up the dependencies along all paths:
dependencies=[("A", "C"),
                ("B", "C"),
                ("A", "D"),
                ("C", "E"),
                ("D", "E")]

# initialize (directed) graph
G=nx.DiGraph()

# add tasks and dependencies (edges)
G.add_nodes_from(tasks)
G.add_edges_from(dependencies)

# set up the (arbitrary) positions of the tasks (nodes):
pos_nodes={"A": (1, 3),
             "B": (1, 1),
             "C": (2, 2),
             "D": (3, 3),
             "E": (4, 2)}

# draw the nodes
nx.draw(G, with_labels=True, pos=pos_nodes,
        node_color='lightblue', arrowsize=20)


# set up the (arbitrary) positions of the durations labels (attributes):
pos_attrs={node: (coord[0], coord[1] + 0.2)
                  for node, coord in pos_nodes.items()}
attrs=nx.get_node_attributes(G, 'Duration')

# draw (write) the node attributes (duration)
nx.draw_networkx_labels(G, pos=pos_attrs, labels=attrs)


# set a little margin (padding) for the graph so the labels are not cut off
plt.margins(0.1)

# initialize a "project":
proj=Node('Project')

# load the tasks and their durations:
for t in tasks:
    proj.add(Node(t[0], duration=t[1]["Duration"]))

# load the dependencies (or sequence):
for d in dependencies:
    proj.link(d[0], d[1])

# update the "project":
proj.update_all()


# proj.get_critical_path() will return a list of nodes
# however, we want to store them as strings so that they can be easily used for visualization later
crit_path=[str(n) for n in proj.get_critical_path()]

# get the current duration of the project
proj_duration=proj.duration

print(f"The current critical path is: {crit_path}")
print(">"*50)
print(f"The current project duration is: {proj_duration} days")

# create a list of edges using the current critical path list:
crit_edges=[(n, crit_path[i+1]) for i, n in enumerate(crit_path[:-1])]

# first, recreate the network visualization:
nx.draw(G, with_labels=True, pos=pos_nodes,
        node_color='lightblue', arrowsize=20)
nx.draw_networkx_labels(G, pos=pos_attrs, labels=attrs)

# now add the critical path as an additional layer on top of the original graph:
nx.draw_networkx_edges(G, pos=pos_nodes, edgelist=crit_edges,
                       width=10, alpha=0.5, edge_color='r')

# again, leaving some margin so the labels are not cut off
plt.margins(0.1)

# Q29. Write codes for  Job Sequencing Problem

def printJobScheduling(arr, t):

    # length of array
    n=len(arr)

    # Sort all jobs according to
    # decreasing order of profit
    for i in range(n):
        for j in range(n - 1 - i):
            if arr[j][2] < arr[j + 1][2]:
                arr[j], arr[j + 1]=arr[j + 1], arr[j]

    # To keep track of free time slots
    result=[False] * t

    # To store result (Sequence of jobs)
    job=['-1'] * t

    # Iterate through all given jobs
    for i in range(len(arr)):

        # Find a free slot for this job
        # (Note that we start from the
        # last possible slot)
        for j in range(min(t - 1, arr[i][1] - 1), -1, -1):

            # Free slot found
            if result[j] is False:
                result[j]=True
                job[j]=arr[i][0]
                break

    # print the sequence
    print(job)

# Driver COde
arr=[['a', 2, 100],  # Job Array
       ['b', 1, 19],
       ['c', 2, 27],
       ['d', 1, 25],
       ['e', 3, 15]]

print("Following is maximum profit sequence of jobs")

# Function Call
printJobScheduling(arr, 3)

#!----------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
# Q33: Binary search tree algorithm

def bin_search(list, n):
    l=0
    u=len(list)-1
    while l <= u:
        mid=(l+u)//2
        if list[mid] == n:
            globals()['pos']=mid
            return True
        else:
            if list[mid] < n:
                l=mid
            else:
                u=mid

list=[4, 7, 8, 12, 45, 99, 102, 702, 10987, 5666]
n=7

if bin_search(list, n):
    print("Found at", pos+1)
else:
    print("Not Found")



# Q34: Python Program for Linear Search
# Q35: Python Program for Insertion Sort
# Q36:Python Program for Recursive Insertion Sort
# Q37: Python Program for QuickSort
# Q38: Python Program for Iterative Quick Sort
# Q39: Python Program for Selection Sort
# Q40: Python Program for Heap Sort
# 41:-Binary tree and Binary search tree algorithm

#! 1st: Approch sorting element/string data

class BinarySearchTreeNode:
    def __init__(self, data):
        self.data=data
        self.left=None
        self.right=None

    def add_child(self, data):
        if data == self.data:
            return
        if data < self.data:
            # add data in the left
            if self.left:
                self.left.add_child(data)
            else:
                self.left=BinarySearchTreeNode(data)
        else:
            if self.right:
                self.right.add_child(data)
            else:
                self.right=BinarySearchTreeNode(data)

    def in_order_traversal(self):
        elements=[]
        # visit left tree first
        if self.left:
            elements += self.left.in_order_traversal()
        # visit base node
        elements.append(self.data)
        # visit right tree
        if self.right:
            elements += self.right.in_order_traversal()
        return elements

def build_tree(elements):
    root=BinarySearchTreeNode(elements[0])
    for i in range(1, len(elements)):
        root.add_child(elements[i])
    return root

if __name__ == '__main__':

    numbers=[17, 4, 1, 20, 9, 23, 18, 34]
    list=['India', 'Pakistan', 'Germany', 'USA', 'China', 'India', 'UK', 'USA']
    numbers_tree=build_tree(list)
    print(numbers_tree.in_order_traversal())

#!2nd method-------Now searching method using binarySearchTreeNode:---------

class BinarySearchTreeNode:

    def __init__(self, data):
        self.data=data
        self.left=None
        self.right=None

    def add_child(self, data):
        if data == self.data:
            return
        if data < self.data:
            # add data in the left
            if self.left:
                self.left.add_child(data)
            else:
                self.left=BinarySearchTreeNode(data)
        else:
            if self.right:
                self.right.add_child(data)
            else:
                self.right=BinarySearchTreeNode(data)

    def in_order_traversal(self):
        elements=[]
        # visit left tree first
        if self.left:
            elements += self.left.in_order_traversal()
        # visit base node
        elements.append(self.data)
        # visit right tree
        if self.right:
            elements += self.right.in_order_traversal()
        return elements
    def search(self, val):
        if self.data == val:
            return True
        if val < self.data:
            # val might be in left subtree
            if self.left:
                return self.left.search(val)
            else:
                return False
        if val > self.data:

            # val might be in the right tree
            if self.right:
                return self.right.search(val)
            else:
                return False


def build_tree(elements):
    root=BinarySearchTreeNode(elements[0])
    for i in range(1, len(elements)):
        root.add_child(elements[i])
    return root

if __name__ == '__main__':
    numbers=[17, 4, 1, 20, 9, 23, 18, 34]
    countries=['India', 'Pakistan', 'Germany',
        'USA', 'China', 'India', 'UK', 'USA']
    country_tree=build_tree(countries)

    # print( numbers_tree.search(20))
    print("USA is in the list", country_tree.search("USA"))
    print("Sweeden is in the list", country_tree.search("Sweeden"))







#!--------------------------------------------
# Q42: Recursion programming
# 1st method
def greet():
    print("Hello")
    greet()
print(greet())

# 2nd method

import sys
sys.setrecursionlimit(500)
print(sys.getrecursionlimit())

i=0
def greet():
    global i
    i += 1
    print("Hello", i)
    greet()

greet()
        return n
    else:
# 3rd method: factorial using recursion

def fact(n):
    if n == 1:
        return n
    else:
        return n*fact(n-1)

print(fact(5))

#!---------------------------------------------
# Q43: Convert a shorted list into a binary search tree

# Q44: Bubble short algorithm  (shorting list array: numeric & string)

def bubble_sort(elements):
    size=len(elements)

    for i in range(size-1):
        for j in range(size-1):
            if elements[j] > elements[j+1]:
                tmp=elements[j]
                elements[j]=elements[j+1]
                elements[j+1]=tmp

if __name__ == '__main__':
    # elements = [5,9,2,1,67,34,88,34]
    elements=['zeeshan', 'rehan', 'Tabrez', 'Farhan']
    bubble_sort(elements)
    print(elements)

# 45. Merge short algorithm  (shorting list array: numeric & string)

def merge(customList, l, m, r):
    n1=m-l+1
    n2=r-m

    L=[0]*(n1)
    R=[0]*(n2)

    for i in range(0, n1):
        L[i]=customList[l+i]


    for j in range(0, n2):
        R[j]=customList[m+1+j]

    i=0
    j=0
    k=l

    while i < n1 and j < n2:
        if L[i] <= R[j]:
            customList[k]=L[i]
            i += 1
        else:
            customList[k]=R[j]
            j += 1
        k += 1

    while i < n1:
        customList[k]=L[i]
        i += 1
        k += 1

    while j < n2:
        customList[k]=R[j]
        j += 1
        k += 1

def mergeSort(customList, l, r):
            if l < r:
                m=(l+(r-1))//2
                mergeSort(customList, l, m)
                mergeSort(customList, m+1, r)
                merge(customList, l, m, r)
            return customList

cList=[2, 1, 7, 6, 5, 3, 4, 9, 8]

print(mergeSort(cList, 0, len(cList)-1))

# 46. Quick short algorithm  (shorting list array: numeric & string)

def partition(customList, low, high):
    i=low-1
    pivot=customList[high]

    for j in range(low, high):
        if customList[j] <= pivot:
            i += 1
            customList[i], customList[j]=customList[j], customList[i]
    customList[i+1], customList[high]=customList[high], customList[i+1]
    return (i+1)
def quickSort(customList, low, high):
    if low < high:
        pi=partition(customList, low, high)
        quickSort(customList, low, pi-1)
        quickSort(customList, pi+1, high)
    return customList

cList=[2, 1, 7, 6, 5, 3, 4, 9, 8]

print(quickSort(cList, 0, 8))

# 47. Heap short algorithm  (shorting list array: numeric & string)


# Q48: write Dijkstra's Algorithm (Shortest path method: Greegy Algorithm)


# Q49. Greedy algorithm (Activity  selection problem)


# 1st: method
activities=[["A1", 0, 6],
             ["A2", 3, 4],
             ["A3", 1, 2],
             ["A4", 5, 8],
             ["A5", 5, 7],
             ["A6", 8, 9]
             ]
def printMaxActivities(activities):
    n=len(activities)
    activities.sort(key=lambda x: x[2])
    i=0
    firstA=activities[i][0]
    print(firstA)
    for j in range(n):
        if activities[j][1] > activities[i][2]:
            print(activities[j][0])
            i=j
printMaxActivities(activities)

# 2nd: method

def Activities(s, f):
    n=len(f)
    f.sort()
    print("The selected activities position are after sorted array by its finishing time:")
    # The first activity is always selected
    i=0
    print(i, end='')
    # For rest of the activities
    for j in range(n):
        # if start time is greater than or equal to that of previous activity
        if s[j] >= f[i]:
            print(j, end="")
            i=j
# main
s=[0, 3, 1, 5, 5, 8]
f=[6, 4, 2, 8, 7, 9]

Activities(s, f)

# 3rd: Method (using Job sequence profit method)
'''Logic:
1> Sort all jobs in deacreasing order of profit
2> Iterate on jobs in decreasing order of profit.
(a) Find a time slot i, such that slot is empty and i< deadline and i is greates. Put the job in this slot and mark this slot filled
(b) If no such i exists, then ignore the job.'''

# 4th : Method:- Chocolate distribution problem

'''Given an array of n integers where each value represents the number
of chocolates in a packet. Each packet can have a variable number
of chocolates. There are m students, the task is to distribute chocolate packets such that:
1>Each student gets one packet.
2> The difference between the number of chocolates in the packet with
maximum chocolates and packet with minimum chocolates given to the students is minimum.'''

# arr[0......n-1] reprsents sizes of packets
# M is the number student
# Return minimum difference between maximum and minimum values of distribution.

def Chocolate_problem(arr, N, M):
    # Sort the given packets
    arr.sort()
    # if there are no chocolates or number of students is 0,
    if (N == 0 or M == 0):
        return 0
    # Number of students cannot be more than number of packets
    if (N < M):
        return -1
    # Largest number of chocolates
    diff=arr[N-1]-arr[0]
    # Find the subarray of size m such that difference bw last (maximum in case of sorted) elements of subarray is minimum
    for i in range(len(arr)-M+1):
        diff=min(diff, arr[i+M-1]-arr[i])
    return diff

# Driver code
arr=[12, 4, 7, 9, 2, 23, 25, 41,
          30, 40, 28, 42, 30, 44, 48,
          43, 50]
M=7  # Number of students
N=len(arr)
print("Minimum difference is", Chocolate_problem(arr, n, m))


# 5th : Method:- Chocolate distribution problem

def findMinDiff(self, A, N, M):
    # code here
    i=0
    j=M-1
    d=float('inf')
    A.sort()
    while j < N:
        d=min(d, A[j]-A[i])
        j += 1
        i += 1
    return d


N=8   # N is the length of array
M=5    # M is the number of students
A={3, 4, 1, 9, 56, 7, 9, 12}    # Number of chocolate

print(findMinDiff(A, N, M))




# 6th: Method:- Program for Chocolate and Wrapper Puzzle

'''Given the following three values, the task is to find the
total number of maximum chocolates you can eat.
1> money: Money you have to buy chocolates
2> price: Price of a chocolate
3> wrap: Number of wrappers to be returned for getting one extra chocolate.''''

'''Input: money = 16, price = 2, wrap = 2
Output:   15
Price of a chocolate is 2. You can buy 8 chocolates from
amount 16. You can return 8 wrappers back and get 4 more
chocolates. Then you can return 4 wrappers and get 2 more
chocolates. Finally you can return 2 wrappers to get 1
more chocolate.'''

'''Logic: '''

def countMaxChoco(money, price, wrap):
    if (money < price):
        return 0
    # First find number of chocolates that can be purchased with given amount
    choc=int(money/price)

    # Now just add number of chocolates with the chocolates gained by wrappers
    choc=choc+(choc-1)/(wrap-1)
    return int(choc)

# Driver code
money=16  # total money
price=2  # cost of each candy
wrap=2  # no of wrappers needs to be exchanged for one chocolates

print(countMaxChoco(money, price, wrap))



# 50. Greedy algorithm (Coin Change Problem)

def coinChange(total_number, coins):
    N=total_number
    coins.sort()
    index=len(coins)-1
    while True:
        coinValue=coins[index]
        if N >= coinValue:
            print(coinValue)
            N=N-coinValue
        if N < coinValue:
            index=index-1
        if N == 0:
            break

coins=[1, 2, 5, 20, 50, 100]
coinChange(201, coins)

# 51. Greedy algorithm (FractionalKnapsack problrm: item associates weights and values)
1st method:

class Item:
    def __init__(self, weight, value):
        self.weight=weight
        self.value=value
        self.ratio=value/weight

def knapsackMethod(items, capacity):
    items.sort(key=lambda x: x.ratio, reverse=True)
    usedCapacity=0
    totalValue=0
    for i in items:
        if usedCapacity + i.weight <= capacity:
            usedCapacity += i.weight
            totalValue += i.value
        else:
            unusedWeight=capacity-usedCapacity
            value=i.ratio*unusedWeight
            usedCapacity += unusedWeight
            totalValue += totalValue

        if usedCapacity == capacity:
            break
    print("Total value obtained:"+str(totalValue))



item1=Item(20, 100)
item2=Item(30, 120)
item3=Item(10, 60)

cList=[item1, item2, item3]

knapsackMethod(cList, 50)
print(knapsackMethod(cList, 50))

# -------------------------------------------------
# 2nd Method:   (Not an optimal sol: taking pofit rather profit ratio)

'''Problem: Take the fruit with the highest profit that
Does not exceed the weight limit'''



items=[('Avocado', 2.2, 170), ('Pomelo', 8, 1500), ('Durian', 22, 1500),
        ('Cucamelon', 0.26, 15), ('Lychee', 0.4, 20), ('Star apple', 1, 200)]

def greedy_fruit(items, capacity):
    # Sorting table based on profit (descending order: decreasing order of the profit colum)
    items=sorted(items, key=lambda x: x[1], reverse=True)
    chosen_fruits={}
    profit=0

    for i in range(len(items)):
        name, value, weight=items[i]
        num_of_fruit=(capacity-capacity % weight)/weight
        chosen_fruits[name]=int(num_of_fruit)
        capacity=capacity % weight         # reminder will be the remaining capacity
        profit += num_of_fruit*value
    return round(profit, 2), chosen_fruits

print(greedy_fruit(items, 20000))


# 3rd Method:   (An optimal sol: taking profit ratio)

'''Problem: Take the fruit with the highest profit that
Does not exceed the weight limit'''



items=[('Avocado', 2.2, 170), ('Pomelo', 8, 1500), ('Durian', 22, 1500),
        ('Cucamelon', 0.26, 15), ('Lychee', 0.4, 20), ('Star apple', 1, 200)]

def greedy_fruit(items, capacity):
    # Sorting table based on profit (descending order: decreasing order of the profit colum)
    items=sorted(items, key=lambda x: x[1]/x[2], reverse=True)
    chosen_fruits={}
    profit=0

    for i in range(len(items)):
        name, value, weight=items[i]
        num_of_fruit=(capacity-capacity % weight)/weight
        chosen_fruits[name]=int(num_of_fruit)
        capacity=capacity % weight         # reminder will be the remaining capacity
        profit += num_of_fruit*value
    return round(profit, 2), chosen_fruits

print(greedy_fruit(items, 20000))

# ----------------------------------------------------------

# 52. BFS (Breadth First Search)  Graph Traversal algorithm.


class Graph:
    def __init__(self, gdict=None):
        if gdict is None:
            gdict={}
        self.gdict=gdict

    def addEdge(self, vertex, edge):
        self.gdict[vertex].append(edge)

    def bfs(self, vertex):
        visited=[vertex]
        queue=[vertex]

        while queue:
            deVertex=queue.pop(0)
            print(deVertex)

            for adjacentVertex in self.gdict[deVertex]:
                if adjacentVertex not in visited:
                    visited.append(adjacentVertex)
                    queue.append(adjacentVertex)

customDict={'a': ['b', 'c'],
            'b': ['a', 'd', 'e'],
            'c': ['a', 'e'],
            'd': ['b', 'e', 'f'],
            'e': ['d', 'f'],
            'f': ['d', 'e']
            }

graph=Graph(customDict)
print(graph.bfs('a'))



# 53. DFS (Deapth First Search)  Graph Traversal algorithm.


class Graph:
    def __init__(self, gdict=None):
        if gdict is None:
            gdict={}
        self.gdict=gdict

    def addEdge(self, vertex, edge):
        self.gdict[vertex].append(edge)

    # def bfs(self, vertex):
    #     visited =[vertex]
    #     queue = [vertex]

    #     while queue:
    #         deVertex = queue.pop(0)
    #         print(deVertex)

    #         for adjacentVertex in self.gdict[deVertex]:
    #             if adjacentVertex not in visited:
    #                 visited.append(adjacentVertex)
    #                 queue.append(adjacentVertex)

    def dfs(self, vertex):
        visited=[vertex]
        stack=[vertex]

        while stack:
            popVertex=stack.pop()
            print(popVertex)

            for adjacentVertex in self.gdict[popVertex]:
                if adjacentVertex not in visited:
                    visited.append(adjacentVertex)
                    stack.append(adjacentVertex)

customDict={'a': ['b', 'c'],
            'b': ['a', 'd', 'e'],
            'c': ['a', 'e'],
            'd': ['b', 'e', 'f'],
            'e': ['d', 'f'],
            'f': ['d', 'e']
            }

graph=Graph(customDict)
print(graph.dfs('a'))


# ------------------------------------------------------

# 54: Capacity To Ship Packages Within D Days

'''A conveyor belt has packages that must be shipped from one port to another within  D days days.
The ith package on the conveyor belt has a weight of weights[i].
Each day, we load the ship with packages on the conveyor belt (in the order given by weights).
We may not load more weight than the maximum weight capacity of the ship.
Return the least weight capacity of the ship that will result in all
the packages on the conveyor belt being shipped within days days.'''

'''
Example 1:

Input: weights = [1,2,3,4,5,6,7,8,9,10], days = 5
Output: 15
Explanation: A ship capacity of 15 is the minimum to ship all the packages in 5 days like this:
1st day: 1, 2, 3, 4, 5
2nd day: 6, 7
3rd day: 8
4th day: 9
5th day: 10

Note that the cargo must be shipped in the order given, so using a ship of capacity 14
and splitting the packages into parts like (2, 3, 4, 5), (1, 6, 7), (8), (9), (10) is not allowed.'''



'''
Example 2:

Input: weights = [3,2,2,4,1,4], days = 3
Output: 6
Explanation: A ship capacity of 6 is the minimum to ship all the packages in 3 days like this:
1st day: 3, 2
2nd day: 2, 4
3rd day: 1, 4
'''
# link: https://www.youtube.com/watch?v=4lK5pdSXhCk

def shipWithinDays(self, weights: List[int], days: int):
    l=max(weights)
    r=sum(weights)

    while l < r:
        mid=(l+r)//2

        if self.can_ship(mid, weights, days):
            r=mid
        else:
            l=mid+1
        return r
def can_ship(self, candidate, weights, days):
    cur_weight=0
    days_taken=1

    for weight in weights:
        cur_weight += weight

        if cur_weight > candidate:
            days_taken=1
            cur_weight=weight

    return days_taken <= days

List=[3, 2, 2, 4, 1, 4]
days=3
print(shipWithinDays(List, days))


# 55: Minimize the maximum difference between the heights of tower either by increasing or decreasing with k (only once)

'''Input  : arr[] = {1, 15, 10}, k = 6
Output :  Maximum difference is 5.
Explanation : We change 1 to 7, 15 to
9 and 10 to 4. Maximum difference is 5
(between 4 and 9). We can't get a lower
difference.'''

# User function Template
def getMinDiff(arr, n, k):
    arr.sort()
    ans=arr[n - 1] - arr[0]  # Maximum possible height difference

    tempmin=arr[0]
    tempmax=arr[n - 1]

    for i in range(1, n):
        tempmin=min(arr[0] + k, arr[i] - k)

        # Minimum element when we
        # add k to whole array
        # Maximum element when we
        tempmax=max(arr[i - 1] + k, arr[n - 1] - k)

        # subtract k from whole array
        ans=min(ans, tempmax - tempmin)

    return ans

# Driver Code Starts
k=6
n=6
arr=[7, 4, 8, 8, 8, 9]
ans=getMinDiff(arr, n, k)
print(ans)

# -----------------------

finding the minimum differences among the given array

def FoodDiff(arr, N):


# -----------------------------------------------------------

# 56: finding missing number

# 1st: method
def findmissing(arr):
    n=len(arr)+1
    # n=arr.length
    intended_sum=n*(n+1)/2
    actualsum=0

    for number in arr:
        actualsum=actualsum+number
    return intended_sum-actualsum, actualsum

# arr=[1,2,4,5,6,7,8,9,10]
# arr=[1,2,3,4,6]
arr=[2, 4, 8]

print(findmissing(arr))

# 2nd: Method

from typing import List

class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        currentsum=sum(nums)
        n=len(nums)
        intendedSum=n*(n+1)/2

        return int(intendedSum - currentsum)

s=Solution()
answer=s.missingNumber([0, 3, 1, 2, 5])
print(answer)

# ---------

# 57: Count the total prime number less than a given number N

def Countprime(n):
    if n < 2:
        return 0
    count=0
    for i in range(2, n):
        isPrime=True
        for j in range(2, i):
            if (i % j == 0):
                isPrime=False
                break
        if (isPrime == True):
            count += 1
    return count

print(Countprime(5))

# 58: Finding the maximum sum of the 'k' consequtive elements

def Max_Sum_arr(arr, k):
    max_sum=float('-inf')         # Negative infinity
    n=len(arr)

    for i in range(n-k+1):
        current_sum=0
        for j in range(k):
            current_sum += arr[i+j]
        max_sum=max(max_sum, current_sum)
    return max_sum

arr=[80-50, 90, 100]
k=2
print(Max_Sum_arr(arr, k))


# 59: Boats to save people

def NumberBoats(people, limit):
    people.sort()
    heavyP=len(people)-1
    lightP=0

    boats=0

    while (heavyP >= lightP):
        if (people[heavyP] + people[lightP] <= limit):
            boats += 1
            heavyP -= 1
            lightP += 1

    return boats

people=[1, 3, 2, 2]
limit=4

print(NumberBoats(people, limit))


# 60: Containers with most waters (using towers heigh)

def Maxarea(heights):
    max_area=0
    n=len(heights)-1

    for p1 in range(n+1):
        for p2 in range(n+1):
            lenght=min(heights[p1], heights[p2])
            width=p2-p1
            area=lenght*width
            max_area=max(max_area, area)
    return max_area

heights=[5, 9, 2, 4]
print(Maxarea(heights))

# 61: Finding single number (only once time occurence)

def singlenumber(arr):
    m={}
    for num in arr:
        m[num] += 1
    for key in m:
        if (m[key] == 1):
            return key, m

arr=[1, 1, 3, 3, 9]
print(singlenumber(arr))

# 62: Summing multiple values of each key in a dict?

inpt={'item1': [1, 2, 3, 4, 5, 6],
         'item2': [2, 3, 1],
         'item3': [1, 2, 3, 4, 5, 6, 8, 9, 10],
         'item4': [2, 3, 1, 5, 6, 8],
         'item5': [1, 2, 3],
         'item6': [2, 3, 11, 2, 3]
       }
out1={k: [sum(inpt[k])] for k in inpt.keys()}
print(out1)

# ----------

out2={}

for k in inpt.keys():
    out2[k]=[sum(inpt[k])]
print(out2)

# ---------------

# 63:   Two digit sum numbers to get a target value

# 1st: method

def twosum(numbers, target):
    n=len(numbers)
    for i in range(n):
        for j in range(i+1, n):
            if (numbers[i]+numbers[j] == target):
                return numbers[i], numbers[j]
    return -1, -1

numbers=[2, 11, 7, 15]
target=26

print(twosum(numbers, target))


# 2nd: method

def twosum(numbers, target):
    n=len(numbers)
    for i in range(n):
        valueTofind=target - numbers[i]
        for j in range(i+1, n):
            if (numbers[j] == valueTofind):
                return [numbers[i], numbers[j]]
    return -1, -1
numbers=[2, 11, 7, 15]
target=26

print(twosum(numbers, target))


 # 3rd: method: hashmap

def twosum(numbers, target):
    m={}
    n=len(numbers)
    for i in range(n):
        valueTofind=target - numbers[i]
        if (valueTofind in m):
            return [m[valueTofind], numbers[i]]
        else:
            m[numbers[i]]=numbers[i]


numbers=[2, 11, 7, 15]
target=26

print(twosum(numbers, target))



# 64: Finding duplicate value using hashmap


from collections import defaultdict
from typing import List

class Solution:
    def containDuplicate(self, nums: List[int]) -> bool:
        m=defaultdict(int)
        for num in nums:
            if (m[num]):
                return True
            m[num] += 1
        return False


s=Solution()
answer=s.containDupicate([2, 2, 1, 3])
print(answer)

# 65: Mejority of the element in an array


from typing import List

class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        n=len(nums)
        m={}
        for num in nums:
            # m[num]+=1
            m[num]=m.get(num, 0)+1

        for num in nums:
            if (m[num] > n//2):
                return num

s=Solution()
answer=s.majorityElement([22, 22, 22, 33])
print(answer)


# 66: Group Anangrams using Hashmap

# 1st: Method
from typing import List

# class Solution:
#     def findHash( self, s: str)-> List[str]:
#         m= {}
#         for i in range(len(s)):
#             if (s[i] in m):
#                 m[s[i]]+=1
#             else:
#                 m[s[i]]=1
#         return m

#     def groupAnagrams( self, strs: List[str])-> List[List[str]]:
#         m= {}
#         for s in strs:
#             key= self.findHash(s)
#             if (key in m):
#                 m[key].append(s)
#             else:
#                 m[key]=[s]
#         return m.values()

# s = Solution()
# answer = s.groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
# print(answer)

# ----------------------------------
# 2nd: Method

from typing import List

class Solution:
    def findHash(self, s):
        return ''.join(sorted(s))

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        answers=[]
        m={}

        for s in strs:
            hashed=self.findHash(s)
            if (hashed not in m):
                m[hashed]=[]
            m[hashed].append(s)

        for p in m.values():
            answers.append(p)

        return answers

s=Solution()
answer=s.groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
print(answer)


# ------------------------------------------------


# 67: Recursively sum of the digits

def digit_sum(n):
    if n == 0:
        return 0
    else:
        return ((n % 10) + digit_sum(n//10))

print(digit_sum(1234))


# 68: Recursively fabonacci numbers

def febonacci_num(n):
    if n == 0 or n == 1:
        return n
    else:
        return febonacci_num(n-1)+febonacci_num(n-2)

print(febonacci_num(7))


# 69: Recursively factorial numbers

def fact_num(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n*fact_num(n-1)

print(fact_num(5))

# 70: Recursively powers number calculation

def calculate_powers(a, b):
    if b == 1:
        return a
    else:
        return a*calculate_powers(a, b-1)

print(calculate_powers(2, 3))


# 71: Recursively find greatest commom diviser (GCD)

def GCD(a, b):
    if b == 0:
        return a
    else:
        return GCD(b, a % b)

print(GCD(12, 8))

# 72: Recursively find palindrom

def is_palindrome_str(string):
    if len(string) <= 1:
        return True
    elif string[0] != string[-1]:
        return False
    else:
        return is_palindrome_str(string[1:-1])

print(is_palindrome_str('anna'))

# 73: Recursively print the star pattern

def star_pattern(n):
    if n == 1:
        print("*")
    else:
        print('*'*n)
        star_pattern(n-1)


print(star_pattern(6))

# 74: Recursively solving Binary Search problem.

def Binary_search(arr, low, high, element):
    # low=0
    # high= len(arr)
    if low > high:
        return -1
    else:
        middle=(low+high)//2
        if element == arr[middle]:
            return middle
        elif element < arr[middle]:
            return Binary_search(arr, low, middle-1, element)
        else:
            return Binary_search(arr, middle+1, high, element)


my_list=[1, 2, 3, 4, 5, 6, 7]
print('The index position', Binary_search(my_list, 0, len(my_list), 6))


# 75: Python Calender of months

import calendar

year=int(input("Enter the year (with this formatt yyyyy):"))
month=int(input("Enter the month (with this formatt 0-12):"))

print(calendar.month(year, month))

# ------

# 76: Stock maximum profit

'''(1) Iterate through each number in the list.
(2) At the ith index, get the i+1 index price and check if it is larger than the ith index price.
(3) If so, set buy_price = i and sell_price = i+1. Then calculate the profit: sell_price - buy_price.
(4) If a stock price is found that is cheaper than the current buy_price, set this to be the new buying price and continue from step 2.
(5) Otherwise, continue changing only the sell_price and keep buy_price set.'''


# 1st method
# x = [45, 24, 35, 31, 40, 38, 11]
x=[10, 12, 4, 5, 9]


def stockMax(x):
    diff_arr=[]
    for i in range(len(x)):
        y=x[i+1:]
        for j in y:
            if x[i] < j:
                z=j - x[i]
                diff_arr.append(z)
    if (diff_arr == []):
        return -1
    else:
        return max(diff_arr)
print(stockMax(x))

# 2nd Method


# Function to return the maximum profit
# that can be made after buying and
# selling the given stocks


def maxProfit(price, start, end):

    # If the stocks can't be bought
    if (end <= start):
        return 0;

    # Initialise the profit
    profit=[];

    # The day at which the stock
    # must be bought
    for i in range(start, end, 1):
        # The day at which the
        # stock must be sold
        for j in range(i+1, end+1):

            # If buying the stock at ith day and
            # selling it at jth day is profitable
            if (price[j] > price[i]):
                z=price[j]-price[i]
                profit.append(z)
    # Update the maximum profit so far
    if (profit == []):
        return -1
    else:
        return max(profit)


# Driver code
if __name__ == '__main__':
    # price = [100, 180, 260, 310, 40, 535, 695];
    price=[10, 12, 4, 5, 9];
    n=len(price);

    print(maxProfit(price, 0, n - 1));

####-------
Arrays & Linked Lists
Question 1: Reverse an Array

Problem: Given an array, reverse the elements of the array.
Solution:
python
Copy code
def reverse_array(arr):
    return arr[::-1]

# Example usage
arr = [1, 2, 3, 4, 5]
reversed_arr = reverse_array(arr)
print(reversed_arr)  # Output: [5, 4, 3, 2, 1]
Question 2: Insert a node in a Linked List

Problem: Given a singly linked list and a value, insert a new node with the given value at the end of the list.
Solution:
python
Copy code
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def insert_node(head, val):
    new_node = ListNode(val)
    if not head:
        return new_node
    current = head
    while current.next:
        current = current.next
    current.next = new_node
    return head

# Example usage
head = ListNode(1, ListNode(2, ListNode(3)))
new_head = insert_node(head, 4)
current = new_head
while current:
    print(current.val, end=" -> ")
    current = current.next
# Output: 1 -> 2 -> 3 -> 4 ->
Sorting Algorithms
Question 1: Implement Merge Sort

Problem: Implement the Merge Sort algorithm.
Solution:
python
Copy code
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Example usage
arr = [5, 2, 4, 1, 3]
sorted_arr = merge_sort(arr)
print(sorted_arr)  # Output: [1, 2, 3, 4, 5]
Question 2: Implement Quick Sort

Problem: Implement the Quick Sort algorithm.
Solution:
python
Copy code
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

# Example usage
arr = [5, 2, 4, 1, 3]
sorted_arr = quick_sort(arr)
print(sorted_arr)  # Output: [1, 2, 3, 4, 5]
Searching Algorithms
Question 1: Binary Search

Problem: Implement the Binary Search algorithm.
Solution:
python
Copy code
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Example usage
arr = [1, 2, 3, 4, 5]
index = binary_search(arr, 3)
print(index)  # Output: 2
Question 2: Implement a Hash Table with Separate Chaining

Problem: Implement a basic hash table with separate chaining for collision resolution.
Solution:
python
Copy code
class HashTable:
    def __init__(self, size):
        self.size = size
        self.table = [[] for _ in range(size)]

    def hash_function(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        index = self.hash_function(key)
        for pair in self.table[index]:
            if pair[0] == key:
                pair[1] = value
                return
        self.table[index].append([key, value])

    def search(self, key):
        index = self.hash_function(key)
        for pair in self.table[index]:
            if pair[0] == key:
                return pair[1]
        return None

# Example usage
ht = HashTable(10)
ht.insert("apple", 1)
ht.insert("banana", 2)
print(ht.search("apple"))  # Output: 1
print(ht.search("banana"))  # Output: 2
Recursion
Question 1: Factorial

Problem: Implement a function to calculate the factorial of a number using recursion.
Solution:
python
Copy code
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

# Example usage
print(factorial(5))  # Output: 120
Question 2: Fibonacci Sequence

Problem: Implement a function to calculate the nth Fibonacci number using recursion.
Solution:
python
Copy code
def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# Example usage
print(fibonacci(5))  # Output: 5
Graphs
Question 1: Depth-First Search (DFS)

Problem: Implement DFS for a graph.
Solution:
python
Copy code
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    print(start, end=" ")
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

# Example usage
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}
dfs(graph, 'A')  # Output: A B D E F C
Question 2: Breadth-First Search (BFS)

Problem: Implement BFS for a graph.
Solution:
python
Copy code
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    while queue:
        vertex = queue.popleft()
        print(vertex, end=" ")
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

# Example usage
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}
bfs(graph, 'A')  # Output: A B C D E F
Trees
Question 1: Inorder Traversal of a Binary Tree

Problem: Implement inorder traversal for a binary tree.
Solution:
python
Copy code
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def inorder_traversal(root):
    if root:
        inorder_traversal(root.left)
        print(root.val, end=" ")
        inorder_traversal(root.right)

# Example usage
root = TreeNode(1, TreeNode(2), TreeNode(3))
inorder_traversal(root)  # Output: 2 1 3
Question 2: Check if a Tree is a Binary Search Tree (BST)

Problem: Implement a function to check if a binary tree is a BST.
Solution:
python
Copy code
def is_bst(root, left=float('-inf'), right=float('inf')):
    if not root:
        return True
    if not (left < root.val < right):
        return False
    return is_bst(root.left, left, root.val) and is_bst(root.right, root.val, right)

# Example usage
root = TreeNode(2, TreeNode(1), TreeNode(3))
print(is_bst(root))  # Output: True
Hash Tables
Question 1: Implement a Simple Hash Table

Problem: Implement a simple hash table with basic operations.
Solution:
python
Copy code
class SimpleHashTable:
    def __init__(self, size):
        self.size = size
        self.table = [None] * size

    def hash_function(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        index = self.hash_function(key)
        self.table[index] = value

    def get(self, key):
        index = self.hash_function(key)
        return self.table[index]

# Example usage
ht = SimpleHashTable(10)
ht.insert("apple", 1)
ht.insert("banana", 2)
print(ht.get("apple"))  # Output: 1
print(ht.get("banana"))  # Output: 2
Question 2: Handle Collisions with Linear Probing

Problem: Implement a hash table that handles collisions using linear probing.
Solution:
python
Copy code
class LinearProbingHashTable:
    def __init__(self, size):
        self.size = size
        self.table = [None] * size

    def hash_function(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        index = self.hash_function(key)
        while self.table[index] is not None:
            index = (index + 1) % self.size
        self.table[index] = (key, value)

    def get(self, key):
        index = self.hash_function(key)
        while self.table[index] is not None:
            if self.table[index][0] == key:
                return self.table[index][1]
            index = (index + 1) % self.size
        return None

# Example usage
ht = LinearProbingHashTable(10)
ht.insert("apple", 1)
ht.insert("banana", 2)
print(ht.get("apple"))  # Output: 1
print(ht.get("banana"))  # Output: 2
Dynamic Programming
Question 1: Fibonacci Sequence with Dynamic Programming

Problem: Implement a function to calculate the nth Fibonacci number using dynamic programming.
Solution:
python
Copy code
def fibonacci_dp(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    fib = [0] * (n + 1)
    fib[1] = 1
    for i in range(2, n + 1):
        fib[i] = fib[i - 1] + fib[i - 2]
    return fib[n]

# Example usage
print(fibonacci_dp(5))  # Output: 5
Question 2: Longest Increasing Subsequence

Problem: Implement a function to find the length of the longest increasing subsequence in an array.
Solution:
python
Copy code
def longest_increasing_subsequence(arr):
    if not arr:
        return 0
    dp = [1] * len(arr)
    for i in range(1, len(arr)):
        for j in range(i):
            if arr[i] > arr[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

# Example usage
arr = [10, 9, 2, 5, 3, 7, 101, 18]
print(longest_increasing_subsequence(arr))  # Output: 4
These questions and solutions cover a range of topics in data structures and algorithms, providing a good mix of theory and practical implementation.