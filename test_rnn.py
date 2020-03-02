import torch
import torch.nn as nn
from collections import Counter
import re
import nltk
import numpy as np


print(torch.__version__)

# Read in all training and testing data 
f = open('../VulnerabilityGenerator/FinalizedData/Train.txt','r')
if f.mode == 'r':
    train_file = f.readlines()
    f.close()

f = open('../VulnerabilityGenerator/FinalizedData/Test.txt','r')
if f.mode == 'r':
    test_file = f.readlines()
    f.close()

# Extract and remove the labels in training set
train_labels = []
for i in range(len(train_file)):
    if train_file[i].startswith('__label0__'):
        train_labels.append(0)
        train_file[i] = 'main:\n'
    elif train_file[i].startswith('__label1__'):
        train_labels.append(1)
        train_file[i] = 'main:\n'

# Extract and remove the labels in testing set
test_labels = []
for i in range(len(test_file)):
    if test_file[i].startswith('__label0__'):
        test_labels.append(0)
        test_file[i] = 'main:\n'
    elif test_file[i].startswith('__label1__'):
        test_labels.append(1)
        test_file[i] = 'main:\n'







print( train_file[291])
print(train_labels)
print(test_labels)

