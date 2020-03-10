import torch
import torch.nn as nn
from collections import Counter
import re
import nltk
nltk.download('punkt')
import numpy as np


print(torch.__version__)

# Read in all training and testing data 
f = open('../VulnerabilityGenerator/FinalizedData/Train.txt','r')
# f = open('../../Master/FinalizedData/Train.txt', 'r')
if f.mode == 'r':
    train_file = f.readlines()
    f.close()

f = open('../VulnerabilityGenerator/FinalizedData/Test.txt','r')
# f = open('../../Master/FinalizedData/Test.txt', 'r')
if f.mode == 'r':
    test_file = f.readlines()
    f.close()

# Extract and remove the labels in training set
train_labels = []
main_indexes_train = [] # list to store the main: indexes, aka each new data sample 

for i in range(len(train_file)):
    if train_file[i].startswith('__label0__'):
        train_labels.append(0)
        train_file[i] = 'main :\n'
        main_indexes_train.append(i)
    elif train_file[i].startswith('__label1__'):
        train_labels.append(1)
        train_file[i] = 'main :\n'
        main_indexes_train.append(i)

print(main_indexes_train)

# Extract and remove the labels in testing set
test_labels = []
main_indexes_test = [] # list to store the main: indexes, aka each new data sample 

for i in range(len(test_file)):
    if test_file[i].startswith('__label0__'):
        test_labels.append(0)
        test_file[i] = 'main :\n'
        main_indexes_test.append(i)
    elif test_file[i].startswith('__label1__'):
        test_labels.append(1)
        test_file[i] = 'main :\n'
        main_indexes_test.append(i)

print(main_indexes_test)

# Create dictionary to map all words to the number of times it occurs in all training sencences
# This is the tokenization 
words = Counter()


for i, line in enumerate(train_file):
    for word in nltk.word_tokenize(line):
        words.update([word])
        
# Store all data in 3D array training_data [ [sample1 [sentence1 ],[sentence2]...]...]
training_data = [0]* len(train_labels) 

for x in range(len(training_data)):
    # create a new list which will a singel sample 
    if x < (len(training_data)-1):
        training_sample = [0]*(main_indexes_train[x+1]-main_indexes_train[x]-1)
    else: 
        training_sample = [0]*(len(train_file)-main_indexes_train[x]-1)

    # loop over all lines which are together with a sample and append it to 
    # sample list 
    for j in range(len(training_sample)):
        # normal case last sample 
        if x < len(training_data)-1:
            ind = 0
            for line in train_file[main_indexes_train[x]:main_indexes_train[x+1]]:
                if line != '\n':
                    training_sample[ind] = line.split()
                ind += 1
        # edge case last sample 
        else:
            ind = 0
            for line in train_file[main_indexes_train[x]:len(train_file)]:
                if line != '\n':
                    training_sample[ind] = line.split()
                ind += 1
    training_data[x] = training_sample


# Store all data in 3D array training_data [ [sample1 [sentence1 ],[sentence2]...]...]
test_data = [0]* len(test_labels) 

for x in range(len(test_data)):
    # create a new list which will a singel sample 
    if x < (len(test_data)-1):
        test_sample = [0]*(main_indexes_test[x+1]-main_indexes_test[x]-1)
    else: 
        test_sample = [0]*(len(test_file)-main_indexes_test[x]-1)

    # loop over all lines which are together with a sample and append it to 
    # sample list 
    for j in range(len(test_sample)):
        # normal case last sample 
        if x < len(test_data)-1:
            ind = 0
            for line in test_file[main_indexes_test[x]:main_indexes_test[x+1]]:
                if line != '\n':
                    test_sample[ind] = line.split()
                ind += 1
        # edge case last sample 
        else:
            ind = 0
            for line in test_file[main_indexes_test[x]:len(test_file)]:
                if line != '\n':
                    test_sample[ind] = line.split()
                ind += 1
    test_data[x] = test_sample


del test_file, train_file

print('this is training data\n', test_data[0])

# Sorting the words according to the number of appearances, with the most common word being first
words = sorted(words, key=words.get, reverse=True)
words = ['_PAD','_UNK'] + words
word2idx = {o:i for i,o in enumerate(words)}
idx2word = {i:o for i,o in enumerate(words)}


# Looking up the mapping dictionary and assigning the index to the respective words
for i in range(len(training_data)):
    for j, sentence in enumerate(training_data[i]):
        # Looking up the mapping dictionary and assigning the index to the respective words
        training_data[i][j] = [word2idx[word] if word in word2idx else 1 for word in sentence]


for i in range(len(test_data)):
    for j, sentence in enumerate(test_data[i]):
        # For test sentences, we have to tokenize the sentences as well
        # Trying to use '_UNIK' for unseen words, can change this at a later point 
        test_data[i][j] = [word2idx[word] if word in word2idx else 0 for word in sentence]



# Find index of largest sample: 
longest_sample = len(training_data[i])
longest_sample_index = 0

for i in range(len(training_data)):
    tmp = len(training_data[i])
    if tmp > longest_sample:
        longest_sample = tmp
        longest_sample_index = i

# Function for finding the longest code line in training data
def find_max_list(list):
    list_len = [len(i) for i in list]
    return max(list_len)

# Chose to take the longest line in the largest sample, this does not guarantie longest line, but we add some to it and pad the rest 
max_seq = find_max_list(training_data[longest_sample_index]) + 2
print('this is the max seq_len', max_seq)

# Function for padding shorter lines of code to match the longest 

'''
def pad_input(data, seq_len):
    features = np.zeros((len(data), seq_len),dtype=int)
    for ii, review in enumerate(data):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features



print('this is training data\n', test_data[0])
print(word2idx)

'''
print(train_labels)
print(test_labels)

