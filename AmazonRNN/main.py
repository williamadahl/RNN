
import bz2
from collections import Counter
import re
import nltk
import numpy as np
nltk.download('punkt')


test_file = bz2.BZ2File('./Reviews/test.ft.txt.bz2')

test_file = test_file.readlines()
train_file = test_file

print("Number of test reviews: " + str(len(test_file)))

num_test = 20
num_train = 80

test_file = [x.decode('utf-8') for x in test_file[:num_test]]
train_file = [x.decode('utf-8') for x in train_file[num_test:num_train]]

# Train labels and train sentences
train_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in train_file]
train_sentences = [x.split(' ', 1)[1][:-1].lower() for x in train_file]

# Test labels and test sentences 
test_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in test_file]
test_sentences = [x.split(' ', 1)[1][:-1].lower() for x in test_file]



for i in range(len(train_sentences)):
    train_sentences[i] = re.sub('\d','0',train_sentences[i])

for i in range(len(test_sentences)):
    test_sentences[i] = re.sub('\d','0',test_sentences[i])



del test_file, train_file

# tokenization 
words = Counter()
for i, sentence in enumerate(train_sentences):
    #The sentences will be stored as a list of words/tokens
    train_sentences[i] = []
    for word in nltk.word_tokenize(sentence): #Tokenizing the words
        words.update([word.lower()]) #Converting all the words to lower case
        train_sentences[i].append(word)



# Removing the words that only appear once
words = {k:v for k,v in words.items() if v>1}
# Sorting the words according to the number of appearances, with the most common word being first
words = sorted(words, key=words.get, reverse=True)
# Adding padding and unknown to our vocabulary so that they will be assigned an index
# words = ['_PAD','_UNK'] + words
# Dictionaries to store the word to index mappings and vice versa
word2idx = {o:i for i,o in enumerate(words)}
idx2word = {i:o for i,o in enumerate(words)}


print('this is a training sample', train_sentences[0])



for i, sentence in enumerate(train_sentences):
    # Looking up the mapping dictionary and assigning the index to the respective words
    train_sentences[i] = [word2idx[word] if word in word2idx else 1 for word in sentence]
for i, sentence in enumerate(test_sentences):
    # For test sentences, we have to tokenize the sentences as well
    test_sentences[i] = [word2idx[word.lower()] if word.lower() in word2idx else 0 for word in nltk.word_tokenize(sentence)]

print(train_sentences[0])
print(words)