
import bz2
from collections import Counter
import re
import nltk
import numpy as np
nltk.download('punkt')


test_file = bz2.BZ2File('./Reviews/test.ft.txt.bz2')

test_file = test_file.readlines()

print("Number of test reviews: " + str(len(test_file)))

num_test = 20

test_file = [x.decode('utf-8') for x in test_file[:num_test]]


test_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in test_file]
test_sentences = [x.split(' ', 1)[1][:-1].lower() for x in test_file]

for i in range(len(test_sentences)):
    test_sentences[i] = re.sub('\d','0',test_sentences[i])

print(test_sentences[0])
del test_file

# tokenization 
words = Counter()
for i, sentence in enumerate(test_sentences):
    #The sentences will be stored as a list of words/tokens
    test_sentences[i] = []
    for word in nltk.word_tokenize(sentence): #Tokenizing the words
        words.update([word.lower()]) #Converting all the words to lower case
        test_sentences[i].append(word)

print(test_sentences[0])
print(words)