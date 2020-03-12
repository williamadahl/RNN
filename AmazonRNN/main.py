import torch 
from torch.utils.data import TensorDataset, DataLoader
from torch.utils import data 
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





for i, sentence in enumerate(train_sentences):
    # Looking up the mapping dictionary and assigning the index to the respective words
    train_sentences[i] = [word2idx[word] if word in word2idx else 1 for word in sentence]
for i, sentence in enumerate(test_sentences):
    # For test sentences, we have to tokenize the sentences as well
    test_sentences[i] = [word2idx[word.lower()] if word.lower() in word2idx else 0 for word in nltk.word_tokenize(sentence)]


# Defining a function that either shortens sentences or pads sentences with 0 to a fixed length

def pad_input(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

seq_len = 200 #The length that the sentences will be padded/shortened to

train_sentences = pad_input(train_sentences, seq_len)
test_sentences = pad_input(test_sentences, seq_len)

# Converting our labels into numpy arrays
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

split_frac = 0.5
split_id = int(split_frac * len(test_sentences))
val_sentences, test_sentences = test_sentences[:split_id], test_sentences[split_id:]
val_labels, test_labels = test_labels[:split_id], test_labels[split_id:]

print(train_sentences.shape)
print('type of train_sentenece', type(train_sentences))
print('type of train_sentenece[0]', type(train_sentences[0]))
print('this is a training sample', train_sentences)

test = np.asarray([[0 for x in range(len(train_sentences[0]))] for y in range(len(train_sentences))])

print(test.shape)
np.insert(train_sentences)
print(test)
# np.insert(test, 1, train_sentences)
# test = [train_sentences, train_sentences]
test = np.asarray(test)
test2 = [train_labels, train_labels]
test2 = np.asarray(test2)
# test.insert(train_sentences)

print(test.shape)
print(type(test[0]))

test3 = TensorDataset(torch.from_numpy(test), torch.from_numpy(test2))
print(type(test3))

train_data = TensorDataset(torch.from_numpy(train_sentences), torch.from_numpy(train_labels))
val_data = TensorDataset(torch.from_numpy(val_sentences), torch.from_numpy(val_labels))
test_data = TensorDataset(torch.from_numpy(test_sentences), torch.from_numpy(test_labels))