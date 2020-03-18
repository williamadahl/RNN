import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
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
training_data = [0]*len(train_labels)
# training_data = np.zeros(len(train_labels), dtype=np.int) 
print('first training data type:', type(training_data))

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



# Find index of largest sample and the longest data sample of both train and test dataset: 

longest_sample = len(training_data[0])
longest_sample_index = 0

for i in range(len(training_data)):
    tmp = len(training_data[i])
    if tmp > longest_sample:
        longest_sample = tmp
        longest_sample_index = i


# Find longest sample in test data :

longest_sample_test = len(test_data[0])
longest_sample_index_test = 0

for i in range(len(test_data)):
    tmp = len(test_data[i])
    if tmp > longest_sample_index_test:
        longest_sample_index_test = tmp
        longest_sample_index_test = i


# Flag used as boolean 

train_longest = 1 
if longest_sample_test > longest_sample:
    longest_sample = longest_sample_test 
    longest_sample_index = longest_sample_index_test
    train_longest = 0


# Function for finding the longest code line in training data
def find_max_list(list):
    list_len = [len(i) for i in list]
    return max(list_len)


# Chose to take the longest line in the largest sample, this does not guarantie longest line, but we add some to it and pad the rest 
if train_longest:
    print('train_longest')   
    max_seq = find_max_list(training_data[longest_sample_index]) + 2
else:
    print('test longest')
    max_seq = find_max_list(test_data[longest_sample_index]) + 2



# Function for padding shorter lines of code to match the longest 
def pad_seq_len(data, seq_len):
    features = np.zeros((len(data), seq_len),dtype=int)
    for ii, review in enumerate(data):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

# Function for padding shorter samples to create a uniform 3D matrix of our data 
def pad_sample_len(data, longest_sample):
    diff = longest_sample - len(data)
    padding = [[0]*2]
    for i in range(diff):
        data.extend(padding)
    return data



for i in range(len(training_data)):
    training_data[i] = pad_sample_len(training_data[i], longest_sample)
    training_data[i] = pad_seq_len(training_data[i],max_seq)

for i in range(len(test_data)):
    test_data[i] = pad_sample_len(test_data[i], longest_sample)
    test_data[i] = pad_seq_len(test_data[i],max_seq)



# Convert data and labels into numpy arrays 
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
training_data = np.asarray(training_data)
test_data = np.asarray(test_data)

# We need a dataset for validation during training, Chose to spilt in half, can adjust this later 
split_frac = 0.5
split_id = int(split_frac * len(test_data))
validation_data, test_data = test_data[:split_id], test_data[split_id:]
val_labels, test_labels = test_labels[:split_id], test_labels[split_id:]

print(train_labels)

# Create tensorDatasets for training, validation and testing 

training_data = TensorDataset(torch.from_numpy(training_data), torch.from_numpy(train_labels))
val_data = TensorDataset(torch.from_numpy(validation_data), torch.from_numpy(val_labels))
test_data = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels))

print(len(training_data))
print(len(val_data))
print(len(test_data))

# could chose to use shuffle here, and previously written logic for it 
batch_size = 2
train_loader = DataLoader(training_data, shuffle=False, batch_size=batch_size)
val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)


# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()
# Hardcoded for my personal architecture 
device = torch.device("cpu")
# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    print('gpu')
    # device = torch.device("cuda")
else:
    print('cpu')
    device = torch.device("cpu")

# Now we can set up the architecture 
# Comment on the drop_prob as well 

class SentimentNet(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(SentimentNet, self).__init__()
        # This is set to 1 since binary classification 
        self.output_size = output_size
        # Try two layers first only 
        self.n_layers = n_layers
        # Hidden dimension is 512, but can of course change during hyper parameter optimization 
        self.hidden_dim = hidden_dim
        
        # Set up the embedding lookup table that stores embeddings of a fixed dictionary and size. 
        # vocab_size -- size of the dictionary of embeddings which is the unique number of tokens created in the word2idx (len(word2idx + 1))
        # embedding_dim --  the size of each embedding vector (400)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Set up multi layer LSTM RNN for the input sequence 
        # embedding_dim --  The number of expected features in the input x (400)
        # hidden_dim -- The number of features in the hidden state h (512)
        # n_layers -- Number of recurrent layers (2)
        # dropout --  Dropout layer on the outputs of each LSTM layer except the last layer (0.5) can optimize this also 
        # batch_first -- input and output tensors are provided as (batch, seq, feature). Migth try without this 
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        # During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution
        self.dropout = nn.Dropout(drop_prob)
        # Applies a linear transformation to the incoming data with inputsize hidden_dim (2), outputsize (1) 
        self.fc = nn.Linear(hidden_dim, output_size)
        # Applies the element-wise function sigmoid 
        self.sigmoid = nn.Sigmoid()

    # Forward pass of the training defined here  
    # x is the full tensor sendt into training ([2, 409, 11])
    # hidden is the initialized hidden dimention with zero values should be the same as x      
    def forward(self, x, hidden):
        batch_size = x.size(0)
        # Transform to unlimited precision 
        x = x.long()
        embeds = self.embedding(x)
        # problem here with the 3D vs my 4D dimension 
        lstm_out, hidden = self.lstm(embeds, hidden)
        print('in forward here is x')
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)
        
        out = out.view(batch_size, -1)
        out = out[:,-1]
        return out, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden


vocab_size = len(word2idx) + 1
print('vocab size ', vocab_size)
output_size = 1
embedding_dim = 400
hidden_dim = 512
n_layers = 2

model = SentimentNet(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
print(model)


lr=0.005
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epochs = 1
counter = 0
print_every = 10
clip = 5
valid_loss_min = np.Inf

model.train()

# Se each sample is in this 
for i, (images, labels) in enumerate(train_loader):
    print(images[0][0])
    print(len(images[0]))


test_list = []
label_list = []

for i in range(epochs):

    
    h = model.init_hidden(batch_size)    
    for inputs, labels in train_loader:
        counter += 1
        h = tuple([e.data for e in h])
        inputs, labels = inputs.to(device), labels.to(device)
        test_list.append(inputs)
        label_list.append(labels)   



        model.zero_grad()
        output, h = model(inputs, h)
'''
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        if counter%print_every == 0:
            val_h = model.init_hidden(batch_size)
            val_losses = []
            model.eval()
            for inp, lab in val_loader:
                val_h = tuple([each.data for each in val_h])
                inp, lab = inp.to(device), lab.to(device)
                out, val_h = model(inp, val_h)
                val_loss = criterion(out.squeeze(), lab.float())
                val_losses.append(val_loss.item())
                
            model.train()
            print("Epoch: {}/{}...".format(i+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
            if np.mean(val_losses) <= valid_loss_min:
                torch.save(model.state_dict(), './state_dict.pt')
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
                valid_loss_min = np.mean(val_losses)


'''
print('this is a label', label_list[0])
print('this is a input', test_list[0])
print(torch.eq(test_list[0][0], test_list[1][0]))