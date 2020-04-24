import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import time

from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile
import gensim.downloader as api


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 64
MAX_LENGTH = 250
EPOCHS = 1
EMBEDDING_DIM = 300
HIDDEN_DIM = 128
LR = 1e-4

'''
    UTILS
'''

def get_vocab(data):
    labels = list(set(data["class"].values))
    unique_sents = list(set(data["sentence"].values))
    labels_to_idx = {t: i for i, t in enumerate(labels)}
    sents_to_idx = {t: i for i, t in enumerate(unique_sents)}
    all_words = (" ".join(data["sentence"].values)).split()
    word_count = Counter(all_words)
    sorted_words = word_count.most_common(len(word_count))
    words_to_idx = {k: i+1 for i, (k, v) in enumerate(sorted_words)}

    return sents_to_idx, labels_to_idx, words_to_idx

def seq_to_idx(seq, to_idx):
    return [to_idx[w] for w in seq]
   

def encode_sents(all_sents, words_to_idx):
    encoded_sents = []
    for sent in all_sents:
        encoded_sent = []
        for word in sent.split(" "):
            if word not in words_to_idx.keys():
                encoded_sent.append(0)
            else:
                encoded_sent.append(words_to_idx[word])
        encoded_sents.append(encoded_sent)
    return encoded_sents

def pad_sentence(sentence, labels_to_idx, max_length):
    padded_sentence = []
    padded_sentence.extend([0 for i in range(max_length)])

    if len(sentence) > max_length:
        padded_sentence[:] = sentence[:max_length]
    else:
        padded_sentence[:len(sentence)] = sentence
    return padded_sentence

def tokanize_corpus(corpus):
    tokens = [x.split() for x in corpus]
    return tokens

def clean_dataset(data):
    # remove undesirable "words"
    data["sentence"] = data["sentence"].str.lower()    
    data["sentence"] = data["sentence"].replace("[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+", "", regex=True)
    data["sentence"] = data["sentence"].replace("((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}", "", regex=True)
    #data["sentence"] = data["sentence"].replace("!@#$%^&*()[]{};:,./<>?\|`~-=_+", "") # remove special chars
    data["sentence"] = data["sentence"].replace("\d", "", regex=True)
    return data
    
def create_dataloader(path):
    raw_data = pd.read_csv(path, sep=",", header=None)
    # assume i have what i need later
    raw_data.columns = ["sentence", "class"]
    raw_data = raw_data[1:]
    # preprocess the dataset
    clean_data = clean_dataset(raw_data)

    # get vocab
    sents_to_idx, labels_to_idx, words_to_idx = get_vocab(clean_data)

    # clean data and encode
    clean_data_list = clean_data["sentence"].to_list()
    encoded_data = encode_sents(clean_data_list, words_to_idx)

    labels = clean_data["class"].to_list()
    labels = seq_to_idx(labels, labels_to_idx)

    # shuffle the data set here so we dont have to when creating the data loaders
    encoded_data, labels = shuffle(encoded_data, labels)

    longest_sentence = ""

    # seg fault
    #for sentence in clean_data_list:
    #    if len(sentence) > len(longest_sentence):
    #        longest_sentence = sentence

    #max_length = len(longest_sentence)

    # pad all our entries
    padded_data_list = [pad_sentence(i, labels_to_idx, MAX_LENGTH) for i in encoded_data]

    attention_masks = [[float(i>0) for i in j] for j in padded_data_list]

    # split data into train, validation and test
    train_data, test_data, train_labels, test_labels = train_test_split(padded_data_list, labels, test_size=0.15, shuffle=False)
    td_copy = train_data
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.176, shuffle=False)

    train_masks, test_masks, _, _ = train_test_split(attention_masks, padded_data_list, test_size=0.15, shuffle=False)
    train_masks, val_masks, _, _ = train_test_split(train_masks, td_copy, test_size=0.176, shuffle=False)

    train_dataset = TensorDataset(torch.tensor(train_data), torch.tensor(train_masks), torch.FloatTensor(train_labels))
    #train_sampler = RandomSampler(torch.tensor(train_data))
    valid_dataset = TensorDataset(torch.tensor(val_data), torch.tensor(val_masks), torch.FloatTensor(val_labels))
    #valid_sampler = later
    test_dataset = TensorDataset(torch.tensor(test_data), torch.tensor(test_masks), torch.FloatTensor(test_labels))

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    return train_dataloader, valid_dataloader, test_dataloader, len(words_to_idx)+1

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)


class Net(nn.Module):
    def __init__(self, embedding_size=EMBEDDING_DIM, hidden_size=HIDDEN_DIM, vocab_size=100, target_size=1, pretrained=None):
        super(Net, self).__init__()

        # layer definitions
        if pretrained is not None:
           self.weights = torch.FloatTensor(pretrained.wv.vectors)
           self.embedding_layer = nn.Embedding.from_pretrained(self.weights)
           self.weights.require_grad = False    # freeze

        #else:
        self.embedding_layer2 = nn.Embedding(vocab_size, embedding_size)
        self.recurrent_layer = nn.LSTM(embedding_size, hidden_size, 1, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * 2, 16)
        self.fc2 = nn.Linear(16, target_size)
        self.sigmoid = nn.Sigmoid()

        # loss function
        # self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = nn.BCELoss()  # same as tosin
        self.hidden_size = hidden_size
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        '''
        1. input2tensor_layer
        2. embedding_layer
        3. recurrent_layer
        4. lambda_layer
        5. dense_layer
        '''
        embeds = self.embedding_layer(inputs)
        out, hidden = self.recurrent_layer(embeds)

        # stack up the lstm output (fixes memory issues and matches the size)
        out = out.contiguous().view(-1, self.hidden_size * 2)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(inputs.size(), -1)
        out = out[:, -1]
        return out

    def _train(self, train_loader, val_loader, test_loader, optimizer):
        start_time = time.time()
        for epoch in range(EPOCHS):
            self.train()
            train_loss, train_steps = 0.0, 0.0
            for i, (inputs, masks, labels) in enumerate(train_loader):
                print("Training batch {}/{}...".format(i, len(train_loader)))
                self.zero_grad()
                inputs = inputs.to(DEVICE)
                masks = masks.to(DEVICE)
                labels = labels.to(DEVICE)
                scores = self(inputs)
 
                loss = self.loss_fn(scores, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_steps += 1
            print("Train loss: {}".format(train_loss/train_steps))

            self.eval()
            eval_loss, eval_acc, eval_steps = 0.0, 0.0, 0.0
            for i, (inputs, masks, labels) in enumerate(val_loader):
                print("Val batch {}/{}...".format(i, len(val_loader)))
                with torch.no_grad():
                    inputs = inputs.to(DEVICE)
                    masks = masks.to(DEVICE)
                    labels = labels.to(DEVICE)

                    scores = self(inputs)
                    loss = self.loss_fn(scores, labels)
                    eval_loss += loss.item()
                    eval_steps += 1
            print("Train loss: {}".format(eval_loss/eval_steps))
            print("Time elapsed: {}".format(time.time() - start_time))
        print("Training done..")


if __name__ == "__main__":
    print("device to use: {}".format(DEVICE))
    path = get_tmpfile("models/word2vec.model")
    word2vec = Word2Vec(corpus, size=300)
   
    train_loader, val_loader, test_loader, vocab_size = create_dataloader("data/train.csv")

    # init the network
    model = Net(vocab_size=vocab_size, pretrained=word2vec)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # train the network
    model._train(train_loader, val_loader, test_loader, optimizer)
