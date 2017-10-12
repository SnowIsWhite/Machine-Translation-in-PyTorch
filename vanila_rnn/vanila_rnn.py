"""seq2seq neural machine translation with RNN.

I referred to PyTorch Tutorial http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html.
Aspects:
- GRU unit
- reversed input sequence
"""

import re
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
sys.path.append('../data/')
sys.path.append('../')
from preprocess import *
from utils import *

# initial variables
SOS_token = 0
EOS_token = 1
UNK_token = 2
batch_size = 1 #sentence

# load data
def load_data(data_dir = './data/kor-eng/kor.txt'):
    data = Preprocess('vanila-nmt')
    input_lang, output_lang, pair = prepare_data(data_dir)
    return input_lang, output_lang, pair

# change textdata into vectors of indicies
def phrase_to_index(lang, phrase):
    return [lang.word2index[word] for word in phrase.split()]

def phrase_to_variable(lang, phrase):
    indicies = phrase_to_index(lang, phrase)
    indicies.append(EOS_token)
    var = Variable(torch.LongTensor(indicies).view(-1,1))
    return var

def get_word_vectors(input_lang, ouput_lang, pair):
    input_vec = [phrase_to_variable(input_lang, p[0]) for p in pair]
    output_vec = [phrase_to_variable(output_lang, p[1]) for p in pair]
    return input_vec, output_vec

class SimpleEncoder(nn.Module):
    def __init__(self, input_size, embedding_size, batch, hidden_size, n_layer = 1):
        super(SimpleEncoder, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.batch = batch

        #nn.Embedding: input (N, W) N- minibatch W- number of inidicies to extract per minibatch
        #output: (N, W, embedding_dim)
        #params: num_embeddings: size of dictionary of embeddings,
        #embedding_dim : size of each embedding vector
        self.embedding = nn.Embedding(input_size, embedding_size)

        #RNN: Input: (input, h0)
        #input: seq_len, batch, input_size
        #h0: num_layers * num_directions, batch, hidden_size
        #output: output, h_n
        #output: seq_len, batch, hidden_size * num_directions-> not used in encoder
        #h_n : num_layers * num_directions, batch, hidden_size
        self.gru = nn.GRU(embedding_size, hidden_size, n_layer)

    def forward(self, word_inputs, hidden):
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, self.batch, -1)
        output, hidden = self.gru(embedded, hidden)
        # only use hidden
        return output, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layer, self.batch, self.hidden_size))
        return hidden

class SimpleDecoder(nn.Module):
    def __init__(self, target_size, embedding_size, batch, hidden_size, n_layer = 1):
        super(SimpleDecoder, self).__init__()
        self.target_size = target_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.batch = batch

        self.embedding = nn.Embedding(target_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layer)
        #nn.Linear = params: in_faeture, out_feature
        #input:  (N,*,in_features) where * means any number of additional dimensions
        #output: Output: (N,*,out_features)
        self.out = nn.Linear(hidden_size, target_size)
        #nn.LogSoftmax = Input: (N,L) Output: (N,L)
        self.softmax = nn.LogSoftmax()

    def forward(self, word_inputs, prev_hidden, context_vector):
        #word_inputs : output word of decoder in previous time step
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, self.batch, -1)
        #add context vector and prev_hidden state
        hidden = prev_hidden + context_vector
        #apply gru unit
        output, hidden = self.gru(embedded, hidden)
        output = self.softmax(self.out(output.squeeze(0))) # batch, dic_size
        return output, hidden

    def init_hidden(self, context_vector):
        hidden = F.tanh(context_vector)
        return hidden

def train(encoder, decoder, encoder_optimizer, decoder_optimizer, encoder_input, target, criterion):
    encoder_init_hidden = encoder.init_hidden()
    decoder_input = Variable(torch.LongTensor([SOS_token]))

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_output, encoder_hidden = encoder(encoder_input, encoder_init_hidden)
    decoder_hidden = decoder.init_hidden(encoder_hidden)

    context_vector = encoder_hidden
    loss = 0
    target_length = target.size()[0]
    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, context_vector)
        # update decoder_input
        # decoder_output is a variable
        topv, topi = decoder_output.data.topk(1)
        predicted = topi[0][0]
        decoder_input = Variable(torch.LongTensor([predicted]))
        loss += criterion(decoder_output, target[di])
        if predicted == EOS_token:
            break
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0]/ target_length

def test():
    pass

if __name__ == "__main__":
    input_lang, output_lang, pair = load_data()

    input_vec, output_vec = get_word_vectors(input_lang, output_lang, pair)
    #shuffle, split into train and test data
    embedding_size = 500
    hidden_size = 1000
    num_epochs = 100
    learning_rate = 0.0001 # decrease after some epochs
    print_every = 1000
    plot_every = 100

    encoder = SimpleEncoder(input_lang.n_words, embedding_size, batch_size, hidden_size)
    decoder = SimpleDecoder(output_lang.n_words, embedding_size, batch_size, hidden_size)

    criterion = nn.CrossEntropyLoss()
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    # train
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0
    iter_cnt = 0
    for epoch in range(num_epochs):
        for i in range(len(pair)):
            iter_cnt += 1
            input_variable = input_vec[i]
            target_variable = output_vec[i]
            loss = train(encoder, decoder, encoder_optimizer, decoder_optimizer,
                        input_variable, target_variable, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if iter_cnt % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % ((timeSince(start, iter_cnt /
                (len(pair)*num_epochs*(1.)))), iter_cnt, iter_cnt /
                (len(pair)*num_epochs*(1.)) * 100, print_loss_avg))

            if iter_cnt % plot_every == 0:
                plot_loss_avg = plot_loss_total / (plot_every*1.)
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
    showPlot(plot_losses)

    # save model
    torch.save(encoder.save_dict(), './models')
    torch.save(decoder.save_dict(), './models')
