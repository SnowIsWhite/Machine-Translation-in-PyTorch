"""Brief Implementation of Seq2Seq model."""

import os
import sys
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
sys.path.append('../data')
sys.path.append('../')
from preprocess import*
from utils import*

# encoder
class LSTMEncoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, hidden_size,
    GPU_use, n_layer=1):
        super(LSTMEncoder, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.GPU_use = GPU_use

        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layer, dropout=0.5)

    def forward(self, input_var, lengths, hidden, cell):
        # shape of input_var : B, S
        batch_size = input_var.size()[0]
        seq_len = input_var.size()[1]
        embedded = self.embedding(input_var) # B, S, E
        # change to packed sequence
        embedded = embedded.view(seq_len, batch_size, -1)
        embedded = toPackedVariable(embedded, lengths)
        # change dim (view)
        output, (h, c) = self.lstm(embedded, (hidden, cell))
        return output, h, c

    def initHiddenAndCell(self, mini_batch_size):
        hidden = Variable(torch.zeros(self.n_layer, mini_batch_size,
        self.hidden_size))
        cell = Variable(torch.zeros(self.n_layer, mini_batch_size,
        self.hidden_size))
        if GPU_use:
            hidden = hidden.cuda()
            cell = cell.cuda()
        return hidden, cell

# decoder
class LSTMDecoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, hidden_size,
     GPU_use, n_layer=1):
        super(LSTMDecoder, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.GPU_use = GPU_use
        self.n_layer = n_layer

        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layer, dropout=0.5)
        self.linear = nn.Linear(hidden_size, num_embeddings)
        self.softmax = nn.LogSoftmax()

    def forward(self, decoder_input, prev_hidden, prev_cell, context_vector):
        # decoder input would be a single word: 1, batch
        embedded = self.embedding(decoder_input) #1, B, N
        # integrate context vector into this
        # context_vector: encoder hidden (var)
        # (num_layers * num_directions, batch, hidden_size)
        output, (h, c) = self.lstm(embedded, ((prev_hidden + context_vector),
         prev_cell))
        # output: seq_len, batch, N where seq_len is 1
        output = output.squeeze(0) # batch, N
        out = self.linear(output)
        # out: batch, num_embeddings
        out = self.softmax(out)
        return out, h, c

    def initHiddenAndCell(self, mini_batch_size, context_vector):
        hidden = context_vector
        cell = Variable(torch.zeros(self.n_layer, mini_batch_size,
        self.hidden_size))
        if GPU_use:
            cell = cell.cuda()
        return hidden, cell

# train
def train(encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
encoder_input, target_variable, train_input_lengths, train_target_lengths,
GPU_use):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_hidden, encoder_cell = encoder.initHiddenAndCell(mini_batch_size)
    encoder_output, encoder_hidden, encoder_cell = encoder(encoder_input,
    train_input_lengths, encoder_hidden, encoder_cell)

    context_tensor = encoder_hidden[:encoder.n_layer].data
    context_tensor.repeat(decoder.n_layer,1,1)
    context_vector = Variable(context_tensor)
    if GPU_use:
        context_vector.cuda()
    decoder_hidden, decoder_cell = decoder.initHiddenAndCell(mini_batch_size,
    context_vector)

    batch_size = encoder_input.size()[0]
    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    decoder_input = decoder_input.unsqueeze(0)
    if GPU_use:
        decoder_input.cuda()

    loss = 0
    target_seq_len = target_variable.size()[1]
    target_variable = target_variable.transpose(0,1)
    for i in range(target_seq_len):
        decoder_output, decoder_hidden, decoder_cell = decoder(decoder_input,
        decoder_hidden, decoder_cell, context_vector)
        # decoder output shape: batch, num_embeddings
        topv, topi = decoder_output.data.topk(1)
        # topi : batch, 1
        print(topi)
        break
    return loss
# greedy search decoder
# test
# beam search in decoder

if __name__ == "__main__":
    data_dir = '../data/kor-eng/kor.txt'
    seq_max_length = 30
    GPU_use = False
    mini_batch_size = 20
    learning_rate = 0.7
    hidden_size = 1000
    embedding_size = 1000
    n_layer = 4
    n_epochs = 8
    print_every = 10
    plot_every = 10

    train_input, train_target, test_input, test_target,input_lang, output_lang,\
    train_input_lengths, train_target_lengths\
     = getTrainAndTestSet(data_dir, mini_batch_size, seq_max_length, GPU_use)
    print("Data Preparation Done.")

    # define encoder, decoder, optimizaion, criterion
    encoder = LSTMEncoder(input_lang.n_words, embedding_size, hidden_size,
    GPU_use, n_layer)
    decoder = LSTMDecoder(output_lang.n_words, embedding_size, hidden_size,
    GPU_use, n_layer)
    criterion = nn.CrossEntropyLoss()
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    if GPU_use:
        encoder.cuda()
        decoder.cuda()

    print("Training...")
    # train
    for epoch in range(n_epochs):
        for i in range(len(train_input)):
            input_var = train_input[i]
            target_var = train_target[i]
            loss = train(encoder, decoder, encoder_optimizer, decoder_optimizer,
             criterion, input_var, target_var, train_input_lengths[i],
             train_target_lengths[i], GPU_use)
            break
        break

    print("Training done.")
    # save model
    print("Model Saved.")

    print("Testing...")
    # test
    # turn off training mode
    print("Test done.")
