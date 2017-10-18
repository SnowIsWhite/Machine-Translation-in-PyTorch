"""Brief Implementation of Seq2Seq model."""
"""
To train with certain batch size, implementation of packed sequences
and delicate handling of loss function is required.
"""

import os
import sys
import time
import pickle
import math
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
    GPU_use, n_layer):
        super(LSTMEncoder, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.GPU_use = GPU_use

        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layer, dropout=0.5)

    def forward(self, input_var, hidden, cell):
        # shape of input_var : B, S
        batch_size = input_var.size()[0]
        seq_len = input_var.size()[1]
        embedded = self.embedding(input_var).view(seq_len, batch_size, -1)
        # B, S, E -> S, B, E
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
     GPU_use, n_layer):
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
        # decoder input would be a single word: 1
        batch_size = decoder_input.size()[0]
        seq_len = decoder_input.size()[1]
        embedded = self.embedding(decoder_input).view(seq_len, batch_size, -1)
        #B, 1, N - > 1, B, N
        # integrate context vector into this
        # context_vector: encoder hidden (var)
        # (num_layers * num_directions, batch, hidden_size)
        output, (h, c) = self.lstm(embedded, ((prev_hidden + context_vector),
         prev_cell))
        # output: seq_len, batch, N where seq_len is 1
        output = output.squeeze(0) # batch, N
        out = self.linear(output)
        # out: batch, num_embeddings (batch = 1)
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
    encoder_hidden, encoder_cell)

    context_tensor = encoder_hidden[:encoder.n_layer].data
    context_tensor.repeat(decoder.n_layer, 1, 1)
    context_vector = Variable(context_tensor)
    if GPU_use:
        context_vector.cuda()
    decoder_hidden, decoder_cell = decoder.initHiddenAndCell(mini_batch_size,
    context_vector)

    batch_size = encoder_input.size()[0] # 1
    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    decoder_input = decoder_input.unsqueeze(0)
    if GPU_use:
        decoder_input.cuda()

    loss = 0
    target_seq_len = target_variable.size()[1]
    for i in range(target_seq_len):
        decoder_output, decoder_hidden, decoder_cell = decoder(decoder_input,
        decoder_hidden, decoder_cell, context_vector)
        # decoder output shape: batch, num_embeddings
        topv, topi = decoder_output.data.topk(2)
        # topi : batch, 1
        predicted = topi[0][0]
        loss += criterion(decoder_output, target_variable[0][i])
        decoder_input = Variable(torch.LongTensor([predicted] * batch_size))
        decoder_input = decoder_input.unsqueeze(0)
        if predicted == EOS_token:
            break
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.data[0] / target_seq_len

# test
def test(encoder, decoder, input_sentence, output_lang, GPU_use,
TEST_MAXLENGTH=30):
    encoder.train(False)
    decoder.train(False)
    batch_size = input_sentence.size()[0]
    encoder_hidden, encoder_cell = encoder.initHiddenAndCell(batch_size)
    encoder_output, ecoder_hidden, encoder_cell = encoder(input_sentence,
    encoder_hidden, encoder_cell)

    context_tensor = encoder_hidden[:encoder.n_layer].data
    context_tensor.repeat(decoder.n_layer, 1, 1)
    context_vector = Variable(context_tensor)
    if GPU_use:
        context_vector = context_vector.cuda()

    decoder_hidden, decoder_cell = decoder.initHiddenAndCell(batch_size,
    context_vector)

    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    decoder_input = decoder_input.unsqueeze(0)
    if GPU_use:
        decoder_input = decoder_input.cuda()

    # beam search (2)
    res_array1 = []
    res_array2 = []
    res_prob1 = 1
    res_prob2 = 1
    decoder_output, decoder_hidden, decoder_Cell = decoder(decoder_input,
    decoder_hidden, decoder_cell, context_vector)
    topv, topi = decoder_output.data.topk(2)
    res_array1.append(topi[0][0])
    res_array2.append(topi[0][1])
    res_prob1 += topv[0][0]
    res_prob2 += topv[0][1]
    for i in range(TEST_MAXLENGTH-1):
        decoder_input = Variable(torch.LongTensor([res_array1[-1]]*batch_size))
        decoder_input = decoder_input.unsqueeze(0)
        if GPU_use:
            decoder_input = decoder_input.cuda()
        decoder_output, decoder_hidden, decoder_cell = decoder(decoder_input,
        decoder_hidden, decoder_cell, context_vector)
        topv, topi = decoder_output.data.topk(2)
        if topi[0][0] == EOS_token or topi[0][1] == EOS_token:
            encoder.train(True)
            decoder.train(True)
            res_array1.append(EOS_token)
            return res_array1
        list1 = beamSearch(topv, topi, res_array1, res_prob1)

        decoder_input = Variable(torch.LongTensor([res_array2[-1]]*batch_size))
        decoder_input = decoder_input.unsqueeze(0)
        if GPU_use:
            decoder_input = decoder_input.cuda()
        decoder_output, _, _ = decoder(decoder_input, decoder_hidden,
        decoder_cell, context_vector)
        topv, topi = decoder_output.data.topk(2)
        if topi[0][0] == EOS_token or topi[0][1] == EOS_token:
            encoder.train(True)
            decoder.train(True)
            res_array2.append(EOS_token)
            return res_array2
        list2 = beamSearch(topv, topi, res_array2, res_prob2)

        #get max and update array, prob
        list_ = list1 + list2
        sorted_list = sorted(list_, key=lambda tup: tup[1])
        res_array1 = sorted_list[0][0]
        res_prob1 = sorted_list[0][1]
        res_array2 = sorted_list[1][0]
        res_prob2 = sorted_list[1][1]
    encoder.train(True)
    decoder.train(True)
    return res_array1

if __name__ == "__main__":
    data_dir = '../data/kor-eng/kor.txt'
    seq_max_length = 30
    TEST_MAXLENGTH = 30
    GPU_use = False
    mini_batch_size = 1
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
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0
    total_iter = len(train_input) * n_epochs * 1.
    iter_cnt = 0
    for epoch in range(n_epochs):
        for i in range(len(train_input)):
            iter_cnt += 1
            input_var = train_input[i]
            target_var = train_target[i]
            loss = train(encoder, decoder, encoder_optimizer, decoder_optimizer,
             criterion, input_var, target_var, train_input_lengths[i],
             train_target_lengths[i], GPU_use)
            print_loss_total += loss
            plot_loss_total += loss

            if iter_cnt % print_every == 0:
                print_loss_avg = print_loss_total / print_every*1.
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % ((timeSince(start,iter_cnt/total_iter)),
                iter_cnt, iter_cnt/total_tier * 100, print_loss_avg))

            if iter_cnt % plot_every == 0:
                plot_loss_avg = plot_loss_total / (plot_every*1.)
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
            break
        break
    showPlot(plot_losses, 'lstm')
    print("Training done.")

    # save model
    torch.save(encoder.state_dict(), './lstm_encoder_model.pkl')
    torch.save(decoder.state_dict(), './lstm_decoder_model.pkl')
    print("Model Saved.")

    print("Testing...")
    # test
    results = []
    for s in test_input:
        query = [input_lang.index2word[idx] for idx in s.data[0]]
        translated_idx = test(encoder, decoder, s, output_lang, GPU_use,
        TEST_MAXLENGTH)
        translated = [output_lang.index2word[idx] for idx in translated_idx]
        results.append((query, translated))
    saveTranslatedResults(results, 'lstm_result.txt')
    # turn off training mode
    print("Test done.")
