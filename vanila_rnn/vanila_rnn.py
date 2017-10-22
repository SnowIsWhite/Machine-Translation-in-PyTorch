"""seq2seq neural machine translation with one layer RNN."""
"""
I borrowed some code from PyTorch Tutorial
http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html.
"""

import os
import sys
import time
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
sys.path.append('../data/')
sys.path.append('../')
from preprocess import *
from utils import *

SOS_token = 1
EOS_token = 2
UNK_token = 3

class SimpleEncoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, mini_batch_size,
    hidden_size, n_layer, GPU_use):
        super(SimpleEncoder, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.mini_batch_size = mini_batch_size
        self.GPU_use = GPU_use

        #nn.Embedding: input (N, W) N- minibatch W- number of inidicies to extract per minibatch
        #output: (N, W, embedding_dim)
        #params: num_embeddings: size of dictionary of embeddings,
        #embedding_dim : size of each embedding vector
        self.embedding = nn.Embedding(num_embeddings, embedding_size)

        #RNN: Input: (input, h0)
        #input: seq_len, batch, input_size
        #h0: num_layers * num_directions, batch, hidden_size
        #output: output, h_n
        #output: seq_len, batch, hidden_size * num_directions-> not used in encoder
        #h_n : num_layers * num_directions, batch, hidden_size
        self.gru = nn.GRU(embedding_size, hidden_size, n_layer, dropout=0.1)

    def forward(self, input_variable, hidden):
        batch_size = input_variable.size()[0]
        seq_len = input_variable.size()[1]
        embedded = self.embedding(input_variable).view(seq_len, batch_size, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self, mini_batch_size):
        hidden = Variable(torch.zeros(self.n_layer, mini_batch_size,
        self.hidden_size))
        if self.GPU_use:
            hidden = hidden.cuda()
        return hidden

class SimpleDecoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, mini_batch_size,
    hidden_size, n_layer, GPU_use):
        super(SimpleDecoder, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.mini_batch_size = mini_batch_size
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.GPU_use = GPU_use

        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size*2, n_layer)
        self.out = nn.Linear(hidden_size*2, num_embeddings)
        self.softmax = nn.LogSoftmax()

    def forward(self, word_inputs, prev_hidden, context_vector):
        # context_vector: 1, mini_batch_size, hidden_size * num_directions
        # prev_hidden : n_layer, mini_batch_size,  hidden_size
        seq_len = word_inputs.size()[1]
        batch_size = word_inputs.size()[0]
        n_layer = prev_hidden.size()[0]

        embedded = self.embedding(word_inputs).view(seq_len, batch_size, -1)
        context_vector = torch.unsqueeze(context_vector, self.n_layer)
        hidden = torch.cat((prev_hidden, context_vector), 2)
        # hidden : n_layer, mini_batch_size, hidden_size*2
        output, hidden = self.gru(embedded, hidden)
        # output: seq_len, bath_size, hidden_size*2
        hidden = hidden[:,:,:self.hidden_size]
        output = torch.squeeze(output, 0)
        output = self.softmax(self.out(output))
        # output: batch_size, num_embeddings
        return output, hidden

    def initHidden(self, context_vector):
        context_vector = torch.unsqueeze(context_vector, self.n_layer)
        context_vector = context_vector.repeat(self.n_layer, 1,1)
        hidden = F.tanh(context_vector)
        if self.GPU_use:
            hidden = hidden.cuda()
        return hidden

def train(encoder, decoder, encoder_optimizer, decoder_optimizer, encoder_input,
target_variable, criterion, GPU_use):
    mini_batch_size = encoder_input.size()[0]
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_hidden = encoder.initHidden(mini_batch_size)
    encoder_output, encoder_hidden = encoder(encoder_input, encoder_hidden)
    # encoder_output: seq_len, batch, hidden_size
    context_vector = encoder_output[-1]
    decoder_hidden = decoder.initHidden(context_vector)
    decoder_input = Variable(torch.LongTensor([SOS_token] * mini_batch_size))
    decoder_input = torch.unsqueeze(decoder_input, 1)
    # decoder_input: batch_size, seq_len
    if GPU_use:
        decoder_input = decoder_input.cuda()

    loss = 0
    # target_variable: batch_size, seq_len
    target_length = target_variable.size()[1]
    for di in range(target_length):
        target = target_variable[:,[di]]
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden,
        context_vector)
        topv, topi = decoder_output.data.topk(1)
        predicted = topi[0][0]
        decoder_input = Variable(torch.LongTensor([predicted]*mini_batch_size))
        decoder_input = torch.unsqueeze(decoder_input, 1)
        if GPU_use:
            decoder_input = decoder_input.cuda()
        target = torch.squeeze(target, 0)
        loss += criterion(decoder_output, target)
        if predicted == EOS_token:
            break
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    # loss dim?
    return loss.data[0] / target_length

def test(encoder, decoder, input_sentence, output_lang, GPU_use,
TEST_MAXLENGTH=30):
    encoder.train(False)
    decoder.train(False)
    mini_batch_size = input_sentence.size()[0] #1
    encoder_hidden = encoder.initHidden(mini_batch_size)
    encoder_output, encoder_hidden = encoder(input_sentence, encoder_hidden)
    context_vector = encoder_output[-1]
    decoder_hidden = decoder.initHidden(context_vector)
    decoder_input = Variable(torch.LongTensor([SOS_token]*1))
    decoder_input = torch.unsqueeze(decoder_input, 1)
    if GPU_use:
        decoder_input = decoder_input.cuda()

    result = []
    for i in range(TEST_MAXLENGTH-1):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden,
        context_vector)
        topv, topi = decoder_output.data.topk(1)
        predicted = topi[0][0]
        decoder_input = Variable(torch.LongTensor([predicted]*1))
        decoder_input = torch.unsqueeze(decoder_input, 1)
        if GPU_use:
            decoder_input = decoder_input.cuda()
        result.append(predicted)
        if predicted == EOS_token:
            break
    return result

if __name__ == "__main__":
    data_dir = '../data/kor-eng/kor.txt'
    SEQ_MAX_LENGTH = 30
    TEST_MAXLENGTH = 30
    GPU_use = False
    mini_batch_size = 1
    learning_rate = 0.001
    hidden_size = 1000
    embedding_size = 1000
    n_layer = 1
    n_epochs = 8
    print_every = 1000
    plot_every = 10

    train_input, train_target, test_input, test_target,input_lang, output_lang,\
    train_input_lengths, train_target_lengths\
     = getTrainAndTestSet(data_dir, mini_batch_size, SEQ_MAX_LENGTH, GPU_use)
    print("Data Preparation Done.")

    encoder = SimpleEncoder(input_lang.n_words, embedding_size, mini_batch_size,
    hidden_size, n_layer, GPU_use)
    decoder = SimpleDecoder(output_lang.n_words, embedding_size, mini_batch_size
    , hidden_size, n_layer, GPU_use)
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
    total_iter = len(train_input) * n_epochs *1.
    iter_cnt = 0
    for epoch in range(n_epochs):
        for i in range(len(train_input)):
            iter_cnt += 1
            input_var = train_input[i]
            target_var = train_target[i]
            loss = train(encoder, decoder, encoder_optimizer, decoder_optimizer,
            input_var, target_var, criterion, GPU_use)
            print_loss_total += loss
            plot_loss_total += loss

            if iter_cnt % print_every == 0:
                print_loss_avg = print_loss_total / print_every*1.
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % ((timeSince(start,iter_cnt/total_iter)),
                iter_cnt, iter_cnt/total_iter * 100, print_loss_avg))

            if iter_cnt % plot_every == 0:
                plot_loss_avg = plot_loss_total / (plot_every*1.)
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
            break
        break
    showPlot(plot_losses, 'vanilaRNN')
    print("Training done.")

    #save model
    torch.save(encoder.state_dict(), './rnn_encoder.pkl')
    torch.save(decoder.state_dict(), './rnn_decoder.pkl')
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
