"""Implementaion of Bahdanau et al's attention model."""

import os
import sys
import time
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
sys.path.append('../data')
sys.path.append('../')
from preprocess import *
from utils import *

#encoder
class BahdanauEncoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, hidden_size, n_layer,
    GPU_use):
        super(BahdanauEncoder, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.GPU_use = GPU_use

        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layer, dropout=0.1,
        bidirectional=True)

    def forward(self, input_var, hidden):
        batch_size = input_var.size()[0]
        seq_len = input_var.size()[1]
        embedded = self.embedding(input_var).view(seq_len, batch_size, -1)
        output, hidden = self.gru(embedded, hidden)
        # output: seq_len, batch, hidden*2(bi)
        # hidden: n_layer*2, batch, hidden
        return output, hidden

    def initHidden(self, mini_batch_size):
        hidden = Variable(torch.zeros(self.n_layer*2, mini_batch_size,
        self.hidden_size))
        if GPU_use:
            hidden = hidden.cuda()
        return hidden

class BahdanauDecoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, hidden_size, n_layer,
    a_hidden_size, GPU_use):
        super(BahdanauDecoder, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.a_hidden_size = a_hidden_size
        self.n_layer = n_layer
        self.GPU_use = GPU_use

        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        self.init = nn.Linear(hidden_size, hidden_size)
        self.align = nn.Linear(hidden_size*3, a_hidden_size)
        self.v = nn.Parameter(torch.FloatTensor(1, a_hidden_size))
        self.gru = nn.GRU(embedding_size+(hidden_size*2), hidden_size, n_layer,
        dropout=0.1)
        self.out = nn.Linear(embedding_size+hidden_size*3, num_embeddings)
        self.softmax = nn.LogSoftmax()

    def forward(self, word_inputs, prev_hidden, encoder_outputs):
        # word_inputs: batch 1, word 1
        # prev_hidden: n_layer, batch 1, hidden_size
        # encoder_outputs: seq_len, batch, hidden_size*2
        last_hidden = prev_hidden[-1]
        # e
        annot_length = encoder_outputs.size()[0]
        e = Variable(torch.zeros(annot_length))
        for i in range(annot_length):
            eij = self.align(torch.cat((last_hidden, encoder_outputs[i]),1))
            e[i] = torch.dot(self.v.view(-1), eij.view(-1))
        # alpha
        alpha = F.softmax(e).unsqueeze(0).unsqueeze(0)
        # c
        context_vector = alpha.bmm(encoder_outputs.transpose(0,1))
        # context_vector: 1, 1, H*2
        batch = word_inputs.size()[0]
        seq_len = word_inputs.size()[1]
        embedded = self.embedding(word_inputs).view(seq_len, batch, -1)
        gru_input = torch.cat((embedded, context_vector),2)
        output, hidden = self.gru(gru_input, prev_hidden)
        t = torch.cat((output, embedded, context_vector), 2).squeeze(0)
        out = self.softmax(self.out(t))
        return out, hidden, alpha

    def initHidden(self, backward_state):
        # backward_state: b, h
        hidden = F.tanh(self.init(backward_state.unsqueeze(0)))
        hidden = hidden.repeat(self.n_layer, 1, 1)
        return hidden

#train
def train(encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
encoder_input, target_variable, GPU_use):
    mini_batch_size = encoder_input.size()[0]
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_hidden = encoder.initHidden(mini_batch_size) # L*2, B, H
    hidden_size = encoder_hidden.size()[2]
    encoder_outputs, encoder_hidden = encoder(encoder_input, encoder_hidden)
    # encoder_outputs: seq_len, b, h*2
    # encoder_hidden: l*2, b, h
    backward_state = encoder_outputs[0][:,:-hidden_size]
    prev_hidden = decoder.initHidden(backward_state)
    # prev_hidden: 1, B, H
    decoder_input = Variable(torch.LongTensor([SOS_token]*mini_batch_size))
    decoder_input = torch.unsqueeze(decoder_input, 1)
    if GPU_use:
        decoder_input = decoder_input.cuda()

    loss = 0
    target_length = target_variable.size()[1]
    for di in range(target_length):
        target = target_variable[:,di]
        decoder_output, decoder_hidden, _ = decoder(decoder_input, prev_hidden,
        encoder_outputs)
        prev_hidden = decoder_hidden
        topv, topi = decoder_output.data.topk(1)
        predicted = topi[0][0]
        decoder_input = Variable(torch.LongTensor([predicted]*mini_batch_size))\
        .unsqueeze(1)
        if GPU_use:
            decoder_input = decoder_input.cuda()
        target = torch.squeeze(target, 0)
        loss += criterion(decoder_output, target)
        if predicted == EOS_token:
            break
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.data[0] / target_length

#test
def test(encoder, decoder, input_sentence, output_lang, GPU_use,
TEST_MAXLENGTH):
    encoder.train(False)
    decoder.train(False)
    mini_batch_size = input_sentence.size()[0]
    encoder_hidden = encoder.initHidden(mini_batch_size)
    hidden_size = encoder_hidden.size()[2]
    encoder_outputs, encoder_hidden = encoder(input_sentence, encoder_hidden)
    backward_state = encoder_outputs[0][:, :-hidden_size]
    prev_hidden = decoder.initHidden(backward_state)
    decoder_input = Variable(torch.LongTensor([SOS_token]*mini_batch_size))
    decoder_input = torch.unsqueeze(decoder_input, 1)
    if GPU_use:
        decoder_input = decoder_input.cuda()

    result = []
    decoder_attentions = torch.zeros(TEST_MAXLENGTH, TEST_MAXLENGTH)
    for i in range(TEST_MAXLENGTH-1):
        decoder_output, decoder_hidden, attention = decoder(decoder_input,
        prev_hidden, encoder_outputs)
        prev_hidden = decoder_hidden
        topv, topi = decoder_output.data.topk(1)
        predicted = topi[0][0]
        decoder_input = Variable(torch.LongTensor([predicted]*1)).unsqueeze(1)
        if GPU_use:
            decoder_input = decoder_input.cuda()
        result.append(predicted)
        decoder_attentions[i,:attention.size(2)] += attention.squeeze(0).\
        squeeze(0).cpu().data
        if predicted == EOS_token:
            break

    encoder.train(True)
    decoder.train(True)
    return result, decoder_attentions[:i+1,]

if __name__ == "__main__":
    data_dir = '../data/kor-eng/kor.txt'
    GPU_use = False
    SEQ_MAX_LENGTH = 50
    TEST_MAXLENGTH = 50
    mini_batch_size = 1
    n_epochs = 10
    learning_rate = 0.001
    hidden_size = 1000
    a_hidden_size = 1000
    embedding_size = 620
    n_layer = 1
    print_every = 1000
    plot_every = 100

    # load data
    train_input, train_target, test_input, test_target,input_lang, output_lang,\
    train_input_lengths, train_target_lengths\
    = getTrainAndTestSet(data_dir, mini_batch_size, SEQ_MAX_LENGTH, GPU_use)
    print("Data Preparation Done.")

    # define encoder, decoder
    encoder = BahdanauEncoder(input_lang.n_words, embedding_size, hidden_size,
    n_layer, GPU_use)
    decoder = BahdanauDecoder(output_lang.n_words, embedding_size, hidden_size,
    n_layer, a_hidden_size, GPU_use)
    # define optimizer, criterion
    criterion = nn.CrossEntropyLoss()
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    if GPU_use:
        encoder.cuda()
        decoder.cuda()

    # train
    print("Training...")
    start = time.time()
    plot_losses = []
    plot_loss_total = 0
    print_loss_total = 0
    total_iter = len(train_input) * n_epochs *1.
    iter_cnt = 0
    for epoch in range(n_epochs):
        for i in range(len(train_input)):
            iter_cnt += 1
            input_var = train_input[i]
            target_var = train_target[i]
            loss = train(encoder, decoder, encoder_optimizer, decoder_optimizer,
            criterion, input_var, target_var, GPU_use)
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

    # save model
    torch.save(encoder.state_dict(), './attention1_encoder.pkl')
    torch.save(decoder.state_dict(), './attention1_decoder.pkl')
    print("Model Saved.")

    print("Testing...")
    # test
    results = []
    for s in test_input:
        query = [input_lang.index2word[idx] for idx in s.data[0]]
        output_words, attentions = test(encoder, decoder, s, output_lang,
        GPU_use, TEST_MAXLENGTH)
        translated = [output_lang.index2word[idx] for idx in output_words]
        results.append((query, translated))
        showAttention(' '.join(query), translated, attentions)
    saveTranslatedResults(results, 'attention1_result.txt')
    print("Test done.")
