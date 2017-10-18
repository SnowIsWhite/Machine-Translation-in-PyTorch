"""Preprocess parallel data."""
import json
import os
import re
import random
import torch
import torch.nn.utils.rnn as rnn
from torch.autograd import Variable

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

class Preprocess:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = {0: 'PAD', 1: 'SOS', 2: 'EOS', 3: 'UNK'}
        self.word2count = {}
        self.n_words = 4

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = 1
            self.n_words += 1
        else:
            self.word2count[word] += 1

def normalizeString(s):
    #s = unicodeToAscii(s.lower().strip())
    s = s.lower().strip()
    s = re.sub(r"([.!?':;])", r" \1", s)
    s = s.strip()
    return s

def reverseSentence(s):
    reverse = []
    for word in s.split():
        reverse.append(word)
    return ' '.join(reverse)

def makePair(sa1, sa2):
    pairs = []
    for idx, s in enumerate(sa1):
        pairs.append([s, sa2[idx]])
    return pairs

def readData(data_dir):
    lang1 = []
    lang2 = []
    with open(data_dir, 'r') as f:
        for line in f.readlines():
            lang = line.split('\t')
            lang1.append(lang[0])
            lang2.append(lang[1])
    return lang1, lang2

def prepareData(data_dir, MAX_LENGTH):
    pre_lang1, pre_lang2 = readData(data_dir)
    lang1 = []
    lang2 = []
    for idx, s in enumerate(pre_lang1):
        s = normalizeString(s)
        if len(s.split()) > MAX_LENGTH:
            continue
        lang1.append(s)
        lang2.append(normalizeString(pre_lang2[idx]))
    reversed_lang1 = []
    for s in lang1:
        reversed_lang1.append(reverseSentence(s))
    pairs = makePair(reversed_lang1, lang2)
    print("Counting words...")
    input_lang = Preprocess('lang1')
    output_lang = Preprocess('lang2')
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang, pairs

def phraseToIndex(lang, phrase):
    return [lang.word2index[word] for word in phrase.split()]

def testphraseToIndex(lang, phrase):
    indicies = []
    for word in phrase.split():
        if word not in lang.word2index:
            indicies.append(UNK_token)
        else:
            indicies.append(lang.word2index[word])
    return indicies

def phraseToTensor(lang, batch_data, GPU_use):
    indicies = [phraseToIndex(lang, phrase) for phrase in batch_data]
    for phrase in indicies:
        phrase.append(EOS_token)
    indicies.sort(key=len, reverse=True)
    lengths = [len(phrase) for phrase in indicies]
    longest_seq_len = len(indicies[0])
    tensor = torch.LongTensor(len(indicies), longest_seq_len)
    for idx, phrase in enumerate(indicies):
        while len(phrase) < longest_seq_len:
            phrase.append(PAD_token)
        tensor[idx] = torch.LongTensor(phrase)
    tensor_var = Variable(tensor)
    if GPU_use:
        tensor_var = tensor_var.cuda()
    return tensor_var, lengths

def dataToIndex(input_lang, output_lang, pairs, batch_size, GPU_use):
    # group pairs by batch size
    cnt = 1
    batch_lang1 = []
    batch_lang2 = []
    input_vars = []
    target_vars = []
    input_lengths = []
    target_lengths = []
    for pair in pairs:
        batch_lang1.append(pair[0])
        batch_lang2.append(pair[1])
        if cnt % batch_size == 0:
            v, l = phraseToTensor(input_lang, batch_lang1, GPU_use)
            input_vars.append(v)
            input_lengths.append(l)
            v, l = phraseToTensor(output_lang, batch_lang2, GPU_use)
            target_vars.append(v)
            target_lengths.append(l)
            batch_lang1 = []
            batch_lang2 = []
            cnt = 0
        cnt += 1
    return input_vars, target_vars, input_lengths, target_lengths

def testDataToIndex(input_lang, output_lang, pairs, GPU_use):
    test_input = []
    test_target = []
    for pair in pairs:
        input_ = torch.LongTensor([testphraseToIndex(input_lang, pair[0])])
        target_ = torch.LongTensor([testphraseToIndex(output_lang, pair[1])])
        input_ = Variable(input_)
        target_ = Variable(target_)
        if GPU_use:
            input_ = input_.cuda()
            target_ = target_.cuda()
        test_input.append(input_)
        test_target.append(target_)
    return test_input, test_target

def getTrainAndTestSet(data_dir, batch_size, MAX_LENGTH, GPU_use):
    # load data
    input_lang, output_lang, pairs = prepareData(data_dir, MAX_LENGTH)
    # shuffle pair
    number_of_data = len(pairs)
    random.shuffle(pairs)

    # split into train and test set
    division = (int)(number_of_data / batch_size)
    train_portion = (int)(division * 0.9)
    test_portion = division - train_portion
    train_portion = batch_size * train_portion
    test_portion = batch_size * test_portion
    train_data = pairs[:train_portion]
    test_data = pairs[(train_portion+1):train_portion+test_portion+1]

    # get train dataset
    train_input, train_target, train_input_lengths, train_target_lengths\
     = dataToIndex(input_lang, output_lang, train_data,
    batch_size, GPU_use)

    # get test dataset
    test_input, test_target = testDataToIndex(input_lang, output_lang,
     test_data, GPU_use)

    return train_input, train_target, test_input, test_target, input_lang,\
    output_lang, train_input_lengths, train_target_lengths

def toPackedVariable(variable, lengths):
    # variable dim: B, S, *
    packed_vars = rnn.pack_padded_sequence(variable, lengths)
    return packed_vars
