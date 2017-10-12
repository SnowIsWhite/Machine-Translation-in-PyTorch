"""Prerpocess parallel data."""

"""Make json files of word2index and index2word."""
import json
import os
import re
import unicodedata

SOS_token = 0
EOS_token = 1
UNK_token = 2

data1 = './kor-eng/kor.txt'
data2_en = './news-crawl-koen-v01/news-crawl-koen-v01.en'
data2_ko = './news-crawl-koen-v01/news-crawl-koen-v01.ko'

class Preprocess:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = {0: 'SOS', 1: 'EOS', 2: 'UNK'}
        self.word2count = {}
        self.n_words = 3

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

"""
# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        )
"""
# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    #s = unicodeToAscii(s.lower().strip())
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    #s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = s.strip()
    return s

def reverse_sentence(s):
    reverse = []
    for word in s.split():
        reverse.append(word)
    return ' '.join(reverse)

def make_pair(sa1, sa2):
    pair = []
    for idx, s in enumerate(sa1):
        pair.append([s, sa2[idx]])
    return pair

def read_data(data_dir):
    lang1 = []
    lang2 = []
    with open(data_dir, 'r') as f:
        for line in f.readlines():
            lang = line.split('\t')
            lang1.append(lang[0])
            lang2.append(lang[1])
    """
    with open(data2_en, 'r') as f:
        for line in f.readlines():
            eng.append(line)

    with open(data2_ko, 'r') as f:
        for line in f.readlines():
            kor.append(line)
    """
    return lang1, lang2

def prepare_data(data_dir):
    lang1, lang2 = read_data(data_dir)
    lang1 = [normalizeString(s) for s in lang1]
    lang2 = [normalizeString(s) for s in lang2]
    reversed_lang1 = []
    for s in lang1:
        reversed_lang1.append(reverse_sentence(s))
    pairs = make_pair(reversed_lang1, lang2)
    print("Counting words...")
    input_lang = Preprocess('lang1')
    output_lang = Preprocess('lang2')
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang, pairs

"""
if __name__ == "__main__":
    prepare_data()
"""
