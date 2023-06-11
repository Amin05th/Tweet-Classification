import torch
import nltk
import string
from nltk.stem.porter import PorterStemmer
import itertools


def tokenization(sentence):
    return nltk.word_tokenize(sentence)


def ignore_words(sentence):
    ignore_list = string.punctuation
    return [word for word in sentence if word not in ignore_list]


def stemming(sentence):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in sentence]


def bag_of_words(sentences):
    vocab = set(word for sentence in sentences for word in sentence)
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    indexed_sentences = [[word2idx[word] for word in sentence] for sentence in sentences]
    indexed_sentences = [torch.tensor(sentence) for sentence in indexed_sentences]
    padded_sentences = torch.nn.utils.rnn.pad_sequence(indexed_sentences, batch_first=True)
    return padded_sentences
