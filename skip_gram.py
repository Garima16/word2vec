# implement Mikolov's skip-gram word embedding algorithm

import torch
import numpy as np


class Model:
    def __init__(self, lr, emb_size, context_size):
        """
        :param: lr(learning rate),
                emb_size(nodes in hidden layer),
                context_size(context window of center word; get this much words to left and right)
                    (total context words = 2 * context_size)
        """
        self.lr = lr
        self.emb_size = emb_size  # hidden layer size
        self.context_size = context_size

        # initialise first weight matrix w1 with random values b/w 0 & 1; shape:(vocab_size x embedding_dim)
        # and similarly 2nd matrix w2; shape : (embedding_dim x vocab_size)
        self.w1 = torch.rand(self.voc_size, self.emb_size)  # V x N
        self.w2 = torch.rand(self.emb_size, self.voc_size)  # N x V

        self.word2idx = {}
        self.word2cnt = {}
        self.voc_size = 0

    def extract_unique_words(self, corpus, threshold=10):
        # self.voc_size = 0
        for word in corpus.strip().split():
            if word not in self.word2cnt:
                self.word2cnt[word] = 1
            else:
                self.word2cnt[word] += 1

        for word in self.word2cnt.keys():
            # if the frequency of word is above threshold, then keep it and assign it an index, else discard it
            if self.word2cnt[word] >= threshold:
                self.word2idx[word] = self.voc_size
                self.voc_size += 1
        # return self.word2idx, self.voc_size

    def forward(self, k, indices):
        '''
        :param : k(index of center word), indices(list of indices of context words),
        1. pick kth row of w1 (contains N values) - center word representation of input word - h(size : 1 x N)
        2. for j in vocab(output layer): pick jth col of w2(o), find Y[j] = exp(h.o)
        3. normaliser = sum(Y)
        4. for j in vocab(output layer) :
            a. if j in indices, e[j] = e[j] + ( y[j] - 1 )
            b. else e[j] = e[j] + ( len(indices) * y[j] ) ; update the value that many times as there are context words
        :return: e(accumulated error); this will be back-propagated for each sample in trg data
        '''
        pass

    def backprop(self, k, e, ):
        '''
        :param: k(center word index), e(accumulated error)
        1. for j in vocab,for i in hidden_layer: w2[i][j] -= lr * e[j] * w1[k][i]
        2. eh = zeros(N)
        3. for i in hidden_layer : eh[i] = e.(ith row of w2), w1[k][i] -= lr * eh[i]
        :return:
        '''
        pass

    def train(self):
        '''
        :param: epochs
        1. initialise emb_mat(embedding matrix) = zeros(voc_size, emb_size)
        2. for each epoch :
                for sentence in corpus:
                    for k = 1 to len(sentence):
                        k=center word
                        context_words : get context_size words to the left and right of center word
                        e = forward()
                        backprop()
        3. update all values in emb_mat as : emb_mat[i][j] = w1[i][j] + w2[j][i];
                (final representation of word w : add w's representations from both w1 and w2)
        :return: emb_mat
        '''

        pass
