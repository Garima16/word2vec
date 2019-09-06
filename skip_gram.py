# implement Mikolov's skip-gram word embedding algorithm

import torch
import torch.nn.functional as F
import time


class Model:
    def __init__(self, lr, emb_size, context_size, device):
        """
        :param: lr(learning rate),
                emb_size(nodes in hidden layer),
                context_size(context window of center word; get this much words to left and right)
                    (total context words = 2 * context_size)
        """
        self.lr = lr
        self.emb_size = emb_size  # hidden layer size
        self.context_size = context_size
        self.device = device

        self.word2idx = {}  # word:[id, frequency]
        self.idx2word = {}  # id:word mapping
        self.corpus_idx = []  # represent words in corpus as numbers,store unique index for each word in corpus
        self.voc_size = 0

    def extract_unique_words(self, corpus, threshold=10):
        print("1")
        for word in corpus.strip().split():
            if word not in self.word2idx:
                # first index of the value's list stores the word 2 index mapping and 2nd stores the frequency
                self.word2idx[word] = [self.voc_size, 1]
                self.corpus_idx.append(self.voc_size)
                self.voc_size += 1
            else:
                self.corpus_idx.append(self.word2idx[word][0])  # store corr id already saved in word2idx
                self.word2idx[word][1] += 1  # increase the word frequency
        print("2")
        id = 0
        d = dict()  # temporary dict to hold frequent words and their index
        for word in self.word2idx.keys():
            # if word frequency is above threshold, add to d
            if self.word2idx[word][1] >= threshold:
                d[word] = self.word2idx[word][0]
                self.idx2word[id] = word  # store id:word only for frequent words
                id += 1
        del self.word2idx
        self.word2idx = d
        print("4")
        # in corpus_idx, only keep those indices which are present in idx2word dict(indices corr to frequent words)
        self.corpus_idx = [ind for ind in self.corpus_idx if ind in self.idx2word]
        self.voc_size = len(self.word2idx)
        print("vocabulary size : {}, word2idx len : {},idx2word len : {},len of corpus_idx : {}".format(
            self.voc_size,
            len(self.word2idx),
            len(self.idx2word),
            len(self.corpus_idx),
        ))

    def forward(self, k, context_words):
        '''
        :param : k(index of center word), indices(list of indices of context words),
        1. pick kth row of w1 (contains N values) - center word representation of input word - h(size : 1 x N)
        2. for j in vocab(output layer): pick jth col of w2(o), find Y[j] = exp(h.o)
        3. normaliser = sum(Y)
        4. Y = Y/normaliser
        5. for j in vocab(output layer) :
            a. if j in indices, e[j] = e[j] + ( y[j] - 1 )
            b. else e[j] = e[j] + ( len(indices) * y[j] ) ; update the value that many times as there are context words
        :return: e(accumulated error); this will be back-propagated for each sample in trg data
        '''
        num_context_words = len(context_words)
        h = self.w1[k].view(-1, self.emb_size)  # change to 2D type tensor
        # y = torch.zeros((1, self.voc_size), dtype=torch.float32, device=self.device)
        y = torch.zeros((1, self.voc_size), device=self.device)
        # dot product of h and w2 ; torch.mm requires 2D tensors to multiply
        o = torch.mm(h, self.w2)
        y = F.softmax(o).view(self.voc_size)
        print("y's shape : ", y.shape)
        # y = torch.exp()  # y is 2D tensor of shape (1,voc_size).
        # y = torch.clamp(y, max=100000)
        # normaliser = torch.tensor(torch.sum(y).item())  # get sum result
        # normaliser = torch.clamp(normaliser, max=400000000)
        # print(type(normaliser))
        # print("normaliser value : {}".format(normaliser))
        # y = torch.div(y, normaliser).view(self.voc_size)  # convert to 1D tensor of shape(voc_size)
        # error = torch.zeros(self.voc_size, dtype=torch.float32, device=self.device)  # 1D tensor of shape (voc_size)
        error = torch.zeros(self.voc_size, device=self.device)  # 1D tensor of shape (voc_size)
        for i in range(self.voc_size):
            if i in context_words:
                error[i] = error[i] + (y[i] - 1)
            else:
                error[i] = error[i] + (num_context_words * y[i] - 1)
        # check if all values of error are going to nan

        print("error : ", error)
        return error

    def backprop(self, k, error):
        '''
        :param: k(center word index), error(accumulated error) of shape (voc_size)
        1. for j in vocab,for i in hidden_layer: w2[i][j] -= lr * error[j] * w1[k][i]
        2. eh = zeros(N)
        3. for i in hidden_layer : eh[i] = error.(ith row of w2), w1[k][i] -= lr * eh[i]
        '''
        for j in range(self.voc_size):
            self.w2[:, j] = self.w2[:, j] - (torch.mul(self.lr, torch.mul(error[j].item(), self.w1[k])))
        # eh = torch.zeros((1, self.emb_size), dtype=torch.float32, device=device)
        eh = torch.zeros((1, self.emb_size), device=device)
        # eh is 2D tensor shape :(1, emb_size)
        eh = torch.mm(error.view(1, self.voc_size), torch.t(self.w2))  # dot product of error and w2.T(transpose)
        self.w1[k] = self.w1[k] - torch.mul(self.lr, eh.view(self.emb_size))
        # print("w1 weights : {}, w2 weights : {}".format(self.w1, self.w2))

    def train(self, epochs):
        '''
        :param: epochs
        1. initialise emb_mat(embedding matrix) = zeros(voc_size, emb_size)
        2. for each epoch :
                for k = 1 to len(words_in_corpus):
                    k=center word
                    context_words : get context_size words to the left and right of center word
                    e = forward()
                    backprop()
        3. update all values in emb_mat as : emb_mat[i][j] = w1[i][j] + w2[j][i];
                (final representation of word w : add w's representations from both w1 and w2)
        :return: emb_mat
        '''
        # initialise first weight matrix w1 with random values b/w 0 & 1; shape:(vocab_size x embedding_dim)
        # and similarly 2nd matrix w2; shape : (embedding_dim x vocab_size)
        # self.w1 = torch.rand((self.voc_size, self.emb_size), dtype=torch.float32, device=device)  # V x N
        # self.w2 = torch.rand((self.emb_size, self.voc_size), dtype=torch.float32, device=device)  # N x V
        # emb_mat = torch.zeros((self.voc_size, self.emb_size), dtype=torch.float32, device=device)  # V x N
        self.w1 = torch.rand((self.voc_size, self.emb_size), device=device)  # V x N
        self.w2 = torch.rand((self.emb_size, self.voc_size), device=device)  # N x V
        print("w2 weights: ", self.w2)
        emb_mat = torch.zeros((self.voc_size, self.emb_size), device=device)  # V x N
        for _ in range(epochs):
            words_in_corpus = len(self.corpus_idx)
            count = 1
            for k in self.corpus_idx:  # each word in corpus will act as center word
                print("k value : ", k)
                left_words = []
                rt_words = []
                for i in range(1, self.context_size + 1):
                    if k - i >= 0:
                        left_words.append(self.corpus_idx[k - i])
                    if k + i <= words_in_corpus - 1:
                        rt_words.append(self.corpus_idx[k + i])
                context_words = left_words + rt_words
                # print("center word : {}, context words : {}".format(k, context_words))
                error = self.forward(k, context_words)
                self.backprop(k, error)
                print("finish updating weights for sample {}".format(count))
                count += 1
                print("------------------------------------------------------------------------")
        emb_mat = self.w1 + torch.t(self.w2)
        print("final embedding matrix : ", emb_mat)
        # print("left words : ", left_words)
        # print("right words : ", rt_words)
        # print("index 0 word : ",self.idx2word[0])
        # print("index 1 word : ", self.idx2word[1])
        # print("index 2 word : ", self.idx2word[2])

    def generate_trg_data(self, corpus):
        '''
        generate (center word, context_words) pairs
        1.
        :param corpus:
        :return:
        '''

        pass


if __name__ == '__main__':
    start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device available : ", device)
    with open('../examples/word_language_model/my_data/train.txt') as file:
        data = file.read()[:2000000]
        word2vec = Model(lr=0.001, emb_size=50, context_size=2, device=device)
        word2vec.extract_unique_words(corpus=data, threshold=10)
        word2vec.train(epochs=1)
    print("time taken : ", time.time() - start)
