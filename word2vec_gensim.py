# train using Gensim
import multiprocessing
from gensim.models import Word2Vec
from time import time

corpus = []
path = '../examples/word_language_model/my_data/valid.txt'
with open(path, 'r') as file:
    sentences = file.read()
    sentences = sentences.split('\n')
    for sentence in sentences:
        words = sentence.split()
        corpus.append(words)
# print(len(corpus[1]))


cores = multiprocessing.cpu_count()
model = Word2Vec(
    min_count=10,
    size=50,
    alpha=0.03,
    min_alpha=0.0007,
    workers=cores - 1
)
t = time()

model.build_vocab(corpus, progress_per=10000)

print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
vocab = model.wv.vocab
print(vocab)
print("Vocabulary size : ", len(vocab))
t = time()

model.train(corpus, total_examples=model.corpus_count, epochs=30, report_delay=1)
model.save("word2vec.model")
print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
