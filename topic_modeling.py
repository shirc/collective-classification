__author__ = 'SRC'

from gensim import matutils
from dataset import load_dataset
from gensim import models
import numpy as np

def LDA_process(dataset):
    fea, link, label = load_dataset(dataset)
    corpus = matutils.Dense2Corpus(fea, documents_columns=False)
    num_topics = 100
    print 'performing lda...'
    model = models.LdaModel(corpus, num_topics=num_topics)
    topic_fea = matutils.corpus2dense(model[corpus], num_topics)
    topic_fea = topic_fea.transpose()
    np.save('dataset/'+dataset+'/lda_fea', topic_fea)

def fea2bow(fea):
    bow = []
    m, n = fea.shape
    for i in range(m):
        doc = []
        for j in range(n):
            if fea[i,j] == 1:
                doc.append((j,1))
        bow.append(doc)
    return bow

def LDA_load(dataset):
    return np.load('dataset/'+dataset+'/lda_fea.npy')