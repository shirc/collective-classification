__author__ = 'SRC'

from gensim import matutils
from dataset import load_dataset
from gensim import models
import numpy as np
from scipy.io import loadmat

def LDA_process(dataset):
    fea, link, label = load_dataset(dataset)
    corpus = matutils.Dense2Corpus(fea, documents_columns=False)
    num_topics = 100
    print 'performing lda...'
    model = models.LdaModel(corpus, num_topics=num_topics, passes=10)
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


def load_t_fea(name, t, n_topics):
    prefix = 'dataset/'+name+'/'+name
    mat = loadmat(prefix + '_'+t+'_'+str(n_topics)+'.mat')
    mc = np.load(prefix + '_mc.npy')
    return mat['t_fea'][mc, :]
    # return mat['t_fea']

def LDA_load(dataset):
    return np.load('dataset/'+dataset+'/lda_fea.npy')

if __name__ == '__main__':
    load_t_fea('cora', 'LTM', 50)