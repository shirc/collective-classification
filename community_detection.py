__author__ = 'SRC'

import networkx as nx
import community
import numpy as np
import dataset
import scipy.stats as stats

def get_c_fea(name):
    fea, link, label = dataset.load_dataset(name)
    num_inst = link.shape[0]
    g = nx.Graph(link)
    partition = community.best_partition(g)
    communities = partition.values()
    loc_fea = np.zeros((num_inst, max(communities)+1))
    for i, v in enumerate(communities):
        loc_fea[i, v] = 1
    return loc_fea

def get_c_fea_w(weight):
    num_inst = weight.shape[0]
    g = nx.Graph(weight)
    partition = community.best_partition(g)
    communities = partition.values()
    loc_fea = np.zeros((num_inst, max(communities)+1))
    for i, v in enumerate(communities):
        loc_fea[i, v] = 1
    return loc_fea

def community_label_entropy(name):
    fea, link, label = dataset.load_dataset(name)
    c_fea = get_c_fea(name)
    cl = c_fea.transpose().dot(label)
    l = cl.shape[0]
    entropy = []
    for i in range(l):
        x = cl[i,:]
        entropy.append(stats.entropy(x[x.nonzero()]))
    return np.mean(entropy)

if __name__ == '__main__':
    print community_label_entropy('cora')
    print community_label_entropy('citeseer')