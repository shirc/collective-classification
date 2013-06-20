__author__ = 'SRC'

import numpy as np
from sklearn import preprocessing
import scipy.sparse as sparse
from scipy.io import mmwrite, mmread


def process_dataset(name):
    prefix = 'dataset/' + name + '/' + name
    fea = np.loadtxt(prefix + '.fea')
    # transform link
    link_data = np.loadtxt(prefix + '.link')
    link_data = link_data - 1
    reverse_link_data = np.append(link_data[:, 1][:,np.newaxis], link_data[:, 0][:,np.newaxis], axis=1)
    link_data = np.append(link_data, reverse_link_data, axis=0)
    weight = [1]*link_data.shape[0]
    num_inst = fea.shape[0]
    link = sparse.csr_matrix((weight, link_data.transpose()), shape=(num_inst, num_inst))
    # transform label
    gnd = np.loadtxt(prefix + '.gnd')
    lb = preprocessing.LabelBinarizer()
    label = lb.fit_transform(gnd)
    label = label.astype(np.float)
    # save
    np.save(prefix + '_fea', fea)
    mmwrite(prefix + '_link', link)
    np.save(prefix + '_label', label)
    return fea, link, label

def load_dataset(name):
    prefix = 'dataset/' + name + '/' + name
    fea = np.load(prefix + '_fea.npy')
    link = mmread(prefix + '_link.mtx')
    label = np.load(prefix + '_label.npy')
    return fea, link, label

if __name__ == '__main__':
    # fea, link, label = process_dataset('cora')
    load_dataset('cora')