__author__ = 'SRC'

import numpy as np
from sklearn import preprocessing
import scipy.sparse as sparse
from scipy.io import mmwrite, mmread
import networkx as nx
import numpy.random as random
from sklearn.cross_validation import ShuffleSplit

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
    # use max component
    g = nx.Graph(link)
    mc = nx.connected_components(g)[0]
    link = link[mc, :][:, mc]
    label = label[mc, :]
    fea = fea[mc, :]
    # save
    np.save(prefix + '_fea', fea)
    mmwrite(prefix + '_link', link)
    np.save(prefix + '_label', label)
    np.save(prefix + '_mc', mc)
    return fea, link, label

def load_dataset(name):
    prefix = 'dataset/' + name + '/' + name
    fea = np.load(prefix + '_fea.npy')
    link = mmread(prefix + '_link.mtx')
    label = np.load(prefix + '_label.npy')
    return fea, link, label

class Generator(object):
    def generate(self, num_inst=250, num_attr=10, num_label=5, ap=0.6, ld=0.2, dh=0.7):
        self.total = 0
        self.label = np.zeros(num_inst)
        self.link = np.zeros((num_inst, num_inst))
        self.fea = np.zeros((num_inst, num_attr))
        while self.total < num_inst:
            ld_r = random.random()
            if ld_r < ld and self.total >= 2:
                s = random.randint(0, self.total)
                self.connect(s, dh, num_label)
            else:
                s = self.total
                self.add_node(num_attr, num_label, ap)
                self.connect(s, dh, num_label)
        lb = preprocessing.LabelBinarizer()
        self.label = lb.fit_transform(self.label)
        self.label = self.label.astype(np.float)
        cv = ShuffleSplit(self.fea.shape[0], 10, test_size=0.9, indices=False, random_state=0)
        self.link = sparse.csr_matrix(self.link)
        return self.fea, self.link, self.label, cv


    def add_node(self, num_attr, num_label, ap):
        l = random.randint(num_label)
        self.label[self.total] = l
        for i in range(num_attr):
            if l == i % num_label:
                p = 0.15+(ap - 0.15)*i/(num_attr-1)
            elif l == (i-1) % num_label:
                p = 0.1
            elif l == (i+1) % num_label:
                p = 0.05
            else:
                p = 0.02
            p_r = random.random()
            if p_r < p:
                v = 1
            else:
                v = 0
            self.fea[self.total, i] = v
        self.total += 1


    def connect(self, s, dh, num_label):
        dh_r = random.random()
        if dh_r < dh:
            l = self.label[s]
        else:
            l = -1
        degree = np.sum(self.link, axis=1)
        degree = degree[:self.total]
        total_degree = float(sum(degree)+len(degree))
        ts = random.choice(range(self.total), size=self.total, replace=False, p=(degree+1)/total_degree)
        for t in ts:
            if (self.label[t] == l or l == -1) and s != t and self.link[s, t] == 0:
                self.link[s, t] = 1
                self.link[t, s] = 1
                break

if __name__ == '__main__':
    # fea, link, label = process_dataset('cora')
    # fea, link, label = load_dataset('cora')
    # print fea.shape[0]
    # generator = Generator()
    # fea, link, label, cv = generator.generate(num_inst=1000, num_attr=100, ld=0.5, ap=0.5)
    # np.savetxt('dataset/synthetic/synthetic.fea', fea, fmt='%d')
    # nz = link.nonzero()
    # link = np.concatenate((nz[0][:,np.newaxis]+1, nz[1][:,np.newaxis]+1), axis=1)
    # link.astype(dtype='int')
    # np.savetxt('dataset/synthetic/synthetic.link', link, fmt='%d')
    # label = np.argmax(label, axis=1)
    # label += 1
    # np.savetxt('dataset/synthetic/synthetic.gnd', label, fmt='%d')
    process_dataset('cora')
    process_dataset('citeseer')