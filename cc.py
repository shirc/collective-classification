__author__ = 'SRC'
import numpy as np
from networkx.algorithms.community import k_clique_communities
import networkx as nx
from sklearn.svm import LinearSVC
import community
from sklearn.lda import LDA
import scipy.spatial.distance as dist
import community_detection as cdt
import scipy.sparse as sparse

class ICA(object):
    def __init__(self, classifier, iterate_num=10):
        self.classifier = classifier
        self.iterate_num = iterate_num

    def decode_label(self, label, num_label):
        lm = np.zeros((len(label), num_label))
        for i in range(len(label)):
            lm[i, label[i]] = 1
        return lm

    def encode_label(self, lm):
        return np.argmax(lm, axis=1)

    def fit_predict(self, fea, link, trainInd, label):
        num_label = label.shape[1]
        testInd = ~trainInd
        rel_fea = link*label
        new_fea = np.append(fea, rel_fea, axis=1)
        self.classifier.fit(new_fea[trainInd], self.encode_label(label[trainInd]))
        for i in range(self.iterate_num):
            label[testInd] = self.decode_label(self.classifier.predict(new_fea[testInd]),num_label)
            rel_fea = link*label
            new_fea = np.append(fea, rel_fea, axis=1)
        return label

class semi_ICA(ICA):
    def fit_predict(self, fea, link, trainInd, label):
        num_label = label.shape[1]
        testInd = ~trainInd
        rel_fea = link*label
        new_fea = np.append(fea, rel_fea, axis=1)
        self.classifier.fit(new_fea[trainInd], self.encode_label(label[trainInd]))
        for i in range(self.iterate_num):
            label[testInd] = self.decode_label(self.classifier.predict(new_fea[testInd]),num_label)
            rel_fea = link*label
            new_fea = np.append(fea, rel_fea, axis=1)
            self.classifier.fit(new_fea[trainInd], self.encode_label(label[trainInd]))
            # self.classifier.fit(new_fea, self.encode_label(label))
        return label

class CO(object):
    def __init__(self, classifier):
        self.classifier = classifier

    def decode_label(self, label, num_label):
        lm = np.zeros((len(label), num_label))
        for i in range(len(label)):
            lm[i, label[i]] = 1
        return lm

    def encode_label(self, lm):
        return np.argmax(lm, axis=1)

    def fit_predict(self, fea, link, trainInd, label):
        num_label = label.shape[1]
        testInd = ~trainInd
        self.classifier.fit(fea[trainInd], self.encode_label(label[trainInd]))
        label[testInd] = self.decode_label(self.classifier.predict(fea[testInd]),num_label)
        return label

class wvRN_RL(object):
    def __init__(self,  iterate_num=100, gamma=0.99):
        self.iterate_num = iterate_num
        self.gamma = gamma

    def fit_predict(self, fea, link, trainInd, label,):
        testInd = ~trainInd
        prior = np.sum(label, 0) / np.sum(label)
        label[testInd, :] = prior
        for i in range(self.iterate_num):
            vote = link*label
            vote = vote / vote.sum(axis=1)[:, np.newaxis]
            label[testInd] = (1-self.gamma)*label[testInd] + self.gamma*vote[testInd]
        max_vote_label = np.argmax(vote, axis=1)
        label[testInd] = 0
        for j in range(len(max_vote_label)):
            if testInd[j]:
                label[j, max_vote_label[j]] = 1
        return label

class topical_CO(CO):
    def __init__(self, classifier, topic_fea):
        super(topical_CO, self).__init__(classifier)
        self.topic_fea = topic_fea

    def fit_predict(self, fea, link, trainInd, label):
        num_label = label.shape[1]
        testInd = ~trainInd
        # new_fea = np.concatenate((self.topic_fea), axis=1)
        self.classifier.fit(self.topic_fea[trainInd], self.encode_label(label[trainInd]))
        label[testInd] = self.decode_label(self.classifier.predict(self.topic_fea[testInd]),num_label)
        return label

class topical_CR(CO):
    def __init__(self, classifier, topic_fea):
        super(topical_CR, self).__init__(classifier)
        self.topic_fea = topic_fea

    def fit_predict(self, fea, link, trainInd, label):
        num_label = label.shape[1]
        testInd = ~trainInd
        topical_rel_fea = link*self.topic_fea
        new_fea = np.concatenate((fea, topical_rel_fea), axis=1)
        self.classifier.fit(new_fea[trainInd], self.encode_label(label[trainInd]))
        label[testInd] = self.decode_label(self.classifier.predict(new_fea[testInd]),num_label)
        return label