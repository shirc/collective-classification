__author__ = 'SRC'

import evaluation
import cc
import dataset
from sklearn import svm
import topic_modeling
import community_detection
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.linear_model.logistic import LogisticRegression

def setup(name, train_ratio=0.1):
    dataset.process_dataset(name)
    evaluation.prepare_cv(name, train_ratio)

def test(name):
    classifier = svm.LinearSVC()
    knn_classifier = neighbors.KNeighborsClassifier(n_neighbors = 5)
    # classifier = LogisticRegression()
    LTM_t_fea = topic_modeling.load_t_fea(name, 'LTM', 20)
    LDA_t_fea = topic_modeling.load_t_fea(name, 'LDA', 20)
    results = []
    # results.append({'name':'topical_CR','errors':evaluation.evaluate(name, cc.topical_CR(classifier,LTM_t_fea))})
    # results.append({'name':'ContentOnly','errors':evaluation.evaluate(name, cc.CO(classifier))})
    # results.append({'name':'wvRN_RL','errors':evaluation.evaluate(name, cc.wvRN_RL())})
    # results.append({'name':'topical_CO+LDA','errors':evaluation.evaluate(name, cc.topical_CO(classifier, LDA_t_fea))})
    results.append({'name':'topical_CO+LTM','errors':evaluation.evaluate(name, cc.topical_CO(classifier, LTM_t_fea))})
    results.append({'name':'topical_CO+LTM','errors':evaluation.evaluate(name, cc.topical_CO(knn_classifier, LTM_t_fea))})
    # results.append({'name':'ICA','errors':evaluation.evaluate(name, cc.ICA(classifier))})
    # results.append({'name':'semi_ICA','errors':evaluation.evaluate(name, cc.semi_ICA(classifier))})
    return results

def save_result(name, results):
    f = open('benchmark/result/'+name+'.csv', 'w')
    for result in results:
        row = []
        row.append(result['name'])
        row.append(str(np.mean(result['errors'])))
        row.append(str(np.std(result['errors'])))
        for error in result['errors']:
            row.append(str(error))
        f.write(','.join(row)+'\n')
    f.close()

def visualize(name):
    fea, link, label = dataset.load_dataset(name)
    label = np.argmax(label, axis=1)
    label = label.astype('float')
    label = label + 1
    label = label / max(label)
    link = link.tocsc()
    g = nx.Graph(link)
    nx.draw_networkx(g, node_size=100, with_labels=False, node_color=label)
    plt.show()

def stats(name):
    fea, link, label = dataset.load_dataset(name)
    g = nx.Graph(link)
    components = nx.connected_components(g)
    num_node = link.shape[0]
    num_link = link.sum() / 2
    density =  float(2*num_link) / num_node
    ratio = float(len(components[0])) / link.shape[0]
    row, col = link.nonzero()
    label = np.argmax(label, axis=1)
    homogeneity = float((label[row] == label[col]).sum()) / len(row)
    info = {'name':name, 'ratio':ratio, 'homogeneity':homogeneity, 'num_node':num_node, 'num_link': num_link, 'density':density}
    print info

def save_synthetic_result(results):
    f = open('benchmark/result/synthetic.csv', 'w')
    keys = ['ld','dh','ap','algorithm', 'mean','std']
    f.write(','.join(keys)+'\n')
    for result in results:
        row = []
        for k in keys:
            row.append(str(result[k]))
        f.write(','.join(row)+'\n')
    f.close()

if __name__ == '__main__':
    for name in ['cora']:
        # evaluation.prepare_cv(name)
        results = test(name)
        # save_result(name, results)