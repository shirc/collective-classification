__author__ = 'SRC'

import evaluation
import cc
import dataset
from sklearn import svm
import topic_modeling

if __name__ == '__main__':
    topic_modeling.LDA_process('cora')
    # dataset.process_dataset('citeseer')
    # evaluation.prepare_cv('citeseer', 2)
    # evaluation.evaluate('cora', cc.wvRN_RL())
    # dataset.process_dataset('cora')
    # classifier = svm.LinearSVC()
    # evaluation.evaluate('citeseer', cc.ICA(classifier=classifier))
    # evaluation.evaluate('cora', cc.ContentOnly(classifier=classifier))
    classifier = svm.LinearSVC()
    evaluation.evaluate_lda('cora', cc.ContentOnly(classifier=classifier))
    # evaluation.evaluate('citeseer', cc.wvRN_RL())