__author__ = 'SRC'

from sklearn.cross_validation import KFold
from dataset import load_dataset
import pickle
import numpy as np
import topic_modeling

def prepare_cv(dataset, k):
    fea, link, label = load_dataset(dataset)
    kfold = KFold(fea.shape[0], n_folds=k, indices=False)
    pickle.dump(kfold, open('benchmark/cv/'+dataset,'wb'))

def load_cv(dataset):
    kfold = pickle.load(open('benchmark/cv/'+dataset, 'rb'))
    return kfold

def evaluate_base(fea, link, label, kfold, model):
    tmp_label = label.copy()
    for train, test in kfold:
        tmp_label[test,:] = 0
        tmp_label = model.fit_predict(fea, link, train, tmp_label)
        error = np.abs(tmp_label[test, :]-label[test, :]).sum() / 2 / tmp_label.shape[0]
        print error

def evaluate(dataset, model):
    kfold = load_cv(dataset)
    fea, link, label = load_dataset(dataset)
    evaluate_base(fea, link, label, kfold, model)

def evaluate_lda(dataset, model):
    kfold = load_cv(dataset)
    fea, link, label = load_dataset(dataset)
    lad_fea = topic_modeling.LDA_load(dataset)
    evaluate_base(lad_fea, link, label, kfold, model)