__author__ = 'SRC'

from sklearn.cross_validation import ShuffleSplit
from dataset import load_dataset, Generator
import pickle
import numpy as np
import topic_modeling

def prepare_cv(dataset, train_ratio = 0.1):
    fea, link, label = load_dataset(dataset)
    cv = ShuffleSplit(fea.shape[0], 10, test_size=1-train_ratio, indices=False, random_state=0)
    pickle.dump(cv, open('benchmark/cv/'+dataset,'wb'))

def load_cv(dataset):
    cv = pickle.load(open('benchmark/cv/'+dataset, 'rb'))
    return cv

def evaluate(dataset, model):
    kfold = load_cv(dataset)
    fea, link, label = load_dataset(dataset)
    errors = []
    for train, test in kfold:
        tmp_label = label.copy()
        tmp_label[test,:] = 0
        tmp_label = model.fit_predict(fea, link, train, tmp_label)
        error = np.abs(tmp_label[test, :]-label[test, :]).sum() / 2 / tmp_label.shape[0]
        errors.append(error)
        print error
    print 'mean', np.mean(errors)
    return errors

def evaluate_synthetic(ld, dh, ap, model):
    fea, link, label, cv = Generator().generate(ld=ld, dh=dh, ap=ap)
    errors = []
    for train, test in cv:
        tmp_label = label.copy()
        tmp_label[test,:] = 0
        tmp_label = model.fit_predict(fea, link, train, tmp_label)
        error = np.abs(tmp_label[test, :]-label[test, :]).sum() / 2 / tmp_label.shape[0]
        errors.append(error)
    result = {'ld':ld,'dh':dh,'ap':ap,'algorithm':model.__class__.__name__,'mean':np.mean(errors),'std':np.std(errors)}
    print result
    return result
