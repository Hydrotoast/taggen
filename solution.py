import pickle
import numpy as np

from sklearn import cross_validation
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

def print_accuracy(name, scores):
	print "Mean F1-Score: %0.4f (+/- %0.2f) using %s" % (scores.mean(), scores.std() * 2, name)

# load data
# train = pickle.load(open('train.dat', 'r'))
# test = pickle.load(open('test.dat', 'r'))
from sklearn import datasets
iris = datasets.load_iris()
train = iris.data
test = iris.target

# estimators
gnb = GaussianNB()
mnb = MultinomialNB()
svc = LinearSVC()
rfc = RandomForestClassifier()

# pipelines
gnb_pipeline = Pipeline([('GaussianNB', gnb)])
mnb_pipeline = Pipeline([('GaussianNB', mnb)])
svc_pipeline = Pipeline([('LinearSVC', svc)])
rfc_pipeline = Pipeline([('RandomForestClassifier', rfc)])

# cross validated scores
gnb_scores = cross_validation.cross_val_score(gnb_pipeline, train, test, scoring='f1', cv=5, n_jobs=-1)
mnb_scores = cross_validation.cross_val_score(mnb_pipeline, train, test, scoring='f1', cv=5, n_jobs=-1)
svc_scores = cross_validation.cross_val_score(svc_pipeline, train, test, scoring='f1', cv=5, n_jobs=-1)
rfc_scores = cross_validation.cross_val_score(rfc_pipeline, train, test, scoring='f1', cv=5, n_jobs=-1)

# print scores
print_accuracy('Gaussian NB', gnb_scores)
print_accuracy('Multinomial NB', mnb_scores)
print_accuracy('LinearSVC', svc_scores)
print_accuracy('RandomForestClassifier', rfc_scores)
