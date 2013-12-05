import config
import preprocess

import csv
import os
from operator import itemgetter
from multiprocessing.pool import Pool

import pickle
import numpy as np

from sklearn import cross_validation
from sklearn.pipeline import Pipeline

from sklearn.decomposition import pca

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def print_accuracy(name, scores):
    print "Mean F1-Score: %0.4f (+/- %0.2f) using %s" % (scores.mean(), scores.std() * 2, name)

def analyze_tag(train, target, tag_name):
    # preprocessors
    pca_pp = pca.PCA(n_components=40)

    # estimators
    gnb = GaussianNB()
    svc = LinearSVC()
    rfc = RandomForestClassifier()
    gbc = GradientBoostingClassifier()

    # pipelines
    gnb_pipeline = Pipeline([('PCA', pca_pp), ('GaussianNB', gnb)])
    svc_pipeline = Pipeline([('PCA', pca_pp), ('LinearSVC', svc)])
    rfc_pipeline = Pipeline([('PCA', pca_pp), ('RandomForestClassifier', rfc)])
    gbc_pipeline = Pipeline([('PCA', pca_pp), ('GradientBoostingClassifier', gbc)])

    skf = cross_validation.StratifiedKFold(target, n_folds=3)

    # cross validated scores
    gnb_scores = cross_validation.cross_val_score(gnb_pipeline, train, target, scoring='f1', cv=skf, n_jobs=1)
    svc_scores = cross_validation.cross_val_score(svc_pipeline, train, target, scoring='f1', cv=skf, n_jobs=1)
    rfc_scores = cross_validation.cross_val_score(rfc_pipeline, train, target, scoring='f1', cv=skf, n_jobs=1)
    gbc_scores = cross_validation.cross_val_score(gbc_pipeline, train, target, scoring='f1', cv=skf, n_jobs=1)

    # print scores
    print 'surveying estimators for tag: %s' % feature_file
    print_accuracy('Gaussian NB', gnb_scores)
    print_accuracy('LinearSVC', svc_scores)
    print_accuracy('RandomForestClassifier', rfc_scores)
    print_accuracy('GradientBoostingClassifier', gbc_scores)

    # find the best estimator and train it
    best_estimator = max(
        [
            (gnb_scores.mean(), gnb_pipeline),
            (svc_scores.mean(), svc_pipeline),
            (rfc_scores.mean(), rfc_pipeline),
            (gbc_scores.mean(), gbc_pipeline)
        ],
        key=itemgetter(0)
    )[1]
    best_estimator.fit(train, target)

    # cache the estimator
    pickle.dump(best_estimator, open(os.path.join(config.ESTIMATORS_DIR, tag_name), 'w'))

    # return a list to be used as a CSV row
    result_row = [tag_name, gnb_scores.mean(), svc_scores.mean(), rfc_scores.mean()]
    pickle.dump(result_row, open(os.path.join(config.RESULTS_DIR, tag_name, '.dat'), 'w'))
    return result_row
__ikj
def analyze_tag_task(feature_file):
    # load data
    feature = pickle.load(open(os.path.join(config.FEATURES_DIR, feature_file), 'r'))
    train = feature['data'].toarray()
    target = np.array(feature['target'])

    # analyze the data!
    analyze_tag(train, target, os.path.splitext(os.path.basename(feature_file))[0])

def cache_results(results):
    results_file = os.path.join(config.RESULTS_DIR, config.ESTIMATOR_RESULTS, 'w')
    if os.path.exists(results_file):
        print 'results file %s already exists' % results_file
        return
    print 'writing to results file: %s' % results_file
    with open(results_file, 'w') as fh:
        writer = csv.writer(fh)
        writer.writerow(['tag_name', 'gnb_scores_mean', 'svc_scores_mean', 'rfc_scores_mean'])
        for result in results:
            writer.writerow(result)

@preprocess.timed
def analyze_tags():
    ls = os.listdir(config.FEATURES_DIR)
    pool = Pool(processes=4)
    results = pool.map(analyze_tag_task, ls)
    cache_results(results)

if __name__ == '__main__':
    analyze_tags()
