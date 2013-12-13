import config
import preprocess

import csv
import os
from operator import itemgetter
from functools import partial
from multiprocessing.pool import Pool

import pickle
import numpy as np

from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

from sklearn.decomposition import pca

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

config.FEATURES_DIR = os.path.join(config.FEATURES_DIR, 'tfidf_transformer')
config.RESULTS_DIR = os.path.join(config.RESULTS_DIR, 'tfidf_transformer')

def model_cross_val_score(model):
    for parameters, mean_validation_score, cv_validation_scores in model.grid_scores_:
        if parameters == model.best_params_:
            return cv_validation_scores

def print_accuracy(name, scores):
    print "Mean F1-Score: %0.4f (+/- %0.2f) using %s" % (scores.mean(), scores.std() * 2, name)

def analyze_tag(train, target, tag_name):
    if os.path.exists(os.path.join(config.RESULTS_DIR, tag_name + '.dat')):
        print 'loading previous results for tag: %s' % tag_name
        return pickle.load(open(os.path.join(config.RESULTS_DIR, tag_name + '.dat'), 'r'))

    # preprocessors
    pca_pp = pca.PCA(n_components=40)

    # estimators
    mnb = MultinomialNB()
    # svc = LinearSVC(dual=False)
    # rfc = RandomForestClassifier()
    # gbc = GradientBoostingClassifier()

    # pipelines
    mnb_pipeline = Pipeline([('pca', pca_pp), ('clf', mnb)])
    # svc_pipeline = Pipeline([('pca', pca_pp), ('clf', svc)])
    # rfc_pipeline = Pipeline([('pca', pca_pp), ('clf', rfc)])
    # gbc_pipeline = Pipeline([('pca', pca_pp), ('clf', gbc)])

    # K-Fold cross-validation strategy
    skf = cross_validation.StratifiedKFold(target, n_folds=3)

    # parameter grids
    mnb_param_grid = {} # there are no parameters for Multinomial Naive Bayes
    # svc_param_grid = dict(clf__C=10 ** np.arange(0, 9))
    # rfc_param_grid = dict(
    #     clf__n_estimators=[10, 20, 30],
    #     clf__criterion=['gini', 'entropy'],
    #     clf__max_features=['sqrt', 'log2'],
    #     clf__min_samples_split=[1, 2, 3])
    # gbc_param_grid = dict(
    #     clf__n_estimators=[100, 200, 300],
    #     clf__max_features=['sqrt', 'log2'],
    #     clf__min_samples_split=[1, 2, 3])

    # hyperparameter optimization
    mnb_model = GridSearchCV(mnb_pipeline, mnb_param_grid, scoring='f1', cv=skf, n_jobs=4).fit(train, target)
    # svc_model = GridSearchCV(svc_pipeline, svc_param_grid, scoring='f1', cv=skf, n_jobs=4).fit(train, target)
    # rfc_model = GridSearchCV(rfc_pipeline, rfc_param_grid, scoring='f1', cv=skf, n_jobs=4).fit(train, target)
    # gbc_model = GridSearchCV(gbc_pipeline, gbc_param_grid, scoring='f1', cv=skf, n_jobs=4).fit(train, target)

    # cross validated scores
    mnb_scores = model_cross_val_score(mnb_model)
    # svc_scores = model_cross_val_score(svc_model)
    # rfc_scores = model_cross_val_score(rfc_model)
    # gbc_scores = model_cross_val_score(gbc_model)

    # print scores
    print 'surveying estimators for tag: %s' % tag_name
    print_accuracy('Multinomial NB', mnb_scores)
    # print_accuracy('LinearSVC', svc_scores)
    # print_accuracy('RandomForestClassifier', rfc_scores)
    # print_accuracy('GradientBoostingClassifier', gbc_scores)

    # find the best estimator and train it
    best_estimator = max(
        [
            (mnb_scores.mean(), mnb_model),
            # (svc_scores.mean(), svc_model),
            # (rfc_scores.mean(), rfc_model),
            # (gbc_scores.mean(), gbc_model)
        ],
        key=itemgetter(0)
    )[1]

    # cache the estimator
    pickle.dump(best_estimator, open(os.path.join(config.ESTIMATORS_DIR, tag_name), 'w'))

    # return a list to be used as a CSV row
    result_row = [tag_name, mnb_scores.mean()] #, svc_scores.mean(), rfc_scores.mean(), gbc_scores.mean()]
    pickle.dump(result_row, open(os.path.join(config.RESULTS_DIR, tag_name + '.dat'), 'w'))
    return result_row

@preprocess.timed
def analyze_tag_task(feature_file):
    # load data
    fh = open(os.path.join(config.FEATURES_DIR, feature_file), 'r')
    feature = pickle.load(fh)
    fh.close()
    train = feature['data'].toarray()
    target = np.array(feature['target'])

    # analyze the data!
    results = analyze_tag(train, target, os.path.splitext(os.path.basename(feature_file))[0])
    return results

def cache_results(results):
    results_file = os.path.join(config.RESULTS_DIR, config.ESTIMATOR_RESULTS)
    if os.path.exists(results_file):
        print 'results file %s already exists' % results_file
        return
    print 'writing to results file: %s' % results_file
    pickle.dump(results, open(os.path.join(config.CACHE_DIR, 'results.dat'), 'w'))
    with open(results_file, 'w') as fh:
        writer = csv.writer(fh)
        writer.writerow(['tag_name', 'mnb_scores_mean', 'svc_scores_mean', 'rfc_scores_mean', 'gbc_scores_mean'])
        for result in results:
            writer.writerow(result)

@preprocess.timed
def analyze_tags_parallel(selected_tags=None):
    ls = os.listdir(config.FEATURES_DIR) if selected_tags is None else selected_tags
    pool = Pool(processes=4)
    results = pool.map(analyze_tag_task, ls)
    cache_results(results)

@preprocess.timed
def analyze_tags(selected_tags=None):
    ls = os.listdir(config.FEATURES_DIR) if selected_tags is None else selected_tags
    results = map(analyze_tag_task, ls)
    cache_results(results)

if __name__ == '__main__':
    selected_tags = [
        'python', 'java', 'c++', 'c', 'ruby', 'haskell', # popular languages
        'codeigniter', 'spring', 'sqlalchemy', 'oauth', # popular frameworks
        'mysql', 'oracle', 'postgresql', 'sqlite', # databases
        'ubuntu', 'debian', 'centos', 'osx', 'windows-7' # operating systems
    ]
    selected_tags = map(lambda x: x + '.data', selected_tags)
    analyze_tags(selected_tags)
