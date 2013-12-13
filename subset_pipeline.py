import config
from preprocess import timed

import os
import csv
import pickle
from operator import itemgetter

import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer, HashingVectorizer
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

from sklearn.decomposition.truncated_svd import TruncatedSVD

from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

STOPWORDS_FILE = os.path.join(config.CACHE_DIR, "stopwords.txt")
STOPWORDS_DAT = os.path.join(config.CACHE_DIR, "stopwords.dat")

config.RESULTS_DIR = os.path.join(config.RESULTS_DIR, 'latent_semantic_analysis')

@timed
def cached_stopwords(stop_file=STOPWORDS_FILE, dump_file=STOPWORDS_DAT):
    if os.path.isfile(dump_file):
        return pickle.load(open(dump_file, "r"))
    with open(stop_file, "r") as swf:
        stopwords = []
        for line in swf:
            stopwords.append(line.strip())
    pickle.dump(stopwords, open(dump_file, "w"))
    return stopwords

def cache_results(results):
    results_file = os.path.join(config.RESULTS_DIR, config.ESTIMATOR_RESULTS)
    if os.path.exists(results_file):
        print 'results file %s already exists' % results_file
        return
    print 'writing to results file: %s' % results_file
    pickle.dump(results, open(os.path.join(config.CACHE_DIR, 'results.dat'), 'w'))
    with open(results_file, 'w') as fh:
        writer = csv.writer(fh)
        writer.writerow(['tag_name', 'bnb_scores_mean', 'svc_scores_mean', 'rfc_scores_mean', 'gbc_scores_mean'])
        for result in results:
            writer.writerow(result)


@timed
def analyze_subset_pipelines():
    selected_tags = [
        'codeigniter', 'spring', 'sqlalchemy', 'oauth', # popular frameworks
        'mysql', 'oracle', 'postgresql', 'sqlite', # databases
        'ubuntu', 'debian', 'centos', 'osx', 'windows-7', # operating systems
        'python', 'java', 'c++', 'c', 'ruby', 'haskell' # popular languages
    ]
    results = map(analyze_subset_pipeline, selected_tags)
    cache_results(results)

def analyze_subset_pipeline(tag_name):
    print "featurizing tag sample for: %s" % tag_name
    train, target = sample_features(tag_name)
    return analyze_tag(train, target, tag_name)

def sample_features(tag_name):
    with open(os.path.join(config.SAMPLES_DIR, tag_name), "r") as tsf:
        rd = csv.reader(tsf)
        rd.next() # skip headers

        train = []
        target = []
        for line in rd:
            train.append(line[1] + line[2])
            target.append(line[4] == "True")
    return train, target

def print_accuracy(name, scores):
    print "Mean F1-Score: %0.4f (+/- %0.2f) using %s" % (scores.mean(), scores.std() * 2, name)

def model_cross_val_score(model):
    for parameters, mean_validation_score, cv_validation_scores in model.grid_scores_:
        if parameters == model.best_params_:
            return cv_validation_scores

def analyze_tag(train, target, tag_name, stop_words=cached_stopwords()):
    if os.path.exists(os.path.join(config.RESULTS_DIR, tag_name + '.dat')):
        print 'loading previous results for tag: %s' % tag_name
        return pickle.load(open(os.path.join(config.RESULTS_DIR, tag_name + '.dat'), 'r'))

    # vectorize
    hasher = HashingVectorizer(stop_words=stop_words, non_negative=True, norm=None, n_features=2**12)
    vectorizer = TfidfTransformer()
    svd = TruncatedSVD(n_components=2**8)
    transformers = Pipeline([('hasher', hasher), ('vec', vectorizer), ('reduce_dim', svd)])
    train = transformers.fit_transform(train, target)

    # estimators
    bnb = BernoulliNB()
    svc = LinearSVC(dual=False)
    rfc = RandomForestClassifier()
    gbc = GradientBoostingClassifier()

    # pipelines
    bnb_pipeline = Pipeline([('clf', bnb)])
    svc_pipeline = Pipeline([('clf', svc)])
    rfc_pipeline = Pipeline([('clf', rfc)])
    gbc_pipeline = Pipeline([('clf', gbc)])

    # K-Fold cross-validation strategy
    skf = cross_validation.StratifiedKFold(target, n_folds=3)

    # parameter grids
    bnb_param_grid = dict(clf__alpha=[0,1.0,2.0])
    svc_param_grid = dict(clf__C=10 ** np.arange(0, 9))
    rfc_param_grid = dict(
        clf__n_estimators=[10, 20, 30],
        clf__criterion=['gini', 'entropy'],
        clf__max_features=['sqrt', 'log2'],
        clf__min_samples_split=[1, 2, 3])
    gbc_param_grid = dict(
        clf__n_estimators=[100, 200, 300],
        clf__max_features=['sqrt', 'log2'],
        clf__min_samples_split=[1, 2, 3])

    # hyperparameter optimization
    bnb_model = GridSearchCV(bnb_pipeline, bnb_param_grid, scoring='f1', cv=skf, n_jobs=4).fit(train, target)
    svc_model = GridSearchCV(svc_pipeline, svc_param_grid, scoring='f1', cv=skf, n_jobs=4).fit(train, target)
    rfc_model = GridSearchCV(rfc_pipeline, rfc_param_grid, scoring='f1', cv=skf, n_jobs=4).fit(train, target)
    gbc_model = GridSearchCV(gbc_pipeline, gbc_param_grid, scoring='f1', cv=skf, n_jobs=4).fit(train, target)

    # cross validated scores
    bnb_scores = model_cross_val_score(bnb_model)
    svc_scores = model_cross_val_score(svc_model)
    rfc_scores = model_cross_val_score(rfc_model)
    gbc_scores = model_cross_val_score(gbc_model)

    # print scores
    print 'surveying estimators for tag: %s' % tag_name
    print_accuracy('Bernoulli NB', bnb_scores)
    print_accuracy('Linear SVC', svc_scores)
    print_accuracy('Random Forest Classifier', rfc_scores)
    print_accuracy('Gradient Boosting Classifier', gbc_scores)

    # find the best estimator and train it
    best_estimator = max(
        [
            (bnb_scores.mean(), bnb_model),
            (svc_scores.mean(), svc_model),
            (rfc_scores.mean(), rfc_model),
            (gbc_scores.mean(), gbc_model)
        ],
        key=itemgetter(0)
    )[1]

    # cache the estimator
    pickle.dump({'estimator': best_estimator, 'vectorizer': transformers}, open(os.path.join(config.ESTIMATORS_DIR, tag_name), 'w'))

    # return a list to be used as a CSV row
    result_row = [tag_name, bnb_scores.mean(), svc_scores.mean(), rfc_scores.mean(), gbc_scores.mean()]
    pickle.dump(result_row, open(os.path.join(config.RESULTS_DIR, tag_name + '.dat'), 'w'))
    return result_row

if __name__ == '__main__':
    analyze_subset_pipelines()

