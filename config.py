
from os.path import split
PATH=split(__file__)[0]
'''**original training and test dataset locations**'''
TRAIN_FILE='/host/Users/WORKBOOK/My Projects/class/cs175/taggen/data/Train.csv'
TEST_FILE='/host/Users/WORKBOOK/My Projects/class/cs175/taggen/data/Test.csv'
CORPUS_SIZE=6034195

'''sample files storage directory'''
SAMPLES_DIR='samples'

'''generated estimators storage directory'''
ESTIMATORS_DIR=PATH+'/estimators'
ESTIMATOR_RESULTS=PATH+'/estimator_results.csv'
FEATURES_DIR=PATH+'/features'
#COUNT_VECTORIZER=PATH+'/count_vectorizer'
#TFIDF_TRANSFORMER=PATH+'/tfidf_transformer'

'''intermediary result storage directory '''
CACHE_DIR='/host/Users/WORKBOOK/My Projects/class/cs175/taggen/cache'
#CACHE_DIR=PATH+'/cache'
CLEAN_CSV_FILE = CACHE_DIR + '/train_clean.csv'
CLEAN_DAT_FILE = CACHE_DIR + '/train_clean.dat'
TAG_RINDEX_FILE = CACHE_DIR + '/tag_index.dat'
STOPWORDS_FILE = CACHE_DIR + '/stopwords.txt'
STOPWORDS_DAT = CACHE_DIR + '/stopwords.dat'

RESULTS_DIR=PATH+'/results'
RESULTS_DIR_LSA = RESULTS_DIR+'/latent_semantic_analysis'

DEMO_DIR = PATH+'/demo'

if __name__ == '__main__':
    import os
    if os.path.isfile("config.py"):
        if not os.path.isdir(SAMPLES_DIR): os.makedirs(SAMPLES_DIR)
        if not os.path.isdir(ESTIMATORS_DIR): os.makedirs(ESTIMATORS_DIR)
        if not os.path.isdir(FEATURES_DIR): os.makedirs(FEATURES_DIR)
        #os.makedirs(FEATURES_DIR+"/"+COUNT_VECTORIZER)
        #os.makedirs(FEATURES_DIR+"/"+TFIDF_TRANSFORMER)
        if not os.path.isdir(CACHE_DIR): os.makedirs(CACHE_DIR)
        if not os.path.isdir(RESULTS_DIR): os.makedirs(RESULTS_DIR)
        if not os.path.isdir(DEMO_DIR): os.makedirs(DEMO_DIR)