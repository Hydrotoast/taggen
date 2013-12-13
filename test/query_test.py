import query
import unittest
import config
import pickle

class QueryTest(unittest.TestCase):
    def test_sanity(self):
        #result = query.query("hello, world!", feature_dir="../" + config.FEATURES_DIR)
        #pickle.dump(result, open("test_query_result.dat", "w"))
        #print result
        pass

    def test_query(self):
        query.query_csvfile("test_query.csv", "test_query.out", feature_dir=config.FEATURES_DIR, dump="test_query.dat")
