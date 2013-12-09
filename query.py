
import config
import preprocess
import csv
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

def query_csvfile(query_file, output_file, estimator_dir=config.ESTIMATORS_DIR,
                  feature_dir=config.FEATURES_DIR, dump=None):

    if dump and os.path.isfile(dump):
        result = pickle.load(open(dump, "r"))
        with open(output_file, "w") as of:
            for i, row in enumerate(result):
                of.write("---------%d----------\n" % i)
                s = [(v, t) for t, v in row.items()]
                for j, t in enumerate(sorted(s, reverse=True)):
                    of.write("%d : %s\n" % (j, t))
    else:
        with open(query_file, "r") as qf, open(output_file, "w") as of:
            result = []
            rd = csv.reader(qf)
            rd.next()

            for i, row in enumerate(rd):
                raw = preprocess.format_input(preprocess.parse_doc(row))
                result.append(query(raw, estimator_dir, feature_dir))
                if dump:
                    pickle.dump(result, open(dump, "w"))
                for j, tag in enumerate(sorted(result[i], reverse=True)):
                    of.write("%s, %f\n" % (tag, result[i][tag]))


def query(text, estimator_dir, feature_dir):

    print "querying with text : \n%s\n" % text
    edir = os.listdir(estimator_dir)

    result = {}
    for epath in edir:

        print "loading estimator from path: %s" % epath
        tag = epath
        # get the vocabulary associated with the estimator from the feature data
        fpath = os.path.join(feature_dir, tag + ".data")
        fdata = pickle.load(open(fpath, "r"))

        # vectorize the query with estimators vocabulary
        cv = CountVectorizer(vocabulary=fdata["vocabulary"])
        X = cv.fit_transform(text)

        # get the estimator
        #assert isinstance(e, LinearSVC)
        e = pickle.load(open(os.path.join(config.ESTIMATORS_DIR, epath), "r"))
        result[tag] = e.decision_function(X.toarray())[0]
        print "'%s': %f" % (tag, result[tag])
    return result

if __name__ == "__main__":
    query_csvfile()