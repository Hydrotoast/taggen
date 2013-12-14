
__author__ = "Paul Kang"

import config
import preprocess
import csv
import os
import pickle


def query_csvfile(query_file, output_file, estimator_dir=config.ESTIMATORS_DIR):
    result = []
    with open(query_file, "r") as qf, open(output_file, "w") as of:
        wr = csv.writer(of)
        rd = csv.reader(qf)

        for i, row in enumerate(rd):
            raw = preprocess.format_input(preprocess.parse_doc(row))
            result.append(query(raw, estimator_dir))
            wr.writerow([int(row[0]), str(result[i])])
    return result


def query(text, estimator_dir):
    print "querying with text : \n%s\n" % text
    edir = os.listdir(estimator_dir)

    result = []
    for i, epath in enumerate(edir):

        print "%d: loading estimator from path: %s" % (i, epath)
        tag = epath

        e = pickle.load(open(os.path.join(config.ESTIMATORS_DIR, epath), "r"))
        X = e['vectorizer'].transform([text])
        if e['estimator'].predict(X)[0]:
            result.append(tag)
            print "\tAdding %s to results" % tag
    return result

if __name__ == "__main__":
    query_csvfile("demo/query.csv", "demo.out")