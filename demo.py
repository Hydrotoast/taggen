
__author__ = "Paul Kang"

import config
import os
import pickle
import csv
import extract
import query
import subset_pipeline
from sklearn.metrics import f1_score
from util import timed


@timed
def find_neighboring_tags(tag, r_index):
    if os.path.isfile(os.path.join(config.DEMO_DIR, 'demo_' + tag + '_neighbor.dat')):
        return pickle.load(open(os.path.join(config.DEMO_DIR, 'demo_' + tag + '_neighbor.dat'), 'r'))

    documents = r_index[tag]
    n = []
    ls = os.listdir(config.SAMPLES_DIR)
    for t in ls:
        for d in r_index[t]:
            if d in documents:
                n.append(t)
                break

    pickle.dump(n, open(os.path.join(config.DEMO_DIR, 'demo_' + tag + '_neighbor.dat'), 'w'))
    return n


@timed
def find_sample_tags(sample_file, csv_file):
    if os.path.isfile(os.path.join(config.DEMO_DIR, 'demo_tags.dat')):
        return pickle.load(open(os.path.join(config.DEMO_DIR, 'demo_tags.dat')))
    ls = os.listdir(config.SAMPLES_DIR)
    doc_ids = []
    with open(sample_file, 'r') as sf:
        rd = csv.reader(sf)
        for row in rd:
            doc_ids.append(int(row[0]))

    n = []
    with open(csv_file, 'r') as cf:
        rd = csv.reader(cf)
        rd.next()

        for row in rd:
            if int(row[0]) in doc_ids:
                n.extend(filter(lambda x: x in ls and x not in n, row[3].split()))
    pickle.dump(n, open(os.path.join(config.DEMO_DIR, "demo_tags.dat"), 'w'))
    return n


def generate_demo_data(sample_file, csv_file):
    if os.path.isfile(os.path.join(config.DEMO_DIR, "query.csv")):
        return pickle.load(open(os.path.join(config.DEMO_DIR, "demo_tags.dat"), 'r')), \
            pickle.load(open(os.path.join(config.DEMO_DIR, "demo_targets.dat"), 'r'))

    doc_ids = []
    ls = os.listdir(config.SAMPLES_DIR)
    with open(sample_file, 'r') as sf:
        rd = csv.reader(sf)
        for row in rd:
            doc_ids.append(int(row[0]))

    n = []
    targets = []
    with open(csv_file, 'r') as cf, open(os.path.join(config.DEMO_DIR, "query.csv"), 'w') as qf:
        wr = csv.writer(qf)
        rd = csv.reader(cf)
        rd.next()

        for row in rd:
            if int(row[0]) in doc_ids:
                wr.writerow(row)
                tags = row[3].split()
                targets.append(tags)
                n.extend(filter(lambda x: x in ls and x not in n, tags))
    pickle.dump(n, open(os.path.join(config.DEMO_DIR, "demo_tags.dat"), 'w'))
    pickle.dump(targets, open(os.path.join(config.DEMO_DIR, "demo_targets.dat"), 'w'))
    return n, targets


def find_targets(sample_file, csv_file):
    if os.path.isfile(os.path.join(config.DEMO_DIR, 'demo_targets.dat')):
        return pickle.load(open(os.path.join(config.DEMO_DIR, 'demo_targets.dat')))
    doc_ids = []
    with open(sample_file, 'r') as sf:
        rd = csv.reader(sf)
        for row in rd:
            doc_ids.append(int(row[0]))

    targets = []
    with open(csv_file, 'r') as cf:
        rd = csv.reader(cf)
        rd.next()

        for row in rd:
            if int(row[0]) in doc_ids:
                targets.append(row[3].split())

    pickle.dump(targets, open(os.path.join(config.DEMO_DIR, "demo_targets.dat"), 'w'))
    return targets


def align(a, b):
    f_vec = a
    for c in b:
        if c not in f_vec:
            f_vec.append(c)
    return [d in a for d in f_vec], [d in b for d in f_vec]


if __name__ == "__main__":
    # Generate the data required for this demo: Neighboring tags, query.csv file, query targets
    n, target = generate_demo_data(os.path.join(config.DEMO_DIR, 'demo_sample') , config.TRAIN_FILE)
    print "neighboring tag count: %d" % len(n)
    # Generate the estimators for the selected tags
    subset_pipeline.analyze_subset_pipelines(subset=n)

    predict = query.query_csvfile(os.path.join(config.DEMO_DIR, 'query.csv'),
                                  os.path.join(config.DEMO_DIR, 'predict.txt'))
    target = find_targets(os.path.join(config.DEMO_DIR, 'query.csv'), config.TRAIN_FILE)

    # Prompt for keyword extraction
    prompt = raw_input("Add keyword extraction? (Y/n):")
    if prompt == 'Y' or prompt == 'y':
        ext = extract.extract_keywords_csv(os.path.join(config.DEMO_DIR, 'query.csv'), predict)

        assert len(target) == len(predict)
        for i, x in enumerate(ext):
            predict[i].extend([c for a, c in x][0:1])

    # Write out the prediction... a bit redundant but will have to do..
    with open(os.path.join(config.DEMO_DIR, 'predict.txt'), 'w') as pf:
        wr = csv.writer(pf)
        for i, row in enumerate(predict):
            wr.writerow([i, str(row)])

    score = []
    for i in range(0, len(target)):
        a = align(target[i], predict[i])
        score.append(f1_score(a[0], a[1]))
        print "%d: %f" % (i, score[i])

    print "avg f1_score: %f" % (sum(score)/len(score))
