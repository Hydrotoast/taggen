__author__ = 'workbook'

import csv
import re
import pickle
import random
import ast
from time import time
from functools import wraps
from os import path
from os import makedirs
from os import listdir
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer


TRAIN_FILE = "/host/Users/Public/Documents/Train.csv"
TEST_FILE = "Test.csv"
CLEAN_CSV_FILE = "train_clean.csv"
CLEAN_DAT_FILE = "train_clean.dat"
SAMPLE_CSV_DIR = "samples"
FEATURE_DIR = "features"
TAG_RINDEX_FILE = "tag_index.dat"
STOPWORDS_FILE = "stopwords.txt"
STOPWORDS_DAT = "stopwords.dat"


def timed(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        start = time()
        result = f(*args, **kwds)
        elapsed = time() - start
        print "%s took %d sec to finish" % (f.__name__, elapsed)
        return result
    return wrapper


@timed
def cleaned_parse(query_file):
    docs = {}
    with open(query_file, "r") as tf:
        rd = csv.reader(tf)
        rd.next()

        text_start = 0
        progress = 10000
        starting_code = False
        for i, line in enumerate(rd):
            docs[int(line[0])] = parse_doc(line)
            if (i+1) % progress == 0:
                print "parse_body progress complete: %d" % (i+1)
    return docs


# ======= Stored to disk ==========
@timed
def cleaned_csv_parse(out_file=CLEAN_CSV_FILE, train=TRAIN_FILE):
    if path.isfile(out_file):
        print "File already exists"
        return
    with open(out_file, "w") as ccf, open(train, "r") as tf:
        wr = csv.writer(ccf)
        wr.writerow(["Id", "Title", "Body", "Code"])
        rd = csv.reader(tf)
        rd.next()

        progress = 10000
        for i, line in enumerate(rd):
            raw_text, code = parse_text(line[2])
            if (i+1) % progress == 0:
                print "parse_body progress complete: %d" % (i+1)
            wr.writerow([line[0], line[1], raw_text, str(code)])
            ccf.flush()


def parse_doc(row):
    raw_text, code = parse_text(row[2])
    return row[0], row[1], raw_text, code


def parse_text(text):
    text_start = 0
    raw_text = ""
    code = []
    starting_code = False
    for match in re.finditer("(<[^<]*>)", text):
        rnge = match.span()
        if text_start == rnge[0]:
            text_start = rnge[1]
        else:
            t = text[text_start:rnge[0]]
            if starting_code:
                code.append(t)
                starting_code = False
            else:
                raw_text += t
                if match.group(1) == "<code>":
                    starting_code = True
                elif match.group(1) == "<pre>":
                    starting_code = True
            text_start = rnge[1]

    raw_text = " ".join(raw_text.split())
    return raw_text, code


@timed
def r_index_parse(train=TRAIN_FILE, dump_file=TAG_RINDEX_FILE):
    if path.isfile(dump_file):
        return pickle.load(open(dump_file, "r"))

    tag_index = {}
    with open(train, "r") as tf:
        rd = csv.reader(tf)
        rd.next()

        progress = 10000
        for i, line in enumerate(rd):
            for tag in line[3].split():
                tag_index[tag] = [int(line[0])] if tag not in tag_index \
                    else tag_index[tag] + [int(line[0])]
            if (i+1) % progress == 0:
                print "parse_tags_index progress complete: %d" % (i+1)

    if dump_file:
        pickle.dump(tag_index, open(dump_file, "w"))
    return tag_index


def tag_sample(tag, r_index, corpus_size, pos_count, neg_count):
    random.seed(time())
    doc_ids = r_index[tag]

    pos_docs = []
    while len(pos_docs) < pos_count and len(pos_docs) < len(doc_ids):
        rid = random.choice(doc_ids)
        if rid not in pos_docs:
            pos_docs.append((rid, True))

    neg_docs = []
    while len(neg_docs) < neg_count:
        rid = random.randint(0, corpus_size)
        if rid not in neg_docs:
            neg_docs.append((rid, False))
    return pos_docs + neg_docs


@timed
def generate_samples(r_index, corpus_size, pos_count=60, neg_count=1500, threshold=2000):
    tag_samples = {}
    for tag in r_index:
        if len(r_index[tag]) > threshold:
            tag_samples[tag] = tag_sample(tag, r_index, corpus_size, pos_count, neg_count)
    return tag_samples


@timed
def tag_sample_index(samples):
    tag_idx = {}
    for tag in samples:
        for doc_id, positive in samples[tag]:
            tag_idx[doc_id] = [(tag, positive)] if doc_id not in tag_idx else tag_idx[doc_id] + [(tag, positive)]
    return tag_idx


@timed
def generate_samples_csv(sample_idx, ctrain=CLEAN_CSV_FILE, sample_dir=SAMPLE_CSV_DIR):
    if not path.isdir(sample_dir):
        makedirs(sample_dir)

    progress = 10000
    with open(ctrain, "r") as ccf:
        rd = csv.reader(ccf)
        rd.next()

        for i, line in enumerate(rd):
            doc_id = int(line[0])
            if doc_id in sample_idx:
                tag_list = sample_idx[doc_id]
                for tag, positive in tag_list:
                    with open("%s/%s" % (sample_dir, tag), "a") as tsf:
                        wr = csv.writer(tsf)
                        wr.writerow(line + [positive])
            if (i + 1) % progress == 0:
                print "generate_samples_csv doc position: %d" % (i+1)


@timed
def cached_stopwords(stop_file=STOPWORDS_FILE, dump_file=STOPWORDS_DAT):
    if path.isfile(dump_file):
        return pickle.load(open(dump_file, "r"))
    with open(stop_file, "r") as swf:
        stopwords = []
        for line in swf:
            stopwords.append(line.strip())
    pickle.dump(stopwords, open(dump_file, "w"))
    return stopwords


@timed
def generate_feature_vectors(feature_dir=FEATURE_DIR, sample_dir=SAMPLE_CSV_DIR, stopwords=cached_stopwords()):
    ls = listdir(sample_dir)
    for f in ls:
        file_path = "%s/%s.data"% (feature_dir, f)
        if path.isfile(file_path):
            print "file %s already exists" % file_path
            continue
        print "featurizing tag sample for: %s" % f
        x, target, vocab = sample_features(f, sample_dir, stopwords=stop_words)
        pickle.dump({"data": x, "target": target, "vocabulary": vocab}, open(file_path, "w"))



def sample_features(tag, sample_dir=SAMPLE_CSV_DIR, stopwords=[]):
    with open("%s/%s" % (sample_dir, tag), "r") as tsf:
        rd = csv.reader(tsf)
        rd.next()

        data = []
        target = []
        for line in rd:
            format_input(line)
            target.append(line[4] == "True")

        #cv = TfIdfVectorizer(stop_words=stopwords)
        cv = CountVectorizer(stop_words=stopwords)
    return cv.fit_transform(data), target, cv.vocabulary_


# Test file 2+GB
def load_test(test_file=TEST_FILE):
    with open(test_file, "r") as tf:
        rd = csv.reader(tf)
        rd.next()

        test = []
        for row in rd:
            test.append(format_input(parse_doc(row)))
    return test


def format_input(row):
    # Title and Body
    return row[1] + row[2]

    # Title, Body, and Code
    #return row[1] + row[2] + " ".join(ast.literal_eval(row[3]))


if __name__ == '__main__':
    cleaned_csv_parse()
    rindex = r_index_parse()
    samples = generate_samples(rindex, 6034195)
    sample_idx = tag_sample_index(samples)
    generate_samples_csv(sample_idx)
    stop_words = cached_stopwords()
    generate_feature_vectors()