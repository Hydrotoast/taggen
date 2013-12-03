__author__ = 'workbook'

import csv
import re
import pickle
import random
from time import time
from functools import wraps
from os import path
from os import makedirs


TRAIN_FILE = "/host/Users/Public/Documents/Train.csv"
#TRAIN_FILE = "/host/Users/Public/Documents/head.csv"
CLEAN_CSV_FILE = "/media/PAULKANG/train_clean.csv"
CLEAN_DAT_FILE = "/media/PAULKANG/train_cleanTAG_RINDEX_FILE.dat"
SAMPLE_CSV_DIR = "/media/PAULKANG/samples"
TAG_RINDEX_FILE = "/media/PAULKANG/tag_index.dat"


def timed(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        start = time()
        result = f(*args, **kwds)
        elapsed = time() - start
        print "%s took %d sec to finish" % (f.__name__, elapsed)
        return result
    return wrapper


#@timed
#def cleaned_parse(train=TRAIN_FILE, dump_file=CLEAN_DAT_FILE):
#    if dump_file and path.isfile(dump_file):
#        return pickle.load(open(dump_file, "r"))
#
#    docs = {}
#    with open(train, "r") as tf:
#        rd = csv.reader(tf)
#        rd.next()idx
#
#        text_start = 0
#        progress = 10000
#        starting_code = False
#        for i, line in enumerate(rd):
#            doc_id = int(line[0])
#            raw_text = ""
#            code = []
#            for match in re.finditer("(<[^<]*>)", line[2]):
#                rnge = match.span()
#                if text_start == rnge[0]:
#                    text_start = rnge[1]
#                else:
#                    text = line[2][text_start:rnge[0]]
#                    if starting_code:
#                        # Do something with the code
#                        code.append(text)
#                        starting_code = False
#                    else:
#                        raw_text += text
#                        if match.group(1) == "<code>":
#                            starting_code = True
#                        elif match.group(1) == "<pre>":
#                            starting_code = True
#                    text_start = rnge[1]
#            if (i+1) % progress == 0:
#                print "parse_body progress complete: %d" % (i+1)
#            raw_text = " ".join(raw_text.split())
#            docs[int(line[0])] = (line[1], raw_text, code)
#    if dump_file:
#        pickle.dump(docs, open(dump_file, "w"))
#    return docs


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
            text_start = 0
            raw_text = ""
            code = []
            starting_code = False
            for match in re.finditer("(<[^<]*>)", line[2]):
                rnge = match.span()
                if text_start == rnge[0]:
                    text_start = rnge[1]
                else:
                    text = line[2][text_start:rnge[0]]
                    if starting_code:
                        code.append(text)
                        starting_code = False
                    else:
                        raw_text += text
                        if match.group(1) == "<code>":
                            starting_code = True
                        elif match.group(1) == "<pre>":
                            starting_code = True
                    text_start = rnge[1]
            if (i+1) % progress == 0:
                print "parse_body progress complete: %d" % (i+1)
            raw_text = " ".join(raw_text.split())
            wr.writerow([line[0], line[1], raw_text, str(code)])
            ccf.flush()


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
    return tag_idx#@timed


#def generate_samples_csv(tag_samples, ctrain=CLEAN_CSV_FILE, sample_dir=SAMPLE_CSV_DIR):
#    if not path.isdir(sample_dir):
#        makedirs(sample_dir)
#
#    progress = 10000
#    with open(ctrain, "w") as ccf:
#        rd = csv.reader(ccf)
#        rd.next()
#
#        for i, line in enumerate(rd):
#            for tag in tag_samples:
#                with open("%s/%s" % (sample_dir, tag), "a") as tsf:
#                    wr = csv.writer(tsf)
#                    sample = tag_samples[tag]
#                    if int(line[0]) in sample[0]:
#                        wr.writerow([line[0], line[1], line[2], True])
#                    elif int(line[0]) in sample[1]:
#                        wr.writerow([line[0], line[1], line[2], False])
#            if (i + 1) % progress == 0:
#                print "generate_samples_csv doc position: %d" % (i+1)


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


if __name__ == '__main__':
    cleaned_csv_parse()
    rindex = r_index_parse()
    samples = generate_samples(rindex, 6034195)
    sample_idx = tag_sample_index(samples)
    generate_samples_csv(sample_idx)




