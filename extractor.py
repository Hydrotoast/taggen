import sklearn as sk
import numpy as np
import scipy as sp

import csv
import pickle
import os

TRAIN_FILE = '/tmp/train.csv'
TRAIN_SELECTED_FILE = 'data/train_selected.csv'
TAG_HIST_FILE = 'tag_hist.dat'
TAG_COUNT_FILE = 'tag_count.dat'
IMPORTANT_TAGS_FILE = 'important_tags.dat'

def cache_tag_data():
	hist = {}
	tag_count = {}
	with open(TRAIN_FILE, 'r') as fh:
		reader = csv.reader(fh)
		for row in reader:
			tags = row[3].split()
			for tag in tags:
				hist[tag] = 1 if tag not in hist else hist[tag] + 1
			tag_count[len(tags)] = 1 if len(tags) not in tag_count else tag_count[len(tags)] + 1
	pickle.dump(hist, open(TAG_HIST_FILE, 'w'))
	pickle.dump(tag_count, open(TAG_COUNT_FILE, 'w'))

def tag_hist():
	if not os.path.exists(TAG_HIST_FILE):
		cache_tag_data()
	return pickle.load(open(TAG_HIST_FILE, 'r'))

def tag_count():
	if not os.path.exists(TAG_COUNT_FILE):
		cache_tag_data()
	return pickle.load(open(TAG_COUNT_FILE, 'r'))

def important_tags(hist, k=200):
	if os.path.exists(IMPORTANT_TAGS_FILE):
		return pickle.load(open(IMPORTANT_TAGS_FILE, 'r'))
	data = [key for key, value in hist.iteritems() if value >= k]
	pickle.dump(data, open(IMPORTANT_TAGS_FILE, 'w'))
	return data

def important_records(selected_tags):
	records = []
	with open(TRAIN_FILE, 'r') as fh:
		reader = csv.reader(fh)
		for row in reader:
			tags = row[3]
			if any([True for e in tags.split() if e in selected_tags]):
				records.append(row)
	with open(TRAIN_SELECTED, 'w') as fh:
		writer = csv.writer(fh)
		for row in records:
			writer.writerow(row)
	return records

hist = tag_hist()
count = tag_count()

h = {}
for v in hist.values():
	h[v] = 1 if v not in h else h[v] + 1
print ','.join(map(str,h.keys()[:100]))
print ','.join(map(str,h.values()[:100]))

selected_tags = important_tags(hist)
print len(selected_tags)
# important_records(selected_tags)
