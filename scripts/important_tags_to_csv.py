import pickle
import csv

with open('important_tags.dat', 'r') as fh:
	d = pickle.load(fh)
	with open('data/important_tags.csv', 'w') as spreadsheet:
		writer = csv.writer(spreadsheet)
		writer.writerow(['tag_count', 'selectivity'])
		for key, value in enumerate(d):
			writer.writerow([key, value])
