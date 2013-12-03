import pickle
import csv

with open('tag_hist.dat', 'r') as fh:
	d = pickle.load(fh)
	with open('data/tag_hist.csv', 'w') as spreadsheet:
		writer = csv.writer(spreadsheet)
		writer.writerow(['tag', 'count'])
		for key, value in d.items():
			writer.writerow([key, value])
