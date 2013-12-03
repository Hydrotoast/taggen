import pickle
import csv

with open('tag_count.dat', 'r') as fh:
	d = pickle.load(fh)
	with open('data/tag_count.csv', 'w') as spreadsheet:
		writer = csv.writer(spreadsheet)
		writer.writerow(['tag_count', 'questions'])
		for key, value in d.items():
			writer.writerow([key, value])
