
import config
import os
import preprocess
import csv
from numpy import mean
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_keywords_csv(query_file, tag_list, sample_dir=config.SAMPLES_DIR, stop_words="english"):
    rslt = []
    with open(query_file, 'r') as qf:
        rd = csv.reader(qf)
        rd.next()
        for i, row in enumerate(rd):
            query_rslt = {}
            tag_list[i] += [
                'sql', 'arrays', 'python' # TODO Should be the most common tags
            ]
            for tag in tag_list[i]:
                kwords = extract_keywords(row, [row for row in open(os.path.join(sample_dir, tag), 'r')], stop_words)
                for v, t in kwords:
                    query_rslt[t] = query_rslt[t] + [v] if v in query_rslt else [v]
            rslt.append(sorted([(mean(v), t) for t, v in query_rslt.items()], reverse=True))
    return rslt


def extract_keywords(query, corpus, stop_words):
    sample = [preprocess.format_input(preprocess.parse_doc(query))] + corpus

    cv = TfidfVectorizer(ngram_range=(1, 2), stop_words=stop_words)
    X = cv.fit_transform(sample)

    x = zip(X.toarray()[0], cv.get_feature_names())
    return sorted(x, reverse=True)[0:10]


if __name__ == "__main__":
    for r in extract_keywords_csv("test/test_query.csv", [[], ['c#', 'asp.net', 'windows-phone-7']]):
        print str([t for v, t in r][0:4])
