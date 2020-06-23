import math
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC, SVC

from MAPScorer import MAPScorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score

import configurations

conf = configurations.configP[0][2][1][2]

corpusAll = pickle.load(open(conf.fileCorpus, 'rb'))
values = pickle.load(open(conf.fileValues, 'rb'))
split = pickle.load(open(conf.fileSplit, 'rb'))
corpusKey = conf.corpusKey

folds = [0,1,2,3,4,5,6,7,8,9]

m = {'acc':0, 'kappa':0, 'f1m':0, 'f1M':0}
mm = {k:0 for k in m}

for fold in folds:
    print("--------")
    corpus = {d:{k:[corpusAll[k][i] for i in split[d][fold]] for k in corpusAll} for d in split}

    y = [values.index(i) for i in corpus['train'][corpusKey]]
    yV = [values.index(i) for i in corpus['valid'][corpusKey]]
    yT = [values.index(i) for i in corpus['test'][corpusKey]]

    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=3)
    transformer = TfidfTransformer(smooth_idf=False)
    #clf = LinearSVC()
    clf = SVC(C=25, kernel='sigmoid', gamma=0.5, shrinking=True)

    #train
    counts = vectorizer.fit_transform(corpus['train']['text'])
    tfidf = transformer.fit_transform(counts)
    clf.fit(tfidf, y)

    #valid
    countsValid = vectorizer.transform(corpus['valid']['text'])
    tfidfValid = transformer.transform(countsValid)
    ypV = clf.predict(tfidfValid)

    #test
    countsTest = vectorizer.transform(corpus['test']['text'])
    tfidfTest = transformer.transform(countsTest)
    ypT = clf.predict(tfidfTest)


    print("Valid Acc. {}".format(accuracy_score(yV,ypV)))
    print("Valid Kappa {}".format(cohen_kappa_score(yV,ypV)))
    print("Valid F1 m {}".format(f1_score(yV,ypV, average="micro")))
    print("Valid F1 M {}".format(f1_score(yV,ypV, average="macro")))

    curr = {
        "acc": accuracy_score(yT,ypT),
        "kappa": cohen_kappa_score(yT,ypT),
        "f1m": f1_score(yT,ypT, average="micro"),
        "f1M": f1_score(yT,ypT, average="macro"),
            }

    print("Test Acc. {}".format(curr['acc']))
    print("Test Kappa {}".format(curr['kappa']))
    print("Test F1 m {}".format(curr['f1m']))
    print("Test F1 M {}".format(curr['f1M']))

    for k in m:
        m[k] += curr[k]
        mm[k] += curr[k] * curr[k]

avg = {k:m[k]/len(folds) for k in m}         
std = {k:math.sqrt((mm[k] / len(folds)) - (avg[k] * avg[k])) for k in m}         
print("========")         
print(avg)         
print(std)

