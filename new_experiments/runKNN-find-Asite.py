import numpy as np
import math
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn import neighbors

from MAPScorer import MAPScorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score

import configurationsMulticlass

conf = configurationsMulticlass.configTbaseMultiBest

corpus = pickle.load(open(conf.fileCorpus, 'rb'))
values = pickle.load(open(conf.fileValues, 'rb'))
corpusKey = conf.corpusKey

#corpus = {d:{k:corpus[d][k][:100] for k in corpus[d]} for d in corpus}

print("--------")

y = [values.index(i) for i in corpus['train'][corpusKey]]
yV = [values.index(i) for i in corpus['valid'][corpusKey]]
yT = [values.index(i) for i in corpus['test'][corpusKey]]

vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=3)
transformer = TfidfTransformer(smooth_idf=False)

#train
counts = vectorizer.fit_transform(corpus['train']['text'])
tfidf = transformer.fit_transform(counts)

#valid
countsValid = vectorizer.transform(corpus['valid']['text'])
tfidfValid = transformer.transform(countsValid)

#test
countsTest = vectorizer.transform(corpus['test']['text'])
tfidfTest = transformer.transform(countsTest)

bestAcc=0.
for nn in range(1,30):
    print("{}: ".format(nn), end="")
    clf = neighbors.KNeighborsClassifier(nn, weights='distance')
    clf.fit(tfidf, y)
    ypV = clf.predict(tfidfValid)
    currAcc = accuracy_score(yV,ypV)
    print(currAcc)
    if currAcc > bestAcc:
        bestAcc = currAcc
        bestClf = clf
print("Best: {}".format(bestAcc))
clf = bestClf

#valid
ypV = clf.predict(tfidfValid)
#test
ypT = clf.predict(tfidfTest)
yppT = clf.predict_proba(tfidfTest)


print("Valid Acc. {}".format(accuracy_score(yV,ypV)))
print("Valid Kappa {}".format(cohen_kappa_score(yV,ypV)))
print("Valid F1 m {}".format(f1_score(yV,ypV, average="micro")))
print("Valid F1 M {}".format(f1_score(yV,ypV, average="macro")))


ypb3 = np.argsort(yppT,1)[:,-3:][:,::-1] #top 3
ypb5 = np.argsort(yppT,1)[:,-5:][:,::-1] #top 5

total=0
correct3=0
correct5=0
for i in range(len(yT)):
    total += 1
    if yT[i] in [clf.classes_[i] for i in ypb3[i]]:
        correct3 += 1
    if yT[i] in [clf.classes_[i] for i in ypb5[i]]:
        correct5 += 1

accT3 = correct3/total
accT5 = correct5/total



curr = {
    "acc": accuracy_score(yT,ypT),
    "kappa": cohen_kappa_score(yT,ypT),
    "f1m": f1_score(yT,ypT, average="micro"),
    "f1M": f1_score(yT,ypT, average="macro"),
    "accT3": accT3,
    "accT5": accT5,
        }

print("Test Acc. {}".format(curr['acc']))
print("Test Kappa {}".format(curr['kappa']))
print("Test F1 m {}".format(curr['f1m']))
print("Test F1 M {}".format(curr['f1M']))
print("Test Acc. T3 {}".format(curr['accT3']))
print("Test Acc. T5 {}".format(curr['accT5']))

pickle.dump({'y':yT, 'yp':ypT, 'ypp':yppT, 'yppClasses':clf.classes_}, open("predictionsKNN-find-Asite.p", 'wb'))

