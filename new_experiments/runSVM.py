import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC

from MAPScorer import MAPScorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score


corpus = pickle.load(open("corpusTemporalBreastLung/corpusTemporal-1000b.p", 'rb'))
values = pickle.load(open('corpusTemporalBreastLung/valuesTemporalSede1.p', 'rb'))

y = [values.index(i) for i in corpus['train']['sede1']]
yV = [values.index(i) for i in corpus['valid']['sede1']]
yT = [values.index(i) for i in corpus['test']['sede1']]

vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=3)
transformer = TfidfTransformer(smooth_idf=False)
clf = LinearSVC()

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
print("Valid F1 {}".format(f1_score(yV,ypV, average="macro")))

print("Test Acc. {}".format(accuracy_score(yT,ypT)))
print("Test Kappa {}".format(cohen_kappa_score(yT,ypT)))
print("Test F1 {}".format(f1_score(yT,ypT, average="macro")))

