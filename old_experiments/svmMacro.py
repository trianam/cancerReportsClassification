import time
import datetime
import numpy as np
import scipy as sp
import pandas as pd
import csv
import sklearn as sl
import sklearn.feature_extraction.text 
import sklearn.svm
import sklearn.cross_validation
import sys
import notifier
import nltk
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import random


now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

outputDir = "outSvm"
fileIsto = "./data/ISTOLOGIE_corr.csv"
fileNeop = "./data/RTRT_NEOPLASI.csv"
fileOut = "./"+outputDir+"/metricsSvm-"+timestamp+".txt"
fileOutDetail = "./"+outputDir+"/metricsSvmDetails-"+timestamp+".txt"
fileTokensNotizie = "./"+outputDir+"/tokensNotizieSvm-"+timestamp+".txt"
fileTokensDiagnosi = "./"+outputDir+"/tokensDiagnosiSvm-"+timestamp+".txt"
fileTokensMacroscopia = "./"+outputDir+"/tokensMacroscopiaSvm-"+timestamp+".txt"
filePrecRecallSede = "./"+outputDir+"/precisionRecallSedeSvm-"+timestamp+".pdf"
filePrecRecallMorfo = "./"+outputDir+"/precisionRecallMorfoSvm-"+timestamp+".pdf"


#encoding = 'iso-8859-1' or encoding = 'cp1252' or encoding = 'latin1'
encoding = 'iso-8859-1'
cutOff = 3
tokenPattern = '(?u)\\b\\w*[a-zA-z_][a-zA-Z_]+\\w*\\b'

random_state = np.random.RandomState(0)

stemmer = nltk.stem.snowball.ItalianStemmer(ignore_stopwords=True)
analyzer = sl.feature_extraction.text.CountVectorizer(analyzer='word', token_pattern=tokenPattern).build_analyzer()
modAnalyzer = lambda doc: (stemmer.stem(w) for w in analyzer(doc))
#modAnalyzer = analyzer

time0 = time.time()

print("{0:.1f} sec - Importing csv".format(time.time()-time0))
dfIsto = pd.read_csv(fileIsto, header = 0, encoding=encoding, quoting=csv.QUOTE_NONNUMERIC, low_memory=False, usecols = ['notizie', 'diagnosi', 'macroscopia', 'ICDO1_T', 'ICDO1_M', 'id_neopl'], dtype={'id_neopl':pd.np.float64, 'ICDO1_T':pd.np.str, 'ICDO1_M':pd.np.str})#, converters={'ICDO1_T':str, 'ICDO1_M':str})
dfNeop = pd.read_csv(fileNeop, header = 0, usecols = ['id_neopl', 'sede', 'morfologia'], low_memory=False, dtype={'id_neopl':pd.np.float64, 'sede':pd.np.str, 'morfologia':pd.np.str})

print("{0:.1f} sec - Merging csv".format(time.time()-time0))
dfMerge = pd.merge(dfIsto, dfNeop, on='id_neopl')
dfMerge = dfMerge.head(100)
#print(len(dfMerge))
#sys.exit()
#dfMerge = dfMerge.ix[random.sample(list(dfMerge.index), 10000)]

#print("{0:.1f} - Combining text fields".format(time.time()-time0))
#combineFun = lambda p: lambda s: "" if pd.isnull(s) else " ".join(map(lambda w: str(p)+w.upper(), str(s).split()))
nullToEmpty = lambda s: "" if pd.isnull(s) else str(s)
dfMerge['notizie'] = dfMerge.notizie.map(nullToEmpty)
dfMerge['diagnosi'] = dfMerge.diagnosi.map(nullToEmpty)
dfMerge['macroscopia'] = dfMerge.macroscopia.map(nullToEmpty)

print("{0:.1f} sec - Tokenizing".format(time.time()-time0))
countVecNotizie = sklearn.feature_extraction.text.CountVectorizer(min_df=cutOff, encoding=encoding, strip_accents='unicode', analyzer=modAnalyzer, stop_words=stemmer.stopwords)
countVecDiagnosi = sklearn.feature_extraction.text.CountVectorizer(min_df=cutOff, encoding=encoding, strip_accents='unicode', analyzer=modAnalyzer, stop_words=stemmer.stopwords)
countVecMacroscopia = sklearn.feature_extraction.text.CountVectorizer(min_df=cutOff, encoding=encoding, strip_accents='unicode', analyzer=modAnalyzer, stop_words=stemmer.stopwords)
termMatrix = sp.sparse.hstack([countVecNotizie.fit_transform(dfMerge.notizie), countVecDiagnosi.fit_transform(dfMerge.diagnosi), countVecMacroscopia.fit_transform(dfMerge.macroscopia)])

#dfTerm = pd.DataFrame(termMatrix.toarray(), columns=countvec.get_feature_names())

#sys.exit()

np.savetxt(fileTokensNotizie, countVecNotizie.get_feature_names(), '%s')
np.savetxt(fileTokensDiagnosi, countVecDiagnosi.get_feature_names(), '%s')
np.savetxt(fileTokensMacroscopia, countVecMacroscopia.get_feature_names(), '%s')

print("{0:.1f} sec - Training".format(time.time()-time0))
targetSedeUnb = dfMerge.sede.tolist()
targetMorfoUnb = dfMerge.morfologia.tolist()
targetSede = sl.preprocessing.label_binarize(targetSedeUnb, np.unique(targetSedeUnb))
targetMorfo = sl.preprocessing.label_binarize(targetMorfoUnb, np.unique(targetMorfoUnb))
#targetSede = np.array(list(map(str, dfMerge.sede.tolist())))
#targetMorfo = np.array(list(map(str, dfMerge.morfologia.tolist())))

trainMatrixSede, testMatrixSede, trainTargetSede, testTargetSede = sl.cross_validation.train_test_split(termMatrix.tocsr(), targetSede, test_size=.1, random_state=random_state)
trainMatrixMorfo, testMatrixMorfo, trainTargetMorfo, testTargetMorfo = sl.cross_validation.train_test_split(termMatrix.tocsr(), targetMorfo, test_size=.1, random_state=random_state)


#clfSede = sklearn.multiclass.OneVsRestClassifier(sklearn.svm.SVC(kernel='linear', probability=True, random_state=random_state))
#clfMorfo = sklearn.multiclass.OneVsRestClassifier(sklearn.svm.SVC(kernel='linear', probability=True, random_state=random_state))
clfSede = sklearn.multiclass.OneVsRestClassifier(sklearn.svm.LinearSVC(random_state=random_state))
clfMorfo = sklearn.multiclass.OneVsRestClassifier(sklearn.svm.LinearSVC(random_state=random_state))


yScoreSede = clfSede.fit(trainMatrixSede, trainTargetSede).decision_function(testMatrixSede)
yScoreMorfo = clfMorfo.fit(trainMatrixMorfo, trainTargetMorfo).decision_function(testMatrixMorfo)


precisionSede = dict()
recallSede = dict()
precisionMorfo = dict()
recallMorfo = dict()

aucSede = dict()
aucMorfo = dict()

precisionSede['macro'] = 0.
recallSede['macro'] = 0.
precisionMorfo['macro'] = 0.
recallMorfo['macro'] = 0.


numClassiSede = yScoreSede.shape[1]
numClassiMorfo = yScoreMorfo.shape[1]


precisionSede[0], recallSede[0], _ = sklearn.metrics.precision_recall_curve(testTargetSede[:,0],yScoreSede[:,0])
precisionSede['macro'] = precisionSede[0]
recallSede['macro'] = recallSede[0]

for i in range(1, numClassiSede):
	precisionSede[i], recallSede[i], threshold = sklearn.metrics.precision_recall_curve(testTargetSede[:,i],yScoreSede[:,i])
	print(threshold)
	#precisionSede['macro'] = precisionSede['macro'] + precisionSede[i]
	#recallSede['macro'] = recallSede['macro'] + recallSede[i]

precisionSede['macro'] = precisionSede['macro']/numClassiSede
recallSede['macro'] = recallSede['macro']/numClassiSede
	
precisionMorfo[0], recallMorfoMacro[0], _ = sklearn.metrics.precision_recall_curve(testTargetMorfo[:,0],yScoreMorfo[:,0])
precisionMorfo['macro'] = precisionMorfo[0]
recallMorfo['macro'] = recallMorfo[0]

for i in range(1, numClassiMorfo):
	precisionMorfo[i], recallMorfoMacro[i], _ = sklearn.metrics.precision_recall_curve(testTargetMorfo[:,i],yScoreMorfo[:,i])
	precisionMorfo['macro'] = precisionMorfo['macro'] + precisionMorfo[i]
	recallMorfo['macro'] = recallMorfo['macro'] + recallMorfo[i]

precisionMorfo['macro'] = precisionMorfo['macro']/numClassiMorfo
recallMorfo['macro'] = recallMorfo['macro']/numClassiMorfo

precisionSede['micro'], recallSedeMicro, _ = sklearn.metrics.precision_recall_curve(testTargetSede.ravel(),yScoreSede.ravel())
precisionMorfo['micro'], recallMorfoMicro, _ = sklearn.metrics.precision_recall_curve(testTargetMorfo.ravel(),yScoreMorfo.ravel())
	
aucSede['micro'] = sklearn.metrics.average_precision_score(testTargetSede, yScoreSede, average="micro")
aucMorfo['micro'] = sklearn.metrics.average_precision_score(testTargetMorfo, yScoreMorfo, average="micro")
aucSede['macro'] = sklearn.metrics.average_precision_score(testTargetSede, yScoreSede, average="macro")
aucMorfo['macro'] = sklearn.metrics.average_precision_score(testTargetMorfo, yScoreMorfo, average="macro")

plt.clf()
ax = plt.figure().gca()
ax.set_xticks(np.arange(0,1,0.05))
ax.set_yticks(np.arange(0,1.,0.05))
plt.xticks(rotation='vertical')
plt.plot([0.,1.],[0.,1.], color='blue')
plt.plot(recallSedeMicro, precisionSede['micro'], color='gold', lw=2, label='micro-average Precision-recall curve (area = {0:0.2f})'''.format(aucSede['micro']))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve Sede')
plt.legend(loc="lower right")
plt.grid()
plt.savefig(filePrecRecallSede)

plt.clf()
ax = plt.figure().gca()
ax.set_xticks(np.arange(0,1,0.05))
ax.set_yticks(np.arange(0,1.,0.05))
plt.xticks(rotation='vertical')
plt.plot([0.,1.],[0.,1.], color='blue')
plt.plot(recallMorfoMicro, precisionMorfo['micro'], color='gold', lw=2, label='micro-average Precision-recall curve (area = {0:0.2f})'''.format(aucMorfo['micro']))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve Morfo')
plt.legend(loc="lower right")
plt.grid()
plt.savefig(filePrecRecallMorfo)

notifier.sendMessage("Terminated "+timestamp+" execution ({0:.1f} sec)".format(time.time()-time0))

print("{0:.1f} sec - End".format(time.time()-time0))

sys.exit()
#print("{0:.1f} - Score".format(time.time()-time0))
#scoreSede = clfSede.score(testMatrix, testTargetSede)
#scoreMorfologia = clfMorfo.score(testMatrix, testTargetMorfo)

print("{0:.1f} sec - Predict".format(time.time()-time0))
testPredictSede = clfSede.predict(testMatrix)
testPredictMorfo = clfMorfo.predict(testMatrix)

print("{0:.1f} sec - Metrics".format(time.time()-time0))
out_file = open(fileOut, 'w')

out_file.write("f1 score sede micro average: ")
out_file.write(str(sl.metrics.f1_score(list(map(str, testTargetSede)), list(map(str, testPredictSede)), average='micro')))
out_file.write("\nf1 score sede macro average: ")
out_file.write(str(sl.metrics.f1_score(list(map(str, testTargetSede)), list(map(str, testPredictSede)), average='macro')))

out_file.write("\nf1 score morfologia micro average: ")
out_file.write(str(sl.metrics.f1_score(list(map(str, testTargetMorfo)), list(map(str, testPredictMorfo)), average='micro')))
out_file.write("\nf1 score morfologia macro average: ")
out_file.write(str(sl.metrics.f1_score(list(map(str, testTargetMorfo)), list(map(str, testPredictMorfo)), average='macro')))
out_file.write("\n")
out_file.close()

out_file_details = open(fileOutDetail, 'w')

out_file_details.write(sl.metrics.classification_report(list(map(str, testTargetSede)), list(map(str, testPredictSede))))
out_file_details.write("\n")
out_file_details.write(sl.metrics.classification_report(list(map(str, testTargetMorfo)), list(map(str, testPredictMorfo))))

out_file_details.close()

notifier.sendFile("Terminated "+timestamp+" execution ({0:.1f} sec)".format(time.time()-time0), fileOut)

print("{0:.1f} sec - End".format(time.time()-time0))

