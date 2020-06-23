import time
import datetime
import numpy as np
import scipy as sp
import pandas as pd
import csv
import sklearn as sl
import sklearn.feature_extraction.text 
import sklearn.svm
import sklearn.model_selection
import sys
import notifier
import nltk
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import random
import seaborn as sns


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
fileLearningCurveSede = "./"+outputDir+"/learningCurveSedeSvm-"+timestamp+".pdf"
fileLearningCurveMorfo = "./"+outputDir+"/learningCurveMorfoSvm-"+timestamp+".pdf"
fileKdeSede = "./"+outputDir+"/kdeSede-"+timestamp+".pdf"
fileKdeMorfo = "./"+outputDir+"/kdeMorfo-"+timestamp+".pdf"
fileFreq = "./"+outputDir+"/freqFeatures-"+timestamp+".pdf"


#encoding = 'iso-8859-1' or encoding = 'cp1252' or encoding = 'latin1'
encoding = 'iso-8859-1'
cutOff = 3
maxDf = 0.5
tokenPattern = '(?u)\\b\\w*[a-zA-z_][a-zA-Z_]+\\w*\\b'

random_state = np.random.RandomState(0)

stemmer = nltk.stem.snowball.ItalianStemmer(ignore_stopwords=True)
analyzer = sl.feature_extraction.text.TfidfVectorizer(analyzer='word', token_pattern=tokenPattern).build_analyzer()
modAnalyzer = lambda doc: (stemmer.stem(w) for w in analyzer(doc))
#modAnalyzer = analyzer

time0 = time.time()

print("{0:.1f} sec - Importing csv".format(time.time()-time0))
dfIsto = pd.read_csv(fileIsto, header = 0, encoding=encoding, quoting=csv.QUOTE_NONNUMERIC, low_memory=False, usecols = ['notizie', 'diagnosi', 'macroscopia', 'ICDO1_T', 'ICDO1_M', 'id_neopl'], dtype={'id_neopl':pd.np.float64, 'ICDO1_T':pd.np.str, 'ICDO1_M':pd.np.str})#, converters={'ICDO1_T':str, 'ICDO1_M':str})
dfNeop = pd.read_csv(fileNeop, header = 0, usecols = ['id_neopl', 'sede', 'morfologia'], low_memory=False, dtype={'id_neopl':pd.np.float64, 'sede':pd.np.str, 'morfologia':pd.np.str})

print("{0:.1f} sec - Merging csv".format(time.time()-time0))
dfMerge = pd.merge(dfIsto, dfNeop, on='id_neopl')
#dfMerge = dfMerge.head(100)
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
vecNotizie = sklearn.feature_extraction.text.TfidfVectorizer(min_df=cutOff, max_df=maxDf, encoding=encoding, strip_accents='unicode', analyzer=modAnalyzer, stop_words=stemmer.stopwords)
vecDiagnosi = sklearn.feature_extraction.text.TfidfVectorizer(min_df=cutOff, max_df=maxDf, encoding=encoding, strip_accents='unicode', analyzer=modAnalyzer, stop_words=stemmer.stopwords)
vecMacroscopia = sklearn.feature_extraction.text.TfidfVectorizer(min_df=cutOff, max_df=maxDf, encoding=encoding, strip_accents='unicode', analyzer=modAnalyzer, stop_words=stemmer.stopwords)
termMatrix = sp.sparse.hstack([vecNotizie.fit_transform(dfMerge.notizie), vecDiagnosi.fit_transform(dfMerge.diagnosi), vecMacroscopia.fit_transform(dfMerge.macroscopia)])

#dfTerm = pd.DataFrame(termMatrix.toarray(), columns=countvec.get_feature_names())

#sys.exit()

np.savetxt(fileTokensNotizie, vecNotizie.get_feature_names(), '%s')
np.savetxt(fileTokensDiagnosi, vecDiagnosi.get_feature_names(), '%s')
np.savetxt(fileTokensMacroscopia, vecMacroscopia.get_feature_names(), '%s')

print("{0:.1f} sec - Features freq plots".format(time.time()-time0))
#snsPlot = sns.kdeplot(np.ravel(termMatrix.todense()))
termMatrixTot = termMatrix.sum()
termMatrixSum = np.array(termMatrix.sum(0))[0]
termMatrixFreq = np.array(list(map(lambda t: t/termMatrixTot, termMatrixSum)))
#snsPlot = sns.kdeplot(termMatrixFreq)
#snsPlot.get_figure().savefig(fileKde)

plt.clf()
plt.plot(termMatrixFreq)
plt.savefig(fileFreq)


print("{0:.1f} sec - Training".format(time.time()-time0))
targetSedeUnb = dfMerge.sede.tolist()
targetMorfoUnb = dfMerge.morfologia.tolist()

labelEncoderSede = sl.preprocessing.LabelEncoder()
labelEncoderMorfo = sl.preprocessing.LabelEncoder()
targetSedeEncoded = labelEncoderSede.fit_transform(targetSedeUnb)
targetMorfoEncoded = labelEncoderMorfo.fit_transform(targetMorfoUnb)
plt.clf()
snsPlotSede = sns.kdeplot(targetSedeEncoded)
snsPlotSede.get_figure().savefig(fileKdeSede)
plt.clf()
snsPlotMorfo = sns.kdeplot(targetMorfoEncoded)
snsPlotMorfo.get_figure().savefig(fileKdeMorfo)


targetSede = sl.preprocessing.label_binarize(targetSedeUnb, np.unique(targetSedeUnb))
targetMorfo = sl.preprocessing.label_binarize(targetMorfoUnb, np.unique(targetMorfoUnb))
#targetSede = np.array(list(map(str, dfMerge.sede.tolist())))
#targetMorfo = np.array(list(map(str, dfMerge.morfologia.tolist())))

trainMatrixSede, testMatrixSede, trainTargetSede, testTargetSede = sl.model_selection.train_test_split(termMatrix.tocsr(), targetSede, test_size=.1, random_state=random_state)
trainMatrixMorfo, testMatrixMorfo, trainTargetMorfo, testTargetMorfo = sl.model_selection.train_test_split(termMatrix.tocsr(), targetMorfo, test_size=.1, random_state=random_state)


#clfSede = sklearn.multiclass.OneVsRestClassifier(sklearn.svm.SVC(kernel='linear', probability=True, random_state=random_state))
#clfMorfo = sklearn.multiclass.OneVsRestClassifier(sklearn.svm.SVC(kernel='linear', probability=True, random_state=random_state))
clfSede = sklearn.multiclass.OneVsRestClassifier(sklearn.svm.LinearSVC(random_state=random_state))
clfMorfo = sklearn.multiclass.OneVsRestClassifier(sklearn.svm.LinearSVC(random_state=random_state))

yScoreSede = clfSede.fit(trainMatrixSede, trainTargetSede).decision_function(testMatrixSede)
yScoreMorfo = clfMorfo.fit(trainMatrixMorfo, trainTargetMorfo).decision_function(testMatrixMorfo)

precisionSede, recallSede, thresholdSede = sklearn.metrics.precision_recall_curve(testTargetSede.ravel(),yScoreSede.ravel())
precisionMorfo, recallMorfo, thresholdMorfo = sklearn.metrics.precision_recall_curve(testTargetMorfo.ravel(),yScoreMorfo.ravel())
#precisionSede, recallSede, thresholdSede = sklearn.metrics.precision_recall_curve(testTargetSede,yScoreSede[:0])
#precisionMorfo, recallMorfo, thresholdMorfo = sklearn.metrics.precision_recall_curve(testTargetMorfo.ravel(),yScoreMorfo[:,0])
average_precisionSede = sklearn.metrics.average_precision_score(testTargetSede, yScoreSede, average="micro")
average_precisionMorfo = sklearn.metrics.average_precision_score(testTargetMorfo, yScoreMorfo, average="micro")

plt.clf()
ax = plt.figure().gca()
ax.set_xticks(np.arange(0,1,0.05))
ax.set_yticks(np.arange(0,1.,0.05))
plt.xticks(rotation='vertical')
plt.plot([0.,1.],[0.,1.], color='blue')
plt.plot(recallSede, precisionSede, color='gold', lw=2, label='micro-average Precision-recall curve (area = {0:0.2f})'''.format(average_precisionSede))

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
plt.plot(recallMorfo, precisionMorfo, color='gold', lw=2, label='micro-average Precision-recall curve (area = {0:0.2f})'''.format(average_precisionMorfo))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve Morfo')
plt.legend(loc="lower right")
plt.grid()
plt.savefig(filePrecRecallMorfo)

print("{0:.1f} sec - Learning Curve".format(time.time()-time0))
folds = 5

train_sizes, train_scores, test_scores = sl.model_selection.learning_curve(clfSede, termMatrix, targetSede, train_sizes=np.linspace(.1, 1.0, 10), cv=folds)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.clf()
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

plt.legend(loc="best")
plt.title("Sede, stratified {0:d}-fold cross validation".format(folds))
plt.savefig(fileLearningCurveSede)


train_sizes, train_scores, test_scores = sl.model_selection.learning_curve(clfSede, termMatrix, targetSede, train_sizes=np.linspace(.1, 1.0, 10), cv=folds)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.clf()
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

plt.legend(loc="best")
plt.title("Morfologia, stratified {0:d}-fold cross validation".format(folds))
plt.savefig(fileLearningCurveMorfo)


#notifier.sendMessage("Terminated "+timestamp+" execution ({0:.1f} sec)".format(time.time()-time0))

#print("{0:.1f} sec - End".format(time.time()-time0))

sys.exit()
#print("{0:.1f} - Score".format(time.time()-time0))
#scoreSede = clfSede.score(testMatrix, testTargetSede)
#scoreMorfologia = clfMorfo.score(testMatrix, testTargetMorfo)

print("{0:.1f} sec - Predict".format(time.time()-time0))
testPredictSede = clfSede.predict(testMatrixSede)
testPredictMorfo = clfMorfo.predict(testMatrixMorfo)

print("{0:.1f} sec - Metrics".format(time.time()-time0))
out_file = open(fileOut, 'w')

out_file.write("precision score sede micro average: ")
out_file.write(str(sl.metrics.precision_score(list(map(str, testTargetSede)), list(map(str, testPredictSede)), average='micro')))
out_file.write("\nrecall score sede micro average: ")
out_file.write(str(sl.metrics.recall_score(list(map(str, testTargetSede)), list(map(str, testPredictSede)), average='micro')))
out_file.write("\nf1 score sede micro average: ")
out_file.write(str(sl.metrics.f1_score(list(map(str, testTargetSede)), list(map(str, testPredictSede)), average='micro')))
out_file.write("\nprecision score sede macro average: ")
out_file.write(str(sl.metrics.precision_score(list(map(str, testTargetSede)), list(map(str, testPredictSede)), average='macro')))
out_file.write("\nrecall score sede macro average: ")
out_file.write(str(sl.metrics.recall_score(list(map(str, testTargetSede)), list(map(str, testPredictSede)), average='macro')))
out_file.write("\nf1 score sede macro average: ")
out_file.write(str(sl.metrics.f1_score(list(map(str, testTargetSede)), list(map(str, testPredictSede)), average='macro')))

out_file.write("\nprecision score morfologia micro average: ")
out_file.write(str(sl.metrics.precision_score(list(map(str, testTargetMorfo)), list(map(str, testPredictMorfo)), average='micro')))
out_file.write("\nrecall score morfologia micro average: ")
out_file.write(str(sl.metrics.recall_score(list(map(str, testTargetMorfo)), list(map(str, testPredictMorfo)), average='micro')))
out_file.write("\nf1 score morfologia micro average: ")
out_file.write(str(sl.metrics.f1_score(list(map(str, testTargetMorfo)), list(map(str, testPredictMorfo)), average='micro')))
out_file.write("\nprecision score morfologia macro average: ")
out_file.write(str(sl.metrics.precision_score(list(map(str, testTargetMorfo)), list(map(str, testPredictMorfo)), average='macro')))
out_file.write("\nrecall score morfologia macro average: ")
out_file.write(str(sl.metrics.recall_score(list(map(str, testTargetMorfo)), list(map(str, testPredictMorfo)), average='macro')))
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

