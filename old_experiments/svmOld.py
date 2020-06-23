import time
import datetime
import numpy as np
import scipy as sp
import pandas as pd
import csv
import sklearn as sl
import sklearn.feature_extraction.text 
import sklearn.svm
import sys
import notifier
import nltk

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

#encoding = 'iso-8859-1' or encoding = 'cp1252' or encoding = 'latin1'
encoding = 'iso-8859-1'
cutOff = 3
tokenPattern = '(?u)\\b\\w*[a-zA-z_][a-zA-Z_]+\\w*\\b'

stemmer = nltk.stem.snowball.ItalianStemmer(ignore_stopwords=True)
analyzer = sl.feature_extraction.text.CountVectorizer(analyzer='word', token_pattern=tokenPattern).build_analyzer()
modAnalyzer = lambda doc: (stemmer.stem(w) for w in analyzer(doc))

time0 = time.time()

print("{0:.1f} sec - Importing csv".format(time.time()-time0))
dfIsto = pd.read_csv(fileIsto, header = 0, encoding=encoding, quoting=csv.QUOTE_NONNUMERIC, low_memory=False, usecols = ['notizie', 'diagnosi', 'macroscopia', 'ICDO1_T', 'ICDO1_M', 'id_neopl'], dtype={'id_neopl':pd.np.float64, 'ICDO1_T':pd.np.str, 'ICDO1_M':pd.np.str})#, converters={'ICDO1_T':str, 'ICDO1_M':str})
dfNeop = pd.read_csv(fileNeop, header = 0, usecols = ['id_neopl', 'sede', 'morfologia'], low_memory=False, dtype={'id_neopl':pd.np.float64, 'sede':pd.np.str, 'morfologia':pd.np.str})

print("{0:.1f} sec - Merging csv".format(time.time()-time0))
dfMerge = pd.merge(dfIsto, dfNeop, on='id_neopl')

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
half = int(termMatrix.shape[0] / 10. * 9.)

trainMatrix = termMatrix.tocsr()[:half,:]
testMatrix = termMatrix.tocsr()[half:,:]

trainTargetSede = dfMerge.sede.tolist()[:half]
trainTargetMorfo = dfMerge.morfologia.tolist()[:half]
testTargetSede = dfMerge.sede.tolist()[half:]
testTargetMorfo = dfMerge.morfologia.tolist()[half:]

clfSede = sklearn.svm.SVC()
clfMorfo = sklearn.svm.SVC()

clfSede.fit(trainMatrix, trainTargetSede)
clfMorfo.fit(trainMatrix, trainTargetMorfo)

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

