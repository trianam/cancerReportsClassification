import numpy as np
import scipy as sp
import pandas as pd
import csv
import sklearn as sk
import sklearn.feature_extraction.text 
import nltk
import random
import collections

class Importer:
    #encoding = 'iso-8859-1' or encoding = 'cp1252' or encoding = 'latin1'
    _encoding = 'iso-8859-1'
    _tokenPattern = '(?u)\\b\\w*[a-zA-Z_][a-zA-Z_]+\\w*\\b'

    def __init__(self, statisticLogger, parameterLogger, plotter, cutOff=3, maxDf=0.5, mostCommonClassesNum=10, leastCommonClassesNum=10, useStemmer=True):
        self._statisticLogger = statisticLogger
        self._plotter = plotter
        self._cutOff = cutOff
        self._maxDf = maxDf
        self._mostCommonClassesNum = mostCommonClassesNum
        self._leastCommonClassesNum = leastCommonClassesNum
        self._useStemmer = useStemmer

        parameterLogger.cutOff = cutOff
        parameterLogger.maxDf = maxDf
        parameterLogger.mostCommonClassesNum = mostCommonClassesNum
        parameterLogger.leastCommonClassesNum = leastCommonClassesNum
        parameterLogger.useStemmer = useStemmer

        self._stemmer = nltk.stem.snowball.ItalianStemmer(ignore_stopwords=True)
        analyzer = sk.feature_extraction.text.TfidfVectorizer(analyzer='word', token_pattern=self._tokenPattern).build_analyzer()
        self._modAnalyzer = lambda doc: (self._stemmer.stem(w) for w in analyzer(doc))

    def importCsv(self, fileIsto, fileNeop):
        dfIsto = pd.read_csv(fileIsto, header = 0, encoding=self._encoding, quoting=csv.QUOTE_NONNUMERIC, low_memory=False, usecols = ['notizie', 'diagnosi', 'macroscopia', 'ICDO1_T', 'ICDO1_M', 'id_neopl'], dtype={'id_neopl':pd.np.float64, 'ICDO1_T':pd.np.str, 'ICDO1_M':pd.np.str})#, converters={'ICDO1_T':str, 'ICDO1_M':str})
        dfNeop = pd.read_csv(fileNeop, header = 0, encoding=self._encoding, quoting=csv.QUOTE_NONNUMERIC, low_memory=False, usecols = ['id_neopl', 'sede', 'morfologia'], dtype={'id_neopl':pd.np.float64, 'sede':pd.np.str, 'morfologia':pd.np.str})

        self._df = pd.merge(dfIsto, dfNeop, on='id_neopl')

        nullToEmpty = lambda s: "" if pd.isnull(s) else str(s)
        self._df['notizie'] = self._df.notizie.map(nullToEmpty)
        self._df['diagnosi'] = self._df.diagnosi.map(nullToEmpty)
        self._df['macroscopia'] = self._df.macroscopia.map(nullToEmpty)

    def cutDataset(self, numSamples):
        self._df = self._df.ix[random.sample(list(self._df.index), numSamples)]
 
    def filterClasses(self, fileClassesSedeOrig, fileClassesMorfoOrig, fileClassesSedeFiltered, fileClassesMorfoFiltered, filePlotClassesSede, filePlotClassesMorfo, cutoffRatio=0.001):
        self._dfSede, self._commonClassesSede, self._uncommonClassesSede, numDocumentsSede, numDocumentsFiltSede, numClassesSede, numClassesFiltSede = self._filterClasses(self._df, self._df.sede.tolist(), fileClassesSedeOrig, fileClassesSedeFiltered, filePlotClassesSede, cutoffRatio)
        self._dfMorfo, self._commonClassesMorfo, self._uncommonClassesMorfo, numDocumentsMorfo, numDocumentsFiltMorfo, numClassesMorfo, numClassesFiltMorfo  = self._filterClasses(self._df, self._df.morfologia.tolist(), fileClassesMorfoOrig, fileClassesMorfoFiltered, filePlotClassesMorfo, cutoffRatio)

        self._statisticLogger.numDocuments = numDocumentsSede
        self._statisticLogger.numDocumentsDelSede = numDocumentsSede - numDocumentsFiltSede
        self._statisticLogger.numDocumentsDelMorfo = numDocumentsMorfo - numDocumentsFiltMorfo

        self._statisticLogger.numClassesSede = numClassesSede
        self._statisticLogger.numClassesMorfo = numClassesMorfo
        self._statisticLogger.numClassesDelSede = numClassesSede - numClassesFiltSede
        self._statisticLogger.numClassesDelMorfo = numClassesMorfo - numClassesFiltMorfo
        

    def tokenize(self, fileTokensNotizieSede, fileTokensDiagnosiSede, fileTokensMacroscopiaSede, fileTokensNotizieMorfo, fileTokensDiagnosiMorfo, fileTokensMacroscopiaMorfo):
        if self._useStemmer:
            vecNotizie = sk.feature_extraction.text.TfidfVectorizer(min_df=self._cutOff, max_df=self._maxDf, encoding=self._encoding, strip_accents='unicode', analyzer=self._modAnalyzer, stop_words=self._stemmer.stopwords)
            vecDiagnosi = sk.feature_extraction.text.TfidfVectorizer(min_df=self._cutOff, max_df=self._maxDf, encoding=self._encoding, strip_accents='unicode', analyzer=self._modAnalyzer, stop_words=self._stemmer.stopwords)
            vecMacroscopia = sk.feature_extraction.text.TfidfVectorizer(min_df=self._cutOff, max_df=self._maxDf, encoding=self._encoding, strip_accents='unicode', analyzer=self._modAnalyzer, stop_words=self._stemmer.stopwords)
        else:
            vecNotizie = sk.feature_extraction.text.TfidfVectorizer(min_df=self._cutOff, max_df=self._maxDf, encoding=self._encoding, strip_accents='unicode')
            vecDiagnosi = sk.feature_extraction.text.TfidfVectorizer(min_df=self._cutOff, max_df=self._maxDf, encoding=self._encoding, strip_accents='unicode')
            vecMacroscopia = sk.feature_extraction.text.TfidfVectorizer(min_df=self._cutOff, max_df=self._maxDf, encoding=self._encoding, strip_accents='unicode')
        
        self._termMatrixSede = sp.sparse.hstack([vecNotizie.fit_transform(self._dfSede.notizie), vecDiagnosi.fit_transform(self._dfSede.diagnosi), vecMacroscopia.fit_transform(self._dfSede.macroscopia)]).tocsr()
        np.savetxt(fileTokensNotizieSede, vecNotizie.get_feature_names(), '%s')
        np.savetxt(fileTokensDiagnosiSede, vecDiagnosi.get_feature_names(), '%s')
        np.savetxt(fileTokensMacroscopiaSede, vecMacroscopia.get_feature_names(), '%s')
        self._statisticLogger.numTermsNotizieSede = len(vecNotizie.get_feature_names())
        self._statisticLogger.numTermsDiagnosiSede = len(vecDiagnosi.get_feature_names())
        self._statisticLogger.numTermsMacroscopiaSede = len(vecMacroscopia.get_feature_names())
        
        self._termMatrixMorfo = sp.sparse.hstack([vecNotizie.fit_transform(self._dfMorfo.notizie), vecDiagnosi.fit_transform(self._dfMorfo.diagnosi), vecMacroscopia.fit_transform(self._dfMorfo.macroscopia)]).tocsr()
        np.savetxt(fileTokensNotizieMorfo, vecNotizie.get_feature_names(), '%s')
        np.savetxt(fileTokensDiagnosiMorfo, vecDiagnosi.get_feature_names(), '%s')
        np.savetxt(fileTokensMacroscopiaMorfo, vecMacroscopia.get_feature_names(), '%s')
        self._statisticLogger.numTermsNotizieMorfo = len(vecNotizie.get_feature_names())
        self._statisticLogger.numTermsDiagnosiMorfo = len(vecDiagnosi.get_feature_names())
        self._statisticLogger.numTermsMacroscopiaMorfo = len(vecMacroscopia.get_feature_names())



    def binarize(self):
        self._targetSedeUnb = np.array(self._dfSede.sede, dtype=str)
        self._targetMorfoUnb = np.array(self._dfMorfo.morfologia, dtype=str)
        self._aladarResultSede = np.array(self._dfSede.ICDO1_T, dtype=str)
        self._aladarResultMorfo = np.array(self._dfMorfo.ICDO1_M, dtype=str)

        self._targetSedeBin, self._binarizerSede, self._unbinarizerSede = self._binarize(self._targetSedeUnb)
        self._targetMorfoBin, self._binarizerMorfo, self._unbinarizerMorfo  = self._binarize(self._targetMorfoUnb)
        self._commonClassesIndexesSede = np.where(self._binarizerSede(self._commonClassesSede)==1)[1]
        self._commonClassesIndexesMorfo = np.where(self._binarizerMorfo(self._commonClassesMorfo)==1)[1]

        self._uncommonClassesIndexesSede = np.where(self._binarizerSede(self._uncommonClassesSede)==1)[1]
        self._uncommonClassesIndexesMorfo = np.where(self._binarizerMorfo(self._uncommonClassesMorfo)==1)[1]

    def getXY(self):
        return (self._termMatrixSede, self._termMatrixMorfo, self._targetSedeBin, self._targetMorfoBin, self._targetSedeUnb, self._targetMorfoUnb)

    def getYAladar(self):
        return (self._aladarResultSede, self._aladarResultMorfo)

    def getLabelBinarizers(self):
        return (self._binarizerSede, self._binarizerMorfo)

    def getLabelUnbinarizers(self):
        return (self._unbinarizerSede, self._unbinarizerMorfo)

    def getCommonClasses(self):
        return (self._commonClassesSede, self._commonClassesMorfo)

    def getUncommonClasses(self):
        return (self._uncommonClassesSede, self._uncommonClassesMorfo)

    def getCommonClassesIndexes(self):
        return (self._commonClassesIndexesSede, self._commonClassesIndexesMorfo)

    def getUncommonClassesIndexes(self):
        return (self._uncommonClassesIndexesSede, self._uncommonClassesIndexesMorfo)

    def getOldMetrics(self):
        curvesSede, metricsSede = self._getOldMetrics(np.array(self._df.sede.map(str)), np.array(self._df.ICDO1_T.map(str)), self._commonClassesSede, self._uncommonClassesSede)
        curvesMorfo, metricsMorfo =  self._getOldMetrics(np.array(self._df.morfologia.map(str)), np.array(self._df.ICDO1_M.map(str)), self._commonClassesMorfo, self._uncommonClassesMorfo)

        return (
            {'sede' : curvesSede, 'morfo' : curvesMorfo},
            {'sede' : metricsSede, 'morfo' : metricsMorfo}
        )

    def _getOldMetrics(self, truth, predict, commonClasses, uncommonClasses):
        labelEncoder = sk.preprocessing.LabelEncoder()
        labelBinarizer = sk.preprocessing.LabelBinarizer()

        labelEncoder.fit(np.append(truth, predict))
        truthE = labelEncoder.transform(truth)
        predictE = labelEncoder.transform(predict)

        labelBinarizer.fit(np.append(truthE, predictE))
        truthB = labelBinarizer.transform(truthE)
        predictB = labelBinarizer.transform(predictE)

        commonClassesIndexes = np.where(labelBinarizer.transform(labelEncoder.transform(commonClasses))==1)[1]
        uncommonClassesIndexes = np.where(labelBinarizer.transform(labelEncoder.transform(uncommonClasses))==1)[1]

        cm = sk.metrics.confusion_matrix(truth, predict)
        accuracy = np.sum(cm.diagonal())/np.sum(cm)
        numClasses = cm.shape[0]
        numRows = len(truth)

        tp = np.zeros(numClasses)
        fp = np.zeros(numClasses)
        tn = np.zeros(numClasses)
        fn = np.zeros(numClasses)

        for i in range(numClasses):
            mask = np.arange(numClasses)!=i
            tp[i] = cm[i,i]
            fp[i] = np.sum(cm[mask, i])
            fn[i] = np.sum(cm[i, mask])
            tn[i] = np.sum(cm[mask,:][:,mask])
        
        tpMean = 0
        fpMean = 0
        tnMean = 0
        fnMean = 0

        tprMacro = 0
        fprMacro = 0
        precMacro = 0
        recMacro = 0

        for i in range(numClasses):
            tpMean += tp[i]
            fpMean += fp[i]
            tnMean += tn[i]
            fnMean += fn[i]

            if (tp[i]+fn[i]) > 0:
                tprMacro += tp[i]/(tp[i]+fn[i])
            if (fp[i]+tn[i]) > 0:
                fprMacro += fp[i]/(fp[i]+tn[i])
            if (tp[i]+fp[i]) > 0:
                precMacro += tp[i]/(tp[i]+fp[i])
            if (tp[i]+fn[i]) > 0:
                recMacro += tp[i]/(tp[i]+fn[i])

        tpMean /= numClasses
        fpMean /= numClasses
        tnMean /= numClasses
        fnMean /= numClasses

        tprMacro /= numClasses
        fprMacro /= numClasses
        precMacro /= numClasses
        recMacro /= numClasses

        if (tpMean+fnMean) > 0:
            tprMicro = tpMean/(tpMean+fnMean)
        if (fpMean+tnMean) > 0:
            fprMicro = fpMean/(fpMean+tnMean)
        if (tpMean+fpMean) > 0:
            precMicro = tpMean/(tpMean+fpMean)
        if (tpMean+fnMean) > 0:
            recMicro = tpMean/(tpMean+fnMean)

        curves = {
            'fpr' : {
                'micro':fprMicro,
                'macro':fprMacro
            },
            'tpr' : {
                'micro' : tprMicro,
                'macro':tprMacro
            },
            'rec' : {
                'micro' : recMicro,
                'macro':recMacro
            },
            'prec' : {
                'micro' : precMicro,
                'macro':precMacro
            },
            'accuracy' : accuracy
        }
        
        metrics = {
            'averaged' : {
                'accuracy' : sk.metrics.accuracy_score(truthB, predictB),
                'multiclassAccuracy' : accuracy,
                'precision' : {
                    'micro' : sk.metrics.precision_score(truthB, predictB, average='micro'),
                    'macro' : sk.metrics.precision_score(truthB, predictB, average='macro'),
                    'weighted' : sk.metrics.precision_score(truthB, predictB, average='weighted')
                },
                'recall' : {
                    'micro' : sk.metrics.recall_score(truthB, predictB, average='micro'),
                    'macro' : sk.metrics.recall_score(truthB, predictB, average='macro'),
                    'weighted' : sk.metrics.recall_score(truthB, predictB, average='weighted')
                },
                'f1' : {
                    'micro' : sk.metrics.f1_score(truthB, predictB, average='micro'),
                    'macro' : sk.metrics.f1_score(truthB, predictB, average='macro'),
                    'weighted' : sk.metrics.f1_score(truthB, predictB, average='weighted')
                }
            },
            'commonClasses' : {},
            'uncommonClasses' : {}
        }

        for i in range(len(commonClasses)):
            truthSingle = truthB[:, commonClassesIndexes[i]]
            predictSingle = predictB[:, commonClassesIndexes[i]]

            metrics['commonClasses'][commonClasses[i]] = {
                'accuracy' : sk.metrics.accuracy_score(truthSingle, predictSingle),
                'precision' : sk.metrics.precision_score(truthSingle, predictSingle),
                'recall' : sk.metrics.recall_score(truthSingle, predictSingle),
                'f1' : sk.metrics.f1_score(truthSingle, predictSingle)
            }

        for i in range(len(uncommonClasses)):
            truthSingle = truthB[:, uncommonClassesIndexes[i]]
            predictSingle = predictB[:, uncommonClassesIndexes[i]]

            metrics['uncommonClasses'][uncommonClasses[i]] = {
                'accuracy' : sk.metrics.accuracy_score(truthSingle, predictSingle),
                'precision' : sk.metrics.precision_score(truthSingle, predictSingle),
                'recall' : sk.metrics.recall_score(truthSingle, predictSingle),
                'f1' : sk.metrics.f1_score(truthSingle, predictSingle)
            }

        return (curves, metrics)

    def _filterClasses(self, df, targetOrig, fileClassesOrig, fileClassesFiltered, filePlotClasses, cutoffRatio):
        counter = collections.Counter(targetOrig)
        self._writeCounter(counter, fileClassesOrig)
        cutTreshold = counter.most_common(1)[0][1] * cutoffRatio
        targetFiltered = []

        orig = np.array(df)
        filt = []

        for i,k in enumerate(targetOrig):
            if k!='nan' and counter[k] > cutTreshold:
                targetFiltered.append(targetOrig[i])
                filt.append(orig[i])

        counterF = collections.Counter(targetFiltered)
        self._writeCounter(counterF, fileClassesFiltered)
        self._plotter.plotCounter(filePlotClasses, counterF)

        return(pd.DataFrame(filt, columns=df.columns), np.array(counterF.most_common(self._mostCommonClassesNum))[:,0], np.array(counterF.most_common())[-self._leastCommonClassesNum:,0], len(orig), len(filt), len(counter), len(counterF))

    def _binarize(self, target):
        labelEncoder = sk.preprocessing.LabelEncoder()
        targetEncoded = labelEncoder.fit_transform(target)

        labelBinarizer = sk.preprocessing.LabelBinarizer()
        
        return (
            labelBinarizer.fit_transform(targetEncoded),
            lambda y: labelBinarizer.transform(labelEncoder.transform(y)),
            lambda y: labelEncoder.inverse_transform(labelBinarizer.inverse_transform(y))
        )

    def _writeCounter(self, counter, fileName):
        with open(fileName, 'w') as handler:
            for k,v in counter.most_common():
                handler.write("{}, {}\n".format(k,v))

