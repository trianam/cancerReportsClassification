import numpy as np
import scipy as sp
import sklearn as sk
import sklearn.model_selection

class CrossValidator:
    def __init__(self, randomState, numFolds):
        self._randomState = randomState
        self._cv = sk.model_selection.StratifiedKFold(n_splits = numFolds)
        self._numMisclassifiedAladar = 0
        self._numMisclassifiedNew = 0
        self._numMisclassifiedRespectAladar = 0
        self._numMisclassifiedRespectNew = 0

    def setSvm(self, svm):
        self._svm = svm

    def setData(self, X, y, yUnb, yAladar):
        self._X = X
        self._y = y
        self._yUnb = yUnb
        self._yAladar = yAladar

    def setLabelUnbinarizer(self, labelUnbinarizer):
        self._labelUnbinarizer = labelUnbinarizer
        self._svm.setLabelUnbinarizer(labelUnbinarizer)

    def setCommonClasses(self, commonClasses, commonClassesIndexes):
        self._commonClasses = commonClasses
        self._commonClassesIndexes = commonClassesIndexes
        self._svm.setCommonClasses(commonClasses, commonClassesIndexes)

    def setUncommonClasses(self, uncommonClasses, uncommonClassesIndexes):
        self._uncommonClasses = uncommonClasses
        self._uncommonClassesIndexes = uncommonClassesIndexes
        self._svm.setUncommonClasses(uncommonClasses, uncommonClassesIndexes)

    def validate(self):
        numPoints = 100
        meanCurves = {
            'precisionRecall': {
                'abscissa': {
                    'micro': np.linspace(0,1,numPoints),
                    'macro': np.linspace(0,1,numPoints)
                },
                'ordinate': {
                    'micro': 0.,
                    'macro': 0.
                },
                'auc': {
                    'micro': 0.,
                    'macro': 0.
                }
            },
            'roc': {
                'abscissa': {
                    'micro': np.linspace(0,1,numPoints),
                    'macro': np.linspace(0,1,numPoints)
                },
                'ordinate': {
                    'micro': 0.,
                    'macro': 0.
                },
                'auc': {
                    'micro': 0.,
                    'macro': 0.
                }
            }
        }

        meanMetrics = {}

        for trainIndex, testIndex in self._cv.split(self._X, self._yUnb):
            self._svm.setDataSplitted(self._X, self._y, trainIndex, testIndex)
            self._svm.train()

            meanMetrics = self._accumulateDict(meanMetrics, self._svm.calculateMetrics())

            predY = self._labelUnbinarizer(self._svm.predict(self._X[testIndex]))

            for iPred,i in enumerate(testIndex):
                if self._yUnb[i] != self._yAladar[i]:
                    self._numMisclassifiedAladar += 1
                    if predY[iPred] == self._yAladar[i]:
                        self._numMisclassifiedRespectNew += 1
                    else:
                        self._numMisclassifiedNew += 1

                elif predY[iPred] != self._yAladar[i]:
                    self._numMisclassifiedNew += 1
                    self._numMisclassifiedRespectAladar += 1

            currCurves = {
                'precisionRecall': self._svm.calculatePrecisionRecallCurve(),
                'roc': self._svm.calculateRocCurve()}
            
            for crv in ['precisionRecall', 'roc']:
                for avg in ['micro', 'macro']:
                    interp = sp.interpolate.interp1d(currCurves[crv]['abscissa'][avg], currCurves[crv]['ordinate'][avg])
                    meanCurves[crv]['ordinate'][avg] += interp(meanCurves[crv]['abscissa'][avg])
                    meanCurves[crv]['auc'][avg] += currCurves[crv]['auc'][avg]

        numSplits = self._cv.get_n_splits(self._X, self._yUnb)

        meanMetrics = self._divideDict(meanMetrics, numSplits)

        for crv in ['precisionRecall', 'roc']:
            for avg in ['micro', 'macro']:
                meanCurves[crv]['ordinate'][avg] /= numSplits
                meanCurves[crv]['auc'][avg] /= numSplits

        return (meanCurves, meanMetrics)

    def getNumMisclassifiedAladar(self):
        return self._numMisclassifiedAladar
    
    def getNumMisclassifiedNew(self):
        return self._numMisclassifiedNew
    
    def getNumMisclassifiedRespectAladar(self):
        return self._numMisclassifiedRespectAladar

    def getNumMisclassifiedRespectNew(self):
        return self._numMisclassifiedRespectNew

    _accumulateDict = lambda self,a,b: {k : ((a.get(k,0.) + b.get(k,0.)) if type(b.get(k,0.))!=dict else self._accumulateDict(a.get(k,{}), b.get(k,0.))) for k in set(b)}
    _divideDict = lambda self,a,b: {k : ((a.get(k,0.) / b) if type(a.get(k,0.))!=dict else self._divideDict(a.get(k,{}), b)) for k in set(a)}

