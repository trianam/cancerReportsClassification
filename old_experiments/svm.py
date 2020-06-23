import numpy as np
import scipy as sp
import sklearn as sk
import sklearn.svm
import sklearn.model_selection
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import math


class Svm:
    def __init__(self, randomState):
        self._randomState = randomState
        self._clf = sk.multiclass.OneVsRestClassifier(sk.svm.LinearSVC(random_state=self._randomState))
        #self._clf = sk.multiclass.OneVsRestClassifier(sk.svm.SVC(kernel='linear', probability=True, random_state=self._randomState))

    def setData(self, X, y, testSize=.1):
        self._X = X
        self._y = y
        self._trainX, self._testX, self._trainY, self._testY = sk.model_selection.train_test_split(X, y, test_size=testSize, random_state=self._randomState)

    def setDataSplitted(self, X, y, trainIndex, testIndex):
        self._X = X
        self._y = y
        self._trainX = X[trainIndex]
        self._testX = X[testIndex]
        self._trainY = y[trainIndex]
        self._testY = y[testIndex]

    def setLabelUnbinarizer(self, labelUnbinarizer):
        self._labelUnbinarizer = labelUnbinarizer

    def setCommonClasses(self, commonClasses, commonClassesIndexes):
        self._commonClasses = commonClasses
        self._commonClassesIndexes = commonClassesIndexes

    def setUncommonClasses(self, uncommonClasses, uncommonClassesIndexes):
        self._uncommonClasses = uncommonClasses
        self._uncommonClassesIndexes = uncommonClassesIndexes

    def train(self):
        self._clf.fit(self._trainX, self._trainY)
        self._yScore = self._clf.decision_function(self._testX)

    def predict(self, x):
        return self._clf.predict(x)

    def calculatePrecisionRecallCurve(self):
        return self._calculateMicroMacroCurve(lambda y,s: (lambda t: (t[1],t[0]))(sk.metrics.precision_recall_curve(y,s)))

    def calculateRocCurve(self):
        return self._calculateMicroMacroCurve(lambda y,s: (lambda t: (t[0],t[1]))(sk.metrics.roc_curve(y,s)))

    def calculateLearningCurve(self, fileNameLearningCurve, folds=5):
        train_sizes, train_scores, test_scores = sk.model_selection.learning_curve(self._clf, self._X, self._y, train_sizes=np.linspace(.1, 1.0, 10), cv=folds)

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
        plt.savefig(fileNameLearningCurve)

    def calculateMetrics(self):
        predY = self._clf.predict(self._testX)
        confusion = sk.metrics.confusion_matrix(np.array([self._labelUnbinarizer(np.array([y])) for y in self._testY]), self._labelUnbinarizer(predY))

        metrics = {
            'averaged' : {
                'accuracy' : sk.metrics.accuracy_score(self._testY, predY),
                'multiclassAccuracy' : np.sum(confusion.diagonal())/np.sum(confusion),
                'precision' : {
                    'micro' : sk.metrics.precision_score(self._testY, predY, average='micro'),
                    'macro' : sk.metrics.precision_score(self._testY, predY, average='macro'),
                    'weighted' : sk.metrics.precision_score(self._testY, predY, average='weighted')
                },
                'recall' : {
                    'micro' : sk.metrics.recall_score(self._testY, predY, average='micro'),
                    'macro' : sk.metrics.recall_score(self._testY, predY, average='macro'),
                    'weighted' : sk.metrics.recall_score(self._testY, predY, average='weighted')
                },
                'f1' : {
                    'micro' : sk.metrics.f1_score(self._testY, predY, average='micro'),
                    'macro' : sk.metrics.f1_score(self._testY, predY, average='macro'),
                    'weighted' : sk.metrics.f1_score(self._testY, predY, average='weighted')
                }
            },
            'commonClasses' : {},
            'uncommonClasses' : {}
        }

        for i in range(len(self._commonClasses)):
            testYSingle = self._testY[:, self._commonClassesIndexes[i]]
            predYSingle = predY[:, self._commonClassesIndexes[i]]

            metrics['commonClasses'][self._commonClasses[i]] = {
                'accuracy' : sk.metrics.accuracy_score(testYSingle, predYSingle),
                'precision' : sk.metrics.precision_score(testYSingle, predYSingle),
                'recall' : sk.metrics.recall_score(testYSingle, predYSingle),
                'f1' : sk.metrics.f1_score(testYSingle, predYSingle)
            }

        for i in range(len(self._uncommonClasses)):
            testYSingle = self._testY[:, self._uncommonClassesIndexes[i]]
            predYSingle = predY[:, self._uncommonClassesIndexes[i]]

            metrics['uncommonClasses'][self._uncommonClasses[i]] = {
                'accuracy' : sk.metrics.accuracy_score(testYSingle, predYSingle),
                'precision' : sk.metrics.precision_score(testYSingle, predYSingle),
                'recall' : sk.metrics.recall_score(testYSingle, predYSingle),
                'f1' : sk.metrics.f1_score(testYSingle, predYSingle)
            }



        return metrics

    def _calculateMicroMacroCurve(self, curveFunction):
        n_classes = self._y.shape[1]
        abscissa = dict()
        ordinate = dict()
        auc = dict()
        for i in range(n_classes):
            abscissa[i], ordinate[i] = curveFunction(self._testY[:, i], self._yScore[:, i])
            auc[i] = sk.metrics.auc(abscissa[i], ordinate[i])
        abscissa["micro"], ordinate["micro"] = curveFunction(self._testY.ravel(), self._yScore.ravel())
        auc["micro"] = sk.metrics.auc(abscissa["micro"], ordinate["micro"])
        # aggregate all
        all_rec = list(filter(lambda x: not math.isnan(x), np.unique(np.concatenate([abscissa[i] for i in range(n_classes)]))))

        # interpolate all prec/rec curves at this points
        mean_ordinate = np.zeros_like(all_rec)
        representedClasses = 0
        unrepresentedClasses = 0
        for i in range(n_classes):
            interp = sp.interpolate.interp1d(abscissa[i], ordinate[i])
            curr_ordinate = interp(all_rec)
            if not np.any([math.isnan(x) for x in abscissa[i]]) and not np.any([math.isnan(x) for x in ordinate[i]]):
                mean_ordinate += curr_ordinate
                representedClasses += 1
            else:
                unrepresentedClasses += 1

        # average it and compute AUC
        mean_ordinate /= representedClasses

        abscissa["macro"] = all_rec
        ordinate["macro"] = mean_ordinate
        auc["macro"] = sk.metrics.auc(abscissa["macro"], ordinate["macro"])

        return {'abscissa':abscissa, 'ordinate':ordinate, 'auc':auc}


