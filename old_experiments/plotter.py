import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

class Plotter:
    def plotPrecisionRecallCurve(self, fileNamePrecRecallCurve, values, oldCurves):
        self._plotCurve(fileNamePrecRecallCurve, values['abscissa'], values['ordinate'], values['auc'], oldCurves['rec'], oldCurves['prec'], 'Precision recall curve', 'Recall', 'Precision')

    def plotRocCurve(self, fileNameRocCurve, values, oldCurves):
        self._plotCurve(fileNameRocCurve, values['abscissa'], values['ordinate'], values['auc'], oldCurves['fpr'], oldCurves['tpr'], 'ROC curve', 'False positive rate', 'True positive rate')

    def plotCounter(self, fileName, counter):
        occurrences = np.array(counter.most_common())[:,1]
        plt.clf()
        plt.plot(range(len(occurrences)), occurrences)
        plt.xlabel("Class")
        plt.ylabel("Occurrences")
        plt.title("Occurrences of (filtered) classes in dataset")
        plt.savefig(fileName)

    def _plotCurve(self, fileName, abscissa, ordinate, auc, oldCurvesAbscissa, oldCurvesOrdinate, plotLabel, abscissaLabel, ordinateLabel):
        plt.clf()
        #for i in [k for k in abscissa.keys() if k != 'micro' and k != 'macro]:
        #    plt.plot(abscissa[i], ordinate[i], color='black')

        plt.plot([0, 1], [0, 1], color='blue')
        plt.plot(abscissa["micro"], ordinate["micro"], label='micro-average curve (area = {0:0.2f})'''.format(auc["micro"]), color='gold', linewidth=2)
        plt.plot(abscissa["macro"], ordinate["macro"], label='macro-average curve (area = {0:0.2f})'''.format(auc["macro"]), color='red', linewidth=2)
        plt.plot(oldCurvesAbscissa['micro'], oldCurvesOrdinate['micro'], color='green', marker='o', label='micro-average old')
        plt.plot(oldCurvesAbscissa['macro'], oldCurvesOrdinate['macro'], color='blue', marker='o', label='macro-average old')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel(abscissaLabel)
        plt.ylabel(ordinateLabel)
        plt.title(plotLabel)
        plt.legend(loc="lower right")
        plt.savefig(fileName)

