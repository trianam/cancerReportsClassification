import math
import pickle
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score, precision_recall_fscore_support, f1_score, accuracy_score, cohen_kappa_score
from scipy.interpolate import interp1d
from scipy import interp

goodModelNumbers = {
    'LSTMng' : {
        'sede1' : 5,
        'sede12' : 5,
        'morfo1' : 4,
        'morfo2' : 2,
    },
    'LSTMcv' : {
        'sede1' : 9,
        'sede12' : 9,
        'morfo1' : 9,
        'morfo2' : 9,
    },
    'LSTMb' : {
        'sede1' : 7,
        'sede12' : 9,
        'morfo1' : 6,
        'morfo2' : 3,
    },
}

fileFolderNames = {
    "SVM" : "filesFolds-SVM",
    "SVMb" : "filesFolds-SVMbigrams",
    "LSTMng" : "filesFolds-LSTMnoGloVe",
    "LSTMcv" : "filesFolds-LSTMconvolutional2",
    "LSTMb" : "filesFolds-LSTMbidirectional5e",
}


#firstFold = 3
#numFolds = 2
firstFold = 0
numFolds = 10
models = ['SVM', 'SVMb', 'LSTMng', 'LSTMcv', 'LSTMb']
svmModels = ['SVM', 'SVMb']

tasks = ['sede1', 'sede12', 'morfo1', 'morfo2']
#tasks = ['sede12']

simpleLabels = ['MAPs', 'MAPc', 'accuracy', 'kappa']
avgLabels = ['precision', 'recall', 'f1score']
avgModes = ['micro', 'macro', 'weighted']
curvesLabels = ['pr-curve', 'roc-curve']
curvesAvgModes = ['micro', 'macro']
axisLabels = ['x', 'y']
plotLen = 1001

plotX = np.linspace(0., 1., plotLen)

def _calculateMicroMacroCurve(curveFunction, yt, yp):
    n_classes = yt.shape[1]
    abscissa = dict()
    ordinate = dict()
    area = dict()
    for i in range(n_classes):
        abscissa[i], ordinate[i] = curveFunction(yt[:, i], yp[:, i])
        area[i] = auc(abscissa[i], ordinate[i])
    abscissa["micro"], ordinate["micro"] = curveFunction(yt.ravel(), yp.ravel())
    area["micro"] = auc(abscissa["micro"], ordinate["micro"])
    # aggregate all
    all_rec = list(filter(lambda x: not math.isnan(x), np.unique(np.concatenate([abscissa[i] for i in range(n_classes)]))))

    # interpolate all prec/rec curves at this points
    mean_ordinate = np.zeros_like(all_rec)
    representedClasses = 0
    unrepresentedClasses = 0
    for i in range(n_classes):
        interp = interp1d(abscissa[i], ordinate[i])
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
    area["macro"] = auc(abscissa["macro"], ordinate["macro"])

    return (abscissa, ordinate, area)




for model in models:
    print("====================== Model {}".format(model))
    filesFolder = "./"+fileFolderNames[model]+"/output"

    curves = {}
    for t in tasks:
        curves[t] = {}
        yt = []
        yp = []
        for fold in range(10):
            filesFolderFold = filesFolder+"/"+str(fold)

            currPredFile = filesFolderFold+"/prediction"+t.capitalize()+".p"
            currPred = pickle.load(open(currPredFile, "rb"))
            if model in svmModels:
                yt.extend(currPred['yt'][t])
                yp.extend(currPred['yp'][t])
            else:
                yt.extend(currPred['yt'])
                yp.extend(currPred['yp'])


        yt = np.array(yt)
        yp = np.array(yp)

        curves[t]['pr-curve'] = {}
        curves[t]['pr-curve']['x'], curves[t]['pr-curve']['y'], curves[t]['pr-curve']['auc'] = _calculateMicroMacroCurve(lambda y,s: (lambda t: (t[1],t[0]))(precision_recall_curve(y,s)), yt, yp)
            
        curves[t]['roc-curve'] = {}
        curves[t]['roc-curve']['x'], curves[t]['roc-curve']['y'], curves[t]['roc-curve']['auc'] = _calculateMicroMacroCurve(lambda y,s: (lambda t: (t[0],t[1]))(roc_curve(y,s)), yt, yp)

    pickle.dump(curves, open(filesFolder+"/curvesAll.p", 'wb'))

