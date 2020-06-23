import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import auc
import pickle
import math
from tabulate import tabulate

def accumulateMeanSD(mean, sd, x, k): 
    delta = x - mean
    newMean = mean + delta / (k+1)
    newSd = sd + delta * (x - newMean)

    return (newMean, newSd)


goodModelNumbers = {
    "sede1":{
        "LSTMng" : 5,
        "LSTMcv" : 9,
        "LSTMb" : 7,
        "LSTMbnn" : 7,
    },
    "sede12":{
        "LSTMng" : 5,
        "LSTMcv" : 9,
        "LSTMb" : 9,
        "LSTMbnn" : 9,
    },
    "morfo1":{
        "LSTMng" : 4,
        "LSTMcv" : 9,
        "LSTMb" : 6,
        "LSTMbnn" : 6,
    },
    "morfo2":{
        "LSTMng" : 2,
        "LSTMcv" : 9,
        "LSTMb" : 3,
        "LSTMbnn" : 3,
    },
}


fileFolderNames = {
    "SVM" : "filesFolds-SVM",
    "SVMb" : "filesFolds-SVMbigrams",
    "SVMs" : "filesFolds-SVMbigramsSeparated",
    "LSTMng" : "filesFolds-LSTMnoGloVe",
    "LSTMcv" : "filesFolds-LSTMconvolutional2",
    "LSTMb" : "filesFolds-LSTMbidirectional5e",
    "LSTMbnn" : "filesFolds-LSTMbidirectional5eNoNegation",
}


#firstFold = 3
#numFolds = 2
firstFold = 0
numFolds = 10
models = ['SVM', 'SVMb', 'SVMs', 'LSTMng', 'LSTMcv', 'LSTMb', 'LSTMbnn']
#models = ['LSTMbnn']
svmModels = ['SVM', 'SVMb', 'SVMs']

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

for model in models:
    print("====================== Model {}".format(model))

    filesFolder = "./"+fileFolderNames[model]+"/output"

    mergedFile = filesFolder+"/evaluationMean.p"
    mergedFileTab = filesFolder+"/evaluationMean.txt"

    metrics = {}
    for t in tasks:
        metrics[t] = {}
        for l in simpleLabels:
            metrics[t][l] = {'mean':0, 'sd':0}
        for l in avgLabels:
            metrics[t][l] = {}
            for m in avgModes:
                metrics[t][l][m] = {'mean':0, 'sd':0}
        for l in curvesLabels:
            metrics[t][l] = {'x':list(plotX), 'y':{}, 'auc':{}, 'xAll':{}, 'yAll':{}, 'aucAll':{}}
            for m in curvesAvgModes:
                metrics[t][l]['y'][m] = {'mean':np.zeros(plotLen), 'sd':np.zeros(plotLen)}
                metrics[t][l]['auc'][m] = {'mean':0, 'sd':0}
                metrics[t][l]['xAll'][m] = []
                metrics[t][l]['yAll'][m] = []
                metrics[t][l]['aucAll'][m] = 0.

    for foldNumber,fold in enumerate(range(firstFold, firstFold+numFolds)):
        print("Elaborating fold {}".format(fold))
        if model in svmModels:
            evaluationFile = filesFolder+"/"+str(fold)+"/evaluation.p"
            currMetricsAll = pickle.load(open(evaluationFile, 'rb'))

        for t in tasks:
            if model in svmModels:
                currMetrics = currMetricsAll[t]
            else:
                evaluationFile = filesFolder+"/"+str(fold)+"/evaluation"+t.capitalize()+"-"+str(goodModelNumbers[t][model])+".p"
                currMetrics = pickle.load(open(evaluationFile, 'rb'))

            for l in simpleLabels:
                metrics[t][l]['mean'], metrics[t][l]['sd'] = accumulateMeanSD(metrics[t][l]['mean'], metrics[t][l]['sd'], currMetrics[l], foldNumber)
            for l in avgLabels:
                for m in avgModes:
                    metrics[t][l][m]['mean'], metrics[t][l][m]['sd'] = accumulateMeanSD(metrics[t][l][m]['mean'], metrics[t][l][m]['sd'], currMetrics[l][m], foldNumber)
 
            for l in curvesLabels:
                for m in curvesAvgModes:
                    interp = interp1d(currMetrics[l]['x'][m], currMetrics[l]['y'][m])
                    interpY = interp(plotX)

                    metrics[t][l]['y'][m]['mean'], metrics[t][l]['y'][m]['sd'] = accumulateMeanSD(metrics[t][l]['y'][m]['mean'], metrics[t][l]['y'][m]['sd'], interpY, foldNumber)  
                    metrics[t][l]['auc'][m]['mean'], metrics[t][l]['auc'][m]['sd'] = accumulateMeanSD(metrics[t][l]['auc'][m]['mean'], metrics[t][l]['auc'][m]['sd'], currMetrics[l]['auc'][m], foldNumber)
  
                    metrics[t][l]['xAll'][m].extend(currMetrics[l]['x'][m])
                    metrics[t][l]['yAll'][m].extend(currMetrics[l]['y'][m])

    for t in tasks:
        for l in simpleLabels:
            metrics[t][l]['sd'] = math.sqrt(metrics[t][l]['sd'] / (numFolds-1))
        for l in avgLabels:
            for m in avgModes:
                metrics[t][l][m]['sd'] = math.sqrt(metrics[t][l][m]['sd'] / (numFolds-1))
        for l in curvesLabels:
            for m in curvesAvgModes:
                metrics[t][l]['y'][m]['sd'] = list((metrics[t][l]['y'][m]['sd'] / (numFolds-1)))
                metrics[t][l]['y'][m]['mean'] = list(metrics[t][l]['y'][m]['mean'])
                
                metrics[t][l]['auc'][m]['sd'] = math.sqrt(metrics[t][l]['auc'][m]['sd'] / (numFolds-1))

                sort = np.argsort(metrics[t][l]['xAll'][m])
                metrics[t][l]['xAll'][m] = np.array(metrics[t][l]['xAll'][m])[sort]
                metrics[t][l]['yAll'][m] = np.array(metrics[t][l]['yAll'][m])[sort]
                metrics[t][l]['aucAll'][m] = auc(metrics[t][l]['xAll'][m], metrics[t][l]['yAll'][m])

    pickle.dump(metrics, open(mergedFile, 'wb'))


    table = [["task", "average", "MAPs", "MAPc", "accur.", "kappa", "prec.", "recall", "f1score"]]
    na = ' '        
    for t in tasks:
        table.append([" ", " ", " ", " ", " ", " ", " ", " "])
        table.append([t, na, "{:.3f} ± {:.3f}".format(metrics[t]['MAPs']['mean'],metrics[t]['MAPs']['sd']), "{:.3f} ± {:.3f}".format(metrics[t]['MAPc']['mean'],metrics[t]['MAPc']['sd']), "{:.3f} ± {:.3f}".format(metrics[t]['accuracy']['mean'],metrics[t]['accuracy']['sd']), "{:.3f} ± {:.3f}".format(metrics[t]['kappa']['mean'],metrics[t]['kappa']['sd']), na, na, na])
        for avg in ['micro', 'macro', 'weighted']:
            table.append([t, avg, na, na, na, na, "{:.3f} ± {:.3f}".format(metrics[t]['precision'][avg]['mean'],metrics[t]['precision'][avg]['sd']), "{:.3f} ± {:.3f}".format(metrics[t]['recall'][avg]['mean'],metrics[t]['recall'][avg]['sd']), "{:.3f} ± {:.3f}".format(metrics[t]['f1score'][avg]['mean'],metrics[t]['f1score'][avg]['sd'])])

    textTab = tabulate(table)
    with open(mergedFileTab, 'wt') as f:
        f.write(textTab)
    print(textTab)
