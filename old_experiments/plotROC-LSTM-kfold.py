#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
from tabulate import tabulate
from scipy.interpolate import interp1d
import numpy as np
import math
from MAPScorer import MAPScorer
from scipy import interp
from myLSTMKfold import MyLSTM
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score, precision_recall_fscore_support, f1_score, accuracy_score, cohen_kappa_score
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt

outputPlotDir = "plots/lstmKfold"

lstm = MyLSTM()
lstm.loadData()
lstm.loadModels()

print("========================= Prediction")
tasks = ['sede1', 'sede12', 'sede2ft3', 'morfo1', 'morfo2']
ftTasks = ['sede2ft3']
yp = {}
ycn = {}
yc = {}
ytn = {}
yt = {}
for fold in range(lstm.stratifications):
#for fold in range(1):
    print("============ fold {}".format(fold))
    yp[fold] = {}
    ycn[fold] = {}
    yc[fold] = {}
    ytn[fold] = {}
    yt[fold] = {}
    for task in tasks:
    #for task in tasks[:1]:
        print("-------- task {}".format(task))
        XTest = lstm.getXTest(fold, task)
        yp[fold][task] = np.memmap("./tmp/yp-"+str(fold)+"-"+str(task)+".dat", mode='w+', shape=(XTest.shape[0], lstm.model[fold][task].output_shape[1]), dtype=np.float) 

        if not task in ftTasks:
            yp[fold][task][:] = lstm.model[fold][task].predict_proba(XTest)
            ycn[fold][task] = lstm.model[fold][task].predict_classes(XTest)
        else:
            XTestFT = lstm.getXTestFT(fold, task)
            yp[fold][task][:] = lstm.model[fold][task].predict_proba([XTestFT, XTest])
            ycn[fold][task] = lstm.model[fold][task].predict_classes([XTestFT, XTest])

        if not task in ftTasks:
            yt[fold][task] = lstm.getYTestName(fold, task, "./tmp/yt-"+str(fold)+"-"+str(task)+".dat")

            ytn[fold][task] = np.zeros_like(ycn[fold][task])
            for i,v in enumerate(yt[fold][task]):
                ytn[fold][task][i] = np.nonzero(yt[fold][task][i])[0][0]

        else:
            yt[fold][task] = yt[fold]['sede12']
            ytn[fold][task] = ytn[fold]['sede12']

        yc[fold][task] = np.zeros_like(yt[fold][task])
        for i,v in enumerate(ycn[fold][task]):
            yc[fold][task][i][v] = 1
   

print("========================= Calculation")
mapScorer = MAPScorer()
table = [["task", "average", "MAPs", "MAPc", "accuracy", "kappa", "precision", "recall", "f1score"]]
na = 'N/A'
metrics = {}

#sys.exit()

for task in tasks:
#for task in tasks[:1]:
    print("======= task {}".format(task))
    table.append([" ", " ", " ", " ", " ", " ", " ", " "])
    acuracy = 0
    
    metrics[task] = {}
    metrics[task][na] = {}
    metrics[task][na]['MAPs'] = {'exp':0.0, 'sd':0.0}
    metrics[task][na]['MAPc'] = {'exp':0.0, 'sd':0.0}
    metrics[task][na]['accuracy'] = {'exp':0.0, 'sd':0.0}
    metrics[task][na]['kappa'] = {'exp':0.0, 'sd':0.0}
    for avg in ['micro', 'macro', 'weighted']:
        metrics[task][avg] = {}
        metrics[task][avg]['precision'] = {'exp':0.0, 'sd':0.0}
        metrics[task][avg]['recall'] = {'exp':0.0, 'sd':0.0}
        metrics[task][avg]['f1score'] = {'exp':0.0, 'sd':0.0}

    for fold in range(lstm.stratifications):
    #for fold in range(1):
        print("--- fold {}".format(fold))
        for curr in [
            ('MAPs', mapScorer.samplesScore(yt[fold][task], yp[fold][task])),
            ('MAPc', mapScorer.classesScore(yt[fold][task], yp[fold][task])),
            ('accuracy', accuracy_score(yt[fold][task], yc[fold][task])),
            ('kappa', cohen_kappa_score(ytn[fold][task], ycn[fold][task]))
        ]:
            metrics[task][na][curr[0]]['exp'] += curr[1]
            metrics[task][na][curr[0]]['sd'] += curr[1] * curr[1]
        
        for avg in ['micro', 'macro', 'weighted']:
            tempPrec, tempRec, tempF1, _ = precision_recall_fscore_support(yt[fold][task], yc[fold][task], average=avg)
            for curr in [
                ('precision', tempPrec),
                ('recall', tempRec),
                ('f1score', tempF1)
            ]:
                metrics[task][avg][curr[0]]['exp'] += curr[1]
                metrics[task][avg][curr[0]]['sd'] += curr[1] * curr[1]
            
    
    for curr in ['MAPs', 'MAPc', 'accuracy', 'kappa']:
        metrics[task][na][curr]['exp'] /= lstm.stratifications
        metrics[task][na][curr]['sd'] = math.sqrt((metrics[task][na][curr]['sd']/lstm.stratifications) - (metrics[task][na][curr]['exp'] * metrics[task][na][curr]['exp']))
    
    table.append([task, na, "{:.3f}".format(metrics[task][na]['MAPs']['exp'])+"~"+"{:.3f}".format(metrics[task][na]['MAPs']['sd']), "{:.3f}".format(metrics[task][na]['MAPc']['exp'])+"~"+"{:.3f}".format(metrics[task][na]['MAPc']['sd']), "{:.3f}".format(metrics[task][na]['accuracy']['exp'])+"~"+"{:.3f}".format(metrics[task][na]['accuracy']['sd']), "{:.3f}".format(metrics[task][na]['kappa']['exp'])+"~"+"{:.3f}".format(metrics[task][na]['kappa']['sd']), na, na, na])
    
    for avg in ['micro', 'macro', 'weighted']:
        for curr in ['precision', 'recall', 'f1score']:
            metrics[task][avg][curr]['exp'] /= lstm.stratifications
            metrics[task][avg][curr]['sd'] = math.sqrt((metrics[task][avg][curr]['sd']/lstm.stratifications) - (metrics[task][avg][curr]['exp'] * metrics[task][avg][curr]['exp']))
        
        table.append([task, avg, na, na, na, na, "{:.3f}".format(metrics[task][avg]['precision']['exp'])+"~"+"{:.3f}".format(metrics[task][avg]['precision']['sd']), "{:.3f}".format(metrics[task][avg]['recall']['exp'])+"~"+"{:.3f}".format(metrics[task][avg]['recall']['sd']), "{:.3f}".format(metrics[task][avg]['f1score']['exp'])+"~"+"{:.3f}".format(metrics[task][avg]['f1score']['sd'])])  

with open(outputPlotDir+"/table.txt", "w") as tabFile:
    tabFile.write(tabulate(table))

print("Finish")

        
