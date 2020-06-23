import numpy as np
import importlib

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

scriptNames = {
    "SVM" : "mySVMFold",
    "SVMb" : "mySVMbigramsFold",
    "LSTMng" : "myLSTMnoGloVeFold",
    "LSTMcv" : "myLSTMconvolutionalFold",
    "LSTMb" : "myLSTMbidirectionalFold5",
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

for model in models:
    print("====================== Model {}".format(model))
    script = importlib.import_module(scriptNames[model])
    
    filesFolder = "./"+fileFolderNames[model]+"/output"

    #for t in tasks:
    for fold in range(10):
        print("---------- Fold {}".format(str(fold)))
        filesFolderFold = filesFolder+"/"+str(fold)
        if model in svmModels:
            myModel = script.MySVM(fold)
            myModel.extractData()
            myModel.loadModels()
        else:
            myModel = script.MyLSTM(fold)
            myModel.extractData()
            myModel.loadSpecificModels(goodModelNumbers[model])

        myModel.evaluate()

