import sys
from mySVMbigramsFoldPROOF import MySVM

for fold in [0]:
    print("########################################### Fold {}".format(fold))
    #svm = MySVM(fold, tasks=['sede12'], foldsFolder="randomSplit") #0.61 acc
    #svm = MySVM(fold, tasks=['sede12'], foldsFolder="folds10PROOF") #0.70 accF0; 0.68 accF5; 0.69 accF9
    svm = MySVM(fold, tasks=['sede12'], foldsFolder="foldsAPROOF") #0.57 acc F0; 0.68 accF5; 0.69 acc F9

    svm.extractData()
    #svm.loadData()
    
    svm.createModels()
    svm.saveModels()
    #svm.loadModels()
    
    svm.evaluate()

