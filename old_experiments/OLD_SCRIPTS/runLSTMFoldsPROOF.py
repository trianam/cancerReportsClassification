import sys
from myLSTMFoldPROOF import MyLSTM

for fold in [0]:
    print("########################################### Fold {}".format(fold))
    #lstm = MyLSTM(fold, tasks=['sede12'], foldsFolder="randomSplit") #0.74 acc
    lstm = MyLSTM(fold, tasks=['sede12'], foldsFolder="folds10PROOF") #0.73 accF0 0.71 accF5
    #lstm = MyLSTM(fold, tasks=['sede12'], foldsFolder="foldsAPROOF") #0.73 accF0


    lstm.extractData()
    #lstm.loadData()
    
    lstm.createModels()
    lstm.saveModels()
    #lstm.loadModels()
    
    lstm.evaluate()

