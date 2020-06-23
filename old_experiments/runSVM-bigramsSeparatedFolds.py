import sys
from mySVMbigramsSeparatedFold import MySVM

for fold in range(10):
    print("########################################### Fold {}".format(fold))
    svm = MySVM(fold)   

    svm.extractData()
    #svm.loadData()
    
    svm.createModels()
    svm.saveModels()
    #svm.loadModels()
    
    svm.evaluate()

