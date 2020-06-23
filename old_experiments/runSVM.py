import sys
from mySVM import MySVM

svm = MySVM()

if False:
    svm.loadData()
    svm.loadModels()
    svm.evaluate()
    svm.evaluateFineTuning3()

else:
    svm.extractData()
    #svm.loadData()
    svm.createModels()
    svm.saveModels()
    #svm.evaluate()

