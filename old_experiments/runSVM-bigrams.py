import sys
from mySVMbigrams import MySVM

svm = MySVM()

svm.extractData()
#svm.loadData()
svm.createModels()
svm.saveModels()
#svm.loadModels()
#svm.evaluate()

