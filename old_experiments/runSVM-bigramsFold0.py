import sys
from mySVMbigramsFold0 import MySVM

svm = MySVM()

svm.extractData()
#svm.loadData()
svm.createModels()
svm.saveModels()
#svm.loadModels()
#svm.evaluate()

