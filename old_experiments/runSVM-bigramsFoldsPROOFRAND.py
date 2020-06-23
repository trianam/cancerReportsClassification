import sys
from mySVMbigramsFoldPROOFRAND import MySVM

svm = MySVM()

svm.extractData()
svm.createModels()
svm.evaluate()

