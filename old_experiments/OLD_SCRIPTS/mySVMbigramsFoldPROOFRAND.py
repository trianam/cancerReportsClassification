import sys
import numpy as np
import pickle
import os
import random
import math
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score, precision_recall_fscore_support, f1_score, accuracy_score, cohen_kappa_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from tabulate import tabulate
from MAPScorer import MAPScorer
from scipy.interpolate import interp1d
from scipy import interp  
from sklearn.model_selection import StratifiedKFold

class MySVM:
    #_tasks = ['sede1', 'sede2', 'sede12', 'morfo1', 'morfo2', 'morfo12']
    _tasks = ['sede1', 'sede12', 'morfo1', 'morfo2']
    #_tasks = ['sede1']
    _filesFolder = "./filesFolds-SVMbigramsPROOFRAND"
    _memmapFolder = "./memmapFolds-SVMbigramsPROOFRAND"
    _corpusFolder = "./corpusLSTM_ICDO3"

    _fileLb = {
        'sede1' : _memmapFolder+"/binarizers/lbSede1.p",
        'sede2' : _memmapFolder+"/binarizers/lbSede2.p",
        'sede12' : _memmapFolder+"/binarizers/lbSede12.p",
        'morfo1' : _memmapFolder+"/binarizers/lbMorfo1.p",
        'morfo2' : _memmapFolder+"/binarizers/lbMorfo2.p",
        'morfo12' : _memmapFolder+"/binarizers/lbMorfo12.p"
    }
    _fileEvaluation = _filesFolder+"/outputSVM/evaluation.txt"
    _fileModel = {
        'sede1' : _filesFolder+"/modelsSVM/modelCatSede1.h5",
        'sede2' : _filesFolder+"/modelsSVM/modelCatSede2.h5",
        'sede12' : _filesFolder+"/modelsSVM/modelCatSede12.h5",
        'morfo1' : _filesFolder+"/modelsSVM/modelCatMorfo1.h5",
        'morfo2' : _filesFolder+"/modelsSVM/modelCatMorfo2.h5",
        'morfo12' : _filesFolder+"/modelsSVM/modelCatMorfo12.h5",
    }

    _textFile = _corpusFolder+"/text.txt"
    _fileSedeClean = _corpusFolder+"/sedeClean.txt"
    _fileMorfoClean = _corpusFolder+"/morfoClean.txt"
    _fileVectors = _corpusFolder+"/vectors.txt"
    
    _fileMemmapX = "./tmp/X.dat"
    _fileMemmapYUn = {
        'sede1' : "./tmp/yUnSede1.dat",
        'sede2' : "./tmp/yUnSede2.dat",
        'sede12' : "./tmp/yUnSede12.dat",
        'morfo1' : "./tmp/yUnMorfo1.dat",
        'morfo2' : "./tmp/yUnMorfo2.dat",
        'morfo12' : "./tmp/yUnMorfo12.dat"
    }
    _fileMemmapY = {
        'sede1' : "./tmp/ySede1.dat",
        'sede2' : "./tmp/ySede2.dat",
        'sede12' : "./tmp/ySede12.dat",
        'morfo1' : "./tmp/yMorfo1.dat",
        'morfo2' : "./tmp/yMorfo2.dat",
        'morfo12' : "./tmp/yMorfo12.dat"
    }

    _fileShapes = _memmapFolder+"/shapes.p"
    _fileIndexes = _memmapFolder+"/indexes.p"
    
    _fileMemmapXTrain = _memmapFolder+"/XTrain.dat"
    _fileMemmapYTrain = {
        'sede1' : _memmapFolder+"/ySede1Train.dat",
        'sede2' : _memmapFolder+"/ySede2Train.dat",
        'sede12' : _memmapFolder+"/ySede12Train.dat",
        'morfo1' : _memmapFolder+"/yMorfo1Train.dat",
        'morfo2' : _memmapFolder+"/yMorfo2Train.dat",
        'morfo12' : _memmapFolder+"/yMorfo12Train.dat"
    }
         
    _fileMemmapXTest = _memmapFolder+"/XTest.dat"
    _fileMemmapYTest = {
        'sede1' : _memmapFolder+"/ySede1Test.dat",
        'sede2' : _memmapFolder+"/ySede2Test.dat",
        'sede12' : _memmapFolder+"/ySede12Test.dat",
        'morfo1' : _memmapFolder+"/yMorfo1Test.dat",
        'morfo2' : _memmapFolder+"/yMorfo2Test.dat",
        'morfo12' : _memmapFolder+"/yMorfo12Test.dat"
    }

    
    def extractData(self):
        self._phraseLen = 100
        self.stratifications = 10

        with open(self._textFile) as fid:
            text = fid.readlines()

        with open(self._fileSedeClean) as fid:
            sedeClean = fid.readlines()

        with open(self._fileMorfoClean) as fid:
            morfoClean = fid.readlines()

        vectorizer = TfidfVectorizer(min_df=3, max_df=0.3, strip_accents='unicode', ngram_range=(1,2))
        #vectorizer = TfidfVectorizer(min_df=3, max_df=0.5, strip_accents='unicode', ngram_range=(1,2))
        vectorizer.fit(text)
        self._vecLen = len(vectorizer.get_feature_names())
        #X = np.memmap(self._fileMemmapX, mode='w+', shape=(len(text), self._vecLen), dtype=np.float)
        #X[:] = vectorizer.transform(text).toarray()
        self.X = vectorizer.transform(text)

        del text

        yUn = {}

        yUn['sede1'] = np.memmap(self._fileMemmapYUn['sede1'], mode='w+', shape=(len(sedeClean)), dtype=np.int)
        yUn['sede2'] = np.memmap(self._fileMemmapYUn['sede2'], mode='w+', shape=(len(sedeClean)), dtype=np.int)
        yUn['sede12'] = np.memmap(self._fileMemmapYUn['sede12'], mode='w+', shape=(len(sedeClean)), dtype=np.int)
        for i,c in enumerate(sedeClean):
            yUn['sede1'][i], yUn['sede2'][i] = c.split()
            yUn['sede12'][i] = yUn['sede1'][i] * 10 + yUn['sede2'][i]

        yUn['morfo1'] = np.memmap(self._fileMemmapYUn['morfo1'], mode='w+', shape=(len(morfoClean)), dtype=np.int)
        yUn['morfo2'] = np.memmap(self._fileMemmapYUn['morfo2'], mode='w+', shape=(len(morfoClean)), dtype=np.int)
        yUn['morfo12'] = np.memmap(self._fileMemmapYUn['morfo12'], mode='w+', shape=(len(morfoClean)), dtype=np.int)
        for i,c in enumerate(morfoClean):
            yUn['morfo1'][i], yUn['morfo2'][i] = c.split()
            yUn['morfo12'][i] = yUn['morfo1'][i] * 10 + yUn['morfo2'][i]

        self.lb = LabelBinarizer()
        self.lb.fit(yUn['sede12'])
            
        self.y = np.memmap(self._fileMemmapY['sede12'], mode='w+', shape=(len(sedeClean), len(self.lb.classes_)), dtype=np.int)
        self.y[:] = self.lb.transform(yUn['sede12'])

        #del yUn[task]

        print("Splitting data")
        skf = StratifiedKFold(n_splits=self.stratifications)

        self.trainIndexes = []
        self.testIndexes = []

        for train, test in skf.split(np.zeros(len(yUn['sede12'])), yUn['sede12']):
            self.trainIndexes.append(train)
            self.testIndexes.append(test)

        #self.fold = random.randint(0,9)
        self.fold = 1
        
        self.XTrain = self.X[self.trainIndexes[self.fold]]
        self.XTest = self.X[self.testIndexes[self.fold]]

        self.yTrain = np.memmap(self._fileMemmapYTrain['sede12'], mode='w+', shape=(len(self.trainIndexes[self.fold]), len(self.lb.classes_)), dtype=np.int)
        self.yTest = np.memmap(self._fileMemmapYTest['sede12'], mode='w+', shape=(len(self.testIndexes[self.fold]), len(self.lb.classes_)), dtype=np.int)

        self.yTrain[:] = self.y[self.trainIndexes[self.fold]]
        self.yTest[:] = self.y[self.testIndexes[self.fold]]

        self.yTrain.flush()
        self.yTest.flush()

    def createModels(self):
        print("Creating models")

        self.model = OneVsRestClassifier(LinearSVC())
        self.model.fit(self.XTrain, self.yTrain)

    def evaluate(self):
        print("Evaluating Test")
        self._evaluate(self.XTest, self.yTest)
        #print("Evaluating Train")
        #self._evaluate(self.XTrain, self.yTrain)
        
    def _evaluate(self, X, y):
        
        metrics = {}

        table = [["task", "average", "MAPs", "MAPc", "accur.", "kappa", "prec.", "recall", "f1score"]]
        na = ' '
        
        table.append([" ", " ", " ", " ", " ", " ", " ", " "])
        yp = self.model.decision_function(X)
        yt = y
        ytn = self.lb.inverse_transform(yt)
        yc = np.zeros(yt.shape, np.int)
        for i,p in enumerate(yp):
            yc[i][np.argmax(p)] = 1
        ycn = self.lb.inverse_transform(yc)

        metrics = {}
        metrics['MAPs'] = MAPScorer().samplesScore(yt, yp)
        metrics['MAPc'] = MAPScorer().classesScore(yt, yp)
        metrics['accuracy'] = accuracy_score(yt, yc)
        metrics['kappa'] = cohen_kappa_score(ytn, ycn)

        metrics['precision'] = {}
        metrics['recall'] = {}
        metrics['f1score'] = {}

        table.append(['sede12', na, "{:.3f}".format(metrics['MAPs']), "{:.3f}".format(metrics['MAPc']), "{:.3f}".format(metrics['accuracy']), "{:.3f}".format(metrics['kappa']), na, na, na])
        for avg in ['micro', 'macro', 'weighted']:
            metrics['precision'][avg], metrics['recall'][avg], metrics['f1score'][avg], _ = precision_recall_fscore_support(yt, yc, average=avg)
            table.append(['sede12', avg, na, na, na, na, "{:.3f}".format(metrics['precision'][avg]), "{:.3f}".format(metrics['recall'][avg]), "{:.3f}".format(metrics['f1score'][avg])])

        #metrics['pr-curve'] = {}
        #metrics['pr-curve']['x'], metrics['pr-curve']['y'], metrics['pr-curve']['auc'] = self._calculateMicroMacroCurve(lambda y,s: (lambda t: (t[1],t[0]))(precision_recall_curve(y,s)), yt, yp)

        #metrics['roc-curve'] = {}
        #metrics['roc-curve']['x'], metrics['roc-curve']['y'], metrics['roc-curve']['auc'] = self._calculateMicroMacroCurve(lambda y,s: (lambda t: (t[0],t[1]))(roc_curve(y,s)), yt, yp)


        print(tabulate(table))


        

