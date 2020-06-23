#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "" 

import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

class MySVM:
    #_tasks = ['sede1', 'sede2', 'sede12', 'morfo1', 'morfo2', 'morfo12']
    _tasks = ['sede1', 'sede12', 'morfo1', 'morfo2']
    #_tasks = ['sede1']
    _filesFolder = "./filesSVM2-bigrams-kfold"
    _memmapFolder = "./memmapSVM-bigrams-kfold"
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

        #vectorizer = TfidfVectorizer(min_df=3, max_df=0.3, strip_accents='unicode', ngram_range=(1,2))
        vectorizer = TfidfVectorizer(min_df=3, max_df=0.5, strip_accents='unicode', ngram_range=(1,2))
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

        self.lb = {}
        self.y = {}
        for task in self._tasks:
            self.lb[task] = LabelBinarizer()
            self.lb[task].fit(yUn[task])
            
            self.y[task] = np.memmap(self._fileMemmapY[task], mode='w+', shape=(len(sedeClean), len(self.lb[task].classes_)), dtype=np.int)
            self.y[task][:] = self.lb[task].transform(yUn[task])

            #del yUn[task]

        print("Splitting data")
        skf = StratifiedKFold(n_splits=self.stratifications, random_state=42)
        self.trainIndexes = {}
        self.testIndexes = {}

        for task in self._tasks:
            self.trainIndexes[task] = []
            self.testIndexes[task] = []

            for train, test in skf.split(np.zeros(len(yUn[task])), yUn[task]):
                self.trainIndexes[task].append(train)
                self.testIndexes[task].append(test)

        return


        self.indexTrain, self.indexTest = train_test_split(np.arange(X.shape[0]), test_size=0.2, random_state=42)

        print("   X")
        #self.XTrain = np.memmap(self._fileMemmapXTrain, mode='w+', shape=(len(self.indexTrain), self._vecLen), dtype=np.float)
        #self.XTest = np.memmap(self._fileMemmapXTest, mode='w+', shape=(len(self.indexTest), self._vecLen), dtype=np.float)
        
        #self.XTrain[:] = X[self.indexTrain]
        #self.XTest[:] = X[self.indexTest]
        
        self.XTrain = X[self.indexTrain]
        self.XTest = X[self.indexTest]
        
        pickle.dump(self.XTrain, open(self._fileMemmapXTrain, 'wb'))
        pickle.dump(self.XTest, open(self._fileMemmapXTest, 'wb'))

        #del X
        #self.XTrain.flush()
        #self.XTest.flush()
        
        shapes = {}
        shapes['XTrain'] = self.XTrain.shape
        shapes['XTest'] = self.XTest.shape

        self.yTrain = {}
        self.yTest = {}

        for task in self._tasks:
            print("   "+task)
            self.yTrain[task] = np.memmap(self._fileMemmapYTrain[task], mode='w+', shape=(len(self.indexTrain), len(self.lb[task].classes_)), dtype=np.int)
            self.yTest[task] = np.memmap(self._fileMemmapYTest[task], mode='w+', shape=(len(self.indexTest), len(self.lb[task].classes_)), dtype=np.int)
        
            self.yTrain[task][:] = y[task][self.indexTrain]
            self.yTest[task][:] = y[task][self.indexTest]
        
            del y[task]
            self.yTrain[task].flush()
            self.yTest[task].flush()
            shapes['yTrain'+task] = self.yTrain[task].shape
            shapes['yTest'+task] = self.yTest[task].shape

        pickle.dump(shapes, open(self._fileShapes, "wb"))

        indexes = {}
        indexes['train'] = self.indexTrain
        indexes['test'] = self.indexTest
        pickle.dump(indexes, open(self._fileIndexes, "wb"))

        for task in self._tasks:
            pickle.dump(self.lb[task], open(self._fileLb[task], "wb"))
            
    def loadData(self):
        shapes = pickle.load(open(self._fileShapes, 'rb'))

        #self.XTrain = np.memmap(self._fileMemmapXTrain, mode='r', shape=shapes['XTrain'], dtype=np.float)
        #self.XTest = np.memmap(self._fileMemmapXTest, mode='r', shape=shapes['XTest'], dtype=np.float)
        self.XTrain = pickle.load(open(self._fileMemmapXTrain, 'rb'))
        self.XTest = pickle.load(open(self._fileMemmapXTest, 'rb'))

        self.yTrain = {}
        self.yTest = {}
        for task in self._tasks:
            self.yTrain[task] = np.memmap(self._fileMemmapYTrain[task], mode='r', shape=shapes['yTrain'+task], dtype=np.int)
            self.yTest[task] = np.memmap(self._fileMemmapYTest[task], mode='r', shape=shapes['yTest'+task], dtype=np.int)

        indexes = pickle.load(open(self._fileIndexes, 'rb'))
        self.indexTrain = indexes['train']
        self.indexTest = indexes['test']

        self.lb = {}
        for task in self._tasks:
            self.lb[task] = pickle.load(open(self._fileLb[task], 'rb'))

        self._vecLen = self.XTrain.shape[1]
    
    def createModels(self, stratification):
        print("=========================== Categorical")

        epochs = 100
        batchSize = 65

        self.model = {}

        for task in self._tasks:
            XTrain = self.X[self.trainIndexes[task][stratification]]
            yTrain = self.y[task][self.trainIndexes[task][stratification]]
            self.model[task] = OneVsRestClassifier(LinearSVC())
            print("------------------- "+task)
            self.model[task].fit(XTrain, yTrain)

    def saveModels(self):
        print("save models")
        for task in self._tasks:
            with open(self._fileModel[task], 'wb') as fid:
                pickle.dump(self.model[task], fid)    

    def loadModels(self):
        print("load models")
        self.model = {}
        for task in self._tasks:
            with open(self._fileModel[task], 'rb') as fid:
                self.model[task] = pickle.load(fid)

    def evaluate(self):
        print("Evaluate")
        acc = {}
        kap = {}

        for task in self._tasks:
            acc[task], kap[task] = self._evaluate(self.model[task], self.yTest[task], self.lb[task])

        with open(self._fileEvaluation, 'w') as fe:
            fe.write("\t|  accuracy  |   kappa\n")
            for task in self._tasks:
                fe.write("{}\t|    {:.2f}    |   {:.2f}\n".format(task, acc[task], kap[task]))

        with open(self._fileEvaluation, 'r') as fe:
            for l in fe.readlines():
                print(l, end='')

    def _evaluate(self, model, y, lb):
        pp = model.decision_function(self.XTest)
        ppb = np.zeros(y.shape, np.int)
        for i,p in enumerate(pp):            
            ppb[i][np.argmax(p)] = 1            


        #media = 0
        #mini = np.inf
        #maxi = 0
        #for i in range(ppb.shape[1]):
        #    currAcc = sk.metrics.accuracy_score(yMorfo2[:,i], ppb[:,i])
        #    media += currAcc
        #    if currAcc < mini:
        #        mini = currAcc
        #    if currAcc > maxi:
        #        maxi = currAcc
        #media /= ppb.shape[1]
        #multi = sk.metrics.accuracy_score(yMorfo2, ppb)

        acc = accuracy_score(lb.inverse_transform(y), lb.inverse_transform(ppb))
        kap = cohen_kappa_score(lb.inverse_transform(y), lb.inverse_transform(ppb))

        return (acc, kap)


