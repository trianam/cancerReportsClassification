import numpy as np
import pickle
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

class MySVM:
    #_tasks = ['sede1', 'sede2', 'sede12', 'morfo1', 'morfo2', 'morfo12']
    _tasks = ['sede12']
    _filesFolder = "./filesSVM2-bigrams"
    _memmapFolder = "./memmapSVM-bigrams"
    _corpusFolder = "./corpusLSTM_ICDO3"
    #_corpusFolder = "./corpusLSTM_ICDO1"

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

    #_textFileTrain = _corpusFolder+"/foldsA/sede12/0/textTrain.txt"
    #_textFileTest = _corpusFolder+"/foldsA/sede12/0/textTest.txt"
    #_fileYTrain = _corpusFolder+"/foldsA/sede12/0/yTrain.txt"
    #_fileYTest = _corpusFolder+"/foldsA/sede12/0/yTest.txt"
    
    #_textFileTrain = _corpusFolder+"/folds10/sede12/0/textTrain.txt"
    #_textFileTest = _corpusFolder+"/folds10/sede12/0/textTest.txt"
    #_fileYTrain = _corpusFolder+"/folds10/sede12/0/yTrain.txt"
    #_fileYTest = _corpusFolder+"/folds10/sede12/0/yTest.txt"
    
    _textFileTrain = _corpusFolder+"/randomSplit/sede12/textTrain.txt"
    _textFileTest = _corpusFolder+"/randomSplit/sede12/textTest.txt"
    _fileYTrain = _corpusFolder+"/randomSplit/sede12/yTrain.txt"
    _fileYTest = _corpusFolder+"/randomSplit/sede12/yTest.txt"

    _textFile = _corpusFolder+"/text.txt"
    #_fileSedeClean = _corpusFolder+"/sedeClean.txt"
    #_fileMorfoClean = _corpusFolder+"/morfoClean.txt"
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

        with open(self._textFile) as fid:
            textFull = fid.readlines()

        with open(self._textFileTrain) as fid:
            textTrain = fid.readlines()

        with open(self._textFileTest) as fid:
            textTest = fid.readlines()

        with open(self._fileYTrain) as fid:
            yTrain = fid.readlines()

        with open(self._fileYTest) as fid:
            yTest = fid.readlines()

        vectorizer = TfidfVectorizer(min_df=3, max_df=0.3, strip_accents='unicode', ngram_range=(1,2))
        vectorizer.fit(textFull)
        #vectorizer.fit(textTrain)
        self._vecLen = len(vectorizer.get_feature_names())
        #X = np.memmap(self._fileMemmapX, mode='w+', shape=(len(text), self._vecLen), dtype=np.float)
        #X[:] = vectorizer.transform(text).toarray()
        self.XTrain = vectorizer.transform(textTrain)
        self.XTest = vectorizer.transform(textTest)

        del textTrain
        del textTest

        pickle.dump(self.XTrain, open(self._fileMemmapXTrain, 'wb'))
        pickle.dump(self.XTest, open(self._fileMemmapXTest, 'wb'))

        self.lb = {}

        for task in self._tasks:
            self.lb[task] = LabelBinarizer()
            self.lb[task].fit(np.concatenate((yTrain, yTest))) 
            #self.lb[task].fit(yTrain)
            

        print("Splitting data")
        
        shapes = {}
        shapes['XTrain'] = self.XTrain.shape
        shapes['XTest'] = self.XTest.shape

        self.yTrain = {}
        self.yTest = {}

        for task in self._tasks:
            print("   "+task)
            self.yTrain[task] = np.memmap(self._fileMemmapYTrain[task], mode='w+', shape=(len(yTrain), len(self.lb[task].classes_)), dtype=np.int)
            self.yTest[task] = np.memmap(self._fileMemmapYTest[task], mode='w+', shape=(len(yTest), len(self.lb[task].classes_)), dtype=np.int)
        
            self.yTrain[task][:] = self.lb[task].transform(yTrain)
            self.yTest[task][:] = self.lb[task].transform(yTest)
        
            self.yTrain[task].flush()
            self.yTest[task].flush()
            shapes['yTrain'+task] = self.yTrain[task].shape
            shapes['yTest'+task] = self.yTest[task].shape

        pickle.dump(shapes, open(self._fileShapes, "wb"))

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

        #indexes = pickle.load(open(self._fileIndexes, 'rb'))
        #self.indexTrain = indexes['train']
        #self.indexTest = indexes['test']

        self.lb = {}
        for task in self._tasks:
            self.lb[task] = pickle.load(open(self._fileLb[task], 'rb'))

        self._vecLen = self.XTrain.shape[1]
    
    def createModels(self):
        print("=========================== Categorical")

        epochs = 100
        batchSize = 65

        self.model = {}

        for task in self._tasks:
            self.model[task] = OneVsRestClassifier(LinearSVC())
            print("------------------- "+task)
            self.model[task].fit(self.XTrain, self.yTrain[task])

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


