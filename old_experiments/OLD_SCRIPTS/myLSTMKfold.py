import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Merge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from keras.models import load_model
from keras.callbacks import EarlyStopping

class MyLSTM:
    #_tasks = ['sede1', 'sede12', 'sede2ft3', 'morfo1', 'morfo2']
    _tasks = ['sede1', 'sede12', 'morfo1', 'morfo2']
    #_tasks = ['sede1']
    _filesFolder = "./files11"
    _memmapFolder = "./memmapLSTMKfold2"
    _corpusFolder = "./corpusLSTM_ICDO3"

    _fileLb = {
        'sede1' : _memmapFolder+"/binarizers/lbSede1.p",
        'sede12' : _memmapFolder+"/binarizers/lbSede12.p",
        'sede2ft3' : _memmapFolder+"/binarizers/lbSede2ft3.p",
        'morfo1' : _memmapFolder+"/binarizers/lbMorfo1.p",
        'morfo2' : _memmapFolder+"/binarizers/lbMorfo2.p",
    }
    _folderDumpHistories = _filesFolder+"/outputLSTM/"
    _fileDumpHistory = {
        'sede1' : "/historyCatSede1.p",
        'sede2ft3' : "/historyCatSede2ft3.p",
        'sede12' : "/historyCatSede12.p",
        'morfo1' : "/historyCatMorfo1.p",
        'morfo2' : "/historyCatMorfo2.p",
    }
    _fileModel = {}
    _textFile = _corpusFolder+"/text.txt"
    _fileSedeClean = _corpusFolder+"/sedeClean.txt"
    _fileMorfoClean = _corpusFolder+"/morfoClean.txt"
    _fileVectors = _corpusFolder+"/vectors.txt"
    
    _fileMemmapX = _memmapFolder+"/X.dat"
    _fileMemmapYUn = {
        'sede1' : "./tmp/yUnSede1.dat",
        'sede12' : "./tmp/yUnSede12.dat",
        'sede2ft3' : "./tmp/yUnSede2ft3.dat",
        'morfo1' : "./tmp/yUnMorfo1.dat",
        'morfo2' : "./tmp/yUnMorfo2.dat",
    }
    _fileMemmapY = {
        'sede1' : _memmapFolder+"/ySede1.dat",
        'sede12' : _memmapFolder+"/ySede12.dat",
        'sede2ft3' : _memmapFolder+"/ySede2ft3.dat",
        'morfo1' : _memmapFolder+"/yMorfo1.dat",
        'morfo2' : _memmapFolder+"/yMorfo2.dat",
    }

    _fileMemmapXTrain = "./tmp/XTrain.dat"
    _fileMemmapXTest = "./tmp/XTest.dat"
    _fileMemmapXTrainFT = "./tmp/XTrainFT.dat"
    _fileMemmapXTestFT = "./tmp/XTestFT.dat"
    _fileMemmapYTrain = "./tmp/YTrain.dat"
    _fileMemmapYTest = "./tmp/YTest.dat"


    _fileShapes = _memmapFolder+"/shapes.p"
    _fileIndexes = _memmapFolder+"/indexes.p"

    def __init__(self):
        self.model = {}
        self.history = {}
        self.stratifications = 10
        
        folderDumpHistories = self._filesFolder+"/outputLSTM/"
        folderModels = self._filesFolder+"/modelsLSTM/"
        
        for fold in range(self.stratifications):
            self._fileDumpHistory[fold] = {
                'sede1' : folderDumpHistories+str(fold)+"/historyCatSede1.p",
                'sede2ft3' : folderDumpHistories+str(fold)+"/historyCatSede2ft3.p",
                'sede12' : folderDumpHistories+str(fold)+"/historyCatSede12.p",
                'morfo1' : folderDumpHistories+str(fold)+"/historyCatMorfo1.p",
                'morfo2' : folderDumpHistories+str(fold)+"/historyCatMorfo2.p",
            }
            self._fileModel[fold] = {
                'sede1' : folderModels+str(fold)+"/modelCatSede1.h5",
                'sede2ft3' : folderModels+str(fold)+"/modelCatSede2ft3.h5",
                'sede12' : folderModels+str(fold)+"/modelCatSede12.h5",
                'morfo1' : folderModels+str(fold)+"/modelCatMorfo1.h5",
                'morfo2' : folderModels+str(fold)+"/modelCatMorfo2.h5",
            }

    def extractData(self):
        self._phraseLen = 100

        with open(self._textFile) as fid:
            text = fid.readlines()

        with open(self._fileSedeClean) as fid:
            sedeClean = fid.readlines()

        with open(self._fileMorfoClean) as fid:
            morfoClean = fid.readlines()

        vectors = {}
        with open(self._fileVectors) as fid:
            for line in fid.readlines():
                sline = line.split()
                currVec = []
                for i in range(1,len(sline)):
                    currVec.append(sline[i])

                vectors[sline[0]] = currVec
            self._vecLen = len(currVec)

        self.X = np.memmap(self._fileMemmapX, mode='w+', shape=(len(text), self._phraseLen, self._vecLen), dtype=np.float)
        for i,l in enumerate(text):
            if i%1000==0:
                print("processed line {}/{}".format(i,len(text)))
            j = 0
            for w in l.split():
                if w in vectors:
                    self.X[i,j] = vectors[w]
                    j += 1
                    if j >= self._phraseLen:
                        break
        
        del text
        del vectors

        shapes = {}
        shapes['X'] = self.X.shape

        yUn = {}

        yUn['sede1'] = np.memmap(self._fileMemmapYUn['sede1'], mode='w+', shape=(len(sedeClean)), dtype=np.int)
        yUn['sede12'] = np.memmap(self._fileMemmapYUn['sede12'], mode='w+', shape=(len(sedeClean)), dtype=np.int)
        for i,c in enumerate(sedeClean):
            yUn['sede1'][i], temp = c.split()
            yUn['sede12'][i] = yUn['sede1'][i] * 10 + int(temp)

        yUn['sede2ft3'] = yUn['sede12']
        yUn['morfo1'] = np.memmap(self._fileMemmapYUn['morfo1'], mode='w+', shape=(len(morfoClean)), dtype=np.int)
        yUn['morfo2'] = np.memmap(self._fileMemmapYUn['morfo2'], mode='w+', shape=(len(morfoClean)), dtype=np.int)
        for i,c in enumerate(morfoClean):
            yUn['morfo1'][i], yUn['morfo2'][i] = c.split()

        self.lb = {}
        self.y = {}
        shapes['y'] = {}
        for task in self._tasks:
            self.lb[task] = LabelBinarizer()
            self.lb[task].fit(yUn[task])
            
            self.y[task] = np.memmap(self._fileMemmapY[task], mode='w+', shape=(len(sedeClean), len(self.lb[task].classes_)), dtype=np.int)
            self.y[task][:] = self.lb[task].transform(yUn[task])
            shapes['y'][task] = self.y[task].shape

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

            del yUn[task]

        pickle.dump(shapes, open(self._fileShapes, "wb"))
        
        indexes = {}
        indexes['train'] = self.trainIndexes
        indexes['test'] = self.testIndexes
        pickle.dump(indexes, open(self._fileIndexes, "wb"))
            
        for task in self._tasks:
            pickle.dump(self.lb[task], open(self._fileLb[task], "wb"))

    def loadData(self):
        shapes = pickle.load(open(self._fileShapes, 'rb'))
        
        self.X = np.memmap(self._fileMemmapX, mode='r', shape=shapes['X'], dtype=np.float)
        self.y = {}
        for task in self._tasks:
            self.y[task] = np.memmap(self._fileMemmapY[task], mode='r', shape=shapes['y'][task], dtype=np.int)
        
        indexes = pickle.load(open(self._fileIndexes, 'rb'))
        self.trainIndexes = indexes['train']
        self.testIndexes = indexes['test']

        self.lb = {}
        for task in self._tasks:
            self.lb[task] = pickle.load(open(self._fileLb[task], 'rb'))

        self._phraseLen = self.X.shape[1]
        self._vecLen = self.X.shape[2]
    
    def _createModelCat(self, outDim):
        model = Sequential()
        #model.add(LSTM(400, input_dim=self._vecLen, input_length=self._phraseLen))
        model.add(LSTM(400, input_dim=self._vecLen, input_length=self._phraseLen, dropout_W=0.3, dropout_U=0.3))
        #model.add(LSTM(200, input_dim=self._vecLen, input_length=self._phraseLen))
        #model.add(Dense(200, activation='relu'))
        model.add(Dense(outDim, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def _getSubset(self, fileMemmap, indexes, source):
        destShape = list(source.shape)
        destShape[0] = len(indexes)
        dest = np.memmap(fileMemmap, mode='w+', shape=tuple(destShape), dtype=source.dtype)
        dest[:] = source[indexes]
        return dest

    getXTrain = lambda self, fold, task: self._getSubset(self._fileMemmapXTrain, self.trainIndexes[task][fold], self.X) 
    
    getYTrain = lambda self, fold, task: self._getSubset(self._fileMemmapYTrain, self.trainIndexes[task][fold], self.y[task])
        
    getXTrainFT = lambda self, fold, task: self._getSubset(self._fileMemmapXTrainFT, self.trainIndexes[task][fold], self.y['sede1'])

    getXTest = lambda self, fold, task: self._getSubset(self._fileMemmapXTest, self.testIndexes[task][fold], self.X)

    getYTest = lambda self, fold, task: self._getSubset(self._fileMemmapYTest, self.testIndexes[task][fold], self.y[task])
   
    getYTestName = lambda self, fold, task, name: self._getSubset(name, self.testIndexes[task][fold], self.y[task])

    getXTestFT = lambda self, fold, task: self._getSubset(self._fileMemmapXTestFT, self.testIndexes[task][fold], self.y['sede1'])
    
    def createModels(self):
        epochs = 100
        batchSize = 65

        early_stopping = EarlyStopping(monitor='val_loss', patience=4)

        for fold in range(self.stratifications):
            print("=========================== Fold: {}".format(fold))

            self.model[fold] = {}
            self.history[fold] = {}

            for task in self._tasks:                
                XTrain = self.getXTrain(fold,task)
                yTrain = self.getYTrain(fold,task)
                
                if task != 'sede2ft3':
                    self.model[fold][task] = self._createModelCat(yTrain.shape[1])
                    print("------------------- "+task)
                    self.history[fold][task] = self.model[fold][task].fit(XTrain, yTrain, validation_split=0.2, nb_epoch=epochs, batch_size=batchSize, callbacks=[early_stopping])

                else:
                    XTrainFT = self.getXTrainFT(fold,task)
                    
                    part1Model = Sequential()
                    part1Model.add(Dense(200, input_dim = XTrainFT.shape[1], activation='relu'))

                    part2Model = Sequential()
                    part2Model.add(LSTM(200, input_dim=self._vecLen, input_length=self._phraseLen, dropout_W=0.3, dropout_U=0.3))
                    #part2Model.add(LSTM(200, input_dim=self._vecLen, input_length=self._phraseLen))

                    self.model[fold][task] = Sequential()
                    self.model[fold][task].add(Merge([part1Model, part2Model], mode='concat'))
                    self.model[fold][task].add(Dense(yTrain.shape[1], activation='softmax'))

                    self.model[fold][task].compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                    self.history[fold][task] = self.model[fold][task].fit([XTrainFT, XTrain], yTrain, validation_split=0.2, nb_epoch=epochs, batch_size=batchSize, callbacks=[early_stopping])


    def saveModels(self):
        print("Save models")

        for fold in range(self.stratifications):
            for task in self._tasks:
                #pickle.dump(self.lb[task], open(self._fileLb[task], "wb"))
                pickle.dump(self.history[fold][task].history, open(self._fileDumpHistory[fold][task], "wb"))
                self.model[fold][task].save(self._fileModel[fold][task])

 
    def loadModels(self):
        print("Load models")
        
        self.history = {}
        self.model = {}
        self.lb = {}
        for fold in range(self.stratifications):
            self.model[fold] = {}
            for task in self._tasks:
                self.model[fold][task] = load_model(self._fileModel[fold][task])
                #self.lb[task] = pickle.load(open(self._fileLb[task], 'rb'))
 

