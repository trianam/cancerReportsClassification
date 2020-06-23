import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Merge
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from keras.models import load_model
from keras.callbacks import EarlyStopping

class MyLSTM:
    _tasks = ['sede1', 'sede2', 'sede12', 'morfo1', 'morfo2', 'morfo12']
    #_tasks = ['sede1']
    _filesFolder = "./files9-ICDO3"
    _memmapFolder = "./memmapLSTM-ICDO3"
    _corpusFolder = "./corpusLSTM_ICDO3"

    _fileLb = {
        'sede1' : _memmapFolder+"/binarizers/lbSede1.p",
        'sede2' : _memmapFolder+"/binarizers/lbSede2.p",
        'sede12' : _memmapFolder+"/binarizers/lbSede12.p",
        'morfo1' : _memmapFolder+"/binarizers/lbMorfo1.p",
        'morfo2' : _memmapFolder+"/binarizers/lbMorfo2.p",
        'morfo12' : _memmapFolder+"/binarizers/lbMorfo12.p"
    }
    _fileEvaluation = _filesFolder+"/outputLSTM/evaluation.txt"
    _fileEvaluationFineTuning = _filesFolder+"/outputLSTM/evaluationft.txt"
    _fileDumpHistory = {
        'sede1' : _filesFolder+"/outputLSTM/historyCatSede1.p",
        'sede2' : _filesFolder+"/outputLSTM/historyCatSede2.p",
        'sede2ft' : _filesFolder+"/outputLSTM/historyCatSede2ft.p",
        'sede2ft2' : _filesFolder+"/outputLSTM/historyCatSede2ft2.p",
        'sede2ft3' : _filesFolder+"/outputLSTM/historyCatSede2ft3.p",
        'sede12' : _filesFolder+"/outputLSTM/historyCatSede12.p",
        'morfo1' : _filesFolder+"/outputLSTM/historyCatMorfo1.p",
        'morfo2' : _filesFolder+"/outputLSTM/historyCatMorfo2.p",
        'morfo12' : _filesFolder+"/outputLSTM/historyCatMorfo12.p",
    }
    _fileModel = {
        'sede1' : _filesFolder+"/modelsLSTM/modelCatSede1.h5",
        'sede2' : _filesFolder+"/modelsLSTM/modelCatSede2.h5",
        'sede2ft' : _filesFolder+"/modelsLSTM/modelCatSede2ft.h5",
        'sede2ft2' : _filesFolder+"/modelsLSTM/modelCatSede2ft2.h5",
        'sede2ft3' : _filesFolder+"/modelsLSTM/modelCatSede2ft3.h5",
        'sede12' : _filesFolder+"/modelsLSTM/modelCatSede12.h5",
        'morfo1' : _filesFolder+"/modelsLSTM/modelCatMorfo1.h5",
        'morfo2' : _filesFolder+"/modelsLSTM/modelCatMorfo2.h5",
        'morfo12' : _filesFolder+"/modelsLSTM/modelCatMorfo12.h5",
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

        X = np.memmap(self._fileMemmapX, mode='w+', shape=(len(text), self._phraseLen, self._vecLen), dtype=np.float)
        for i,l in enumerate(text):
            if i%1000==0:
                print("processed line {}/{}".format(i,len(text)))
            j = 0
            for w in l.split():
                if w in vectors:
                    X[i,j] = vectors[w]
                    j += 1
                    if j >= self._phraseLen:
                        break
        
        del text
        del vectors

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
        y = {}
        for task in self._tasks:
            self.lb[task] = LabelBinarizer()
            self.lb[task].fit(yUn[task])
            
            y[task] = np.memmap(self._fileMemmapY[task], mode='w+', shape=(len(sedeClean), len(self.lb[task].classes_)), dtype=np.int)
            y[task][:] = self.lb[task].transform(yUn[task])

            del yUn[task]

        print("Splitting data")
        self.indexTrain, self.indexTest = train_test_split(np.arange(len(X)), test_size=0.2)

        print("   X")
        self.XTrain = np.memmap(self._fileMemmapXTrain, mode='w+', shape=(len(self.indexTrain), self._phraseLen, self._vecLen), dtype=np.float)
        self.XTest = np.memmap(self._fileMemmapXTest, mode='w+', shape=(len(self.indexTest), self._phraseLen, self._vecLen), dtype=np.float)
        
        self.XTrain[:] = X[self.indexTrain]
        self.XTest[:] = X[self.indexTest]
        del X
        self.XTrain.flush()
        self.XTest.flush()
        
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

        self.XTrain = np.memmap(self._fileMemmapXTrain, mode='r', shape=shapes['XTrain'], dtype=np.float)
        self.XTest = np.memmap(self._fileMemmapXTest, mode='r', shape=shapes['XTest'], dtype=np.float)

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

        self._phraseLen = self.XTrain.shape[1]
        self._vecLen = self.XTrain.shape[2]
    
    def _createModelBin(self, outDim):
        model = Sequential()
        model.add(LSTM(100, input_dim=self._vecLen, input_length=self._phraseLen))
        model.add(Dense(outDim, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def _createModelCat(self, outDim):
        model = Sequential()
        #model.add(LSTM(400, input_dim=self._vecLen, input_length=self._phraseLen))
        model.add(LSTM(200, input_dim=self._vecLen, input_length=self._phraseLen, dropout_W=0.3, dropout_U=0.3))
        #model.add(Dense(200, activation='relu'))
        model.add(Dense(outDim, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def createModels(self):
        print("=========================== Categorical")

        epochs = 100
        batchSize = 65

        early_stopping = EarlyStopping(monitor='val_loss', patience=2)

        self.model = {}
        self.history = {}

        for task in self._tasks:
            self.model[task] = self._createModelCat(self.yTrain[task].shape[1])
            print("------------------- "+task)
            self.history[task] = self.model[task].fit(self.XTrain, self.yTrain[task], validation_split=0.2, nb_epoch=epochs, batch_size=batchSize, callbacks=[early_stopping])

    def createFineTuningModel(self):
        epochs = 100
        batchSize = 65

        early_stopping = EarlyStopping(monitor='val_loss', patience=2)

        part1Model = Sequential()
        for i,l in enumerate(self.model['sede1'].layers):
            l.name = "pretrained{}".format(i)
            l.trainable = False
            part1Model.add(l)
        #self.model['sede1'].compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        part2Model = Sequential()
        part2Model.add(LSTM(200, input_dim=self._vecLen, input_length=self._phraseLen, dropout_W=0.3, dropout_U=0.3))

        self.model['sede2ft'] = Sequential()
        self.model['sede2ft'].add(Merge([part1Model, part2Model], mode='concat'))
        self.model['sede2ft'].add(Dense(self.yTrain['sede2'].shape[1], activation='softmax'))
        
        self.model['sede2ft'].compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.history['sede2ft'] = self.model['sede2ft'].fit([self.XTrain, self.XTrain], self.yTrain['sede2'], validation_split=0.2, nb_epoch=epochs, batch_size=batchSize, callbacks=[early_stopping])

    def createFineTuning2Model(self):
        epochs = 100
        batchSize = 65

        early_stopping = EarlyStopping(monitor='val_loss', patience=2)

        part1Model = Sequential()
        part1Model.add(Dense(200, input_dim = self.yTrain['sede1'].shape[1], activation='relu'))
        
        part2Model = Sequential()
        part2Model.add(LSTM(200, input_dim=self._vecLen, input_length=self._phraseLen, dropout_W=0.3, dropout_U=0.3))

        self.model['sede2ft2'] = Sequential()
        self.model['sede2ft2'].add(Merge([part1Model, part2Model], mode='concat'))
        self.model['sede2ft2'].add(Dense(self.yTrain['sede2'].shape[1], activation='softmax'))
        
        self.model['sede2ft2'].compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.history['sede2ft2'] = self.model['sede2ft2'].fit([self.yTrain['sede1'], self.XTrain], self.yTrain['sede2'], validation_split=0.2, nb_epoch=epochs, batch_size=batchSize, callbacks=[early_stopping])
 
    def createFineTuning3Model(self):
        epochs = 100
        batchSize = 65

        early_stopping = EarlyStopping(monitor='val_loss', patience=2)

        part1Model = Sequential()
        part1Model.add(Dense(200, input_dim = self.yTrain['sede1'].shape[1], activation='relu'))
        
        part2Model = Sequential()
        part2Model.add(LSTM(200, input_dim=self._vecLen, input_length=self._phraseLen, dropout_W=0.3, dropout_U=0.3))

        self.model['sede2ft3'] = Sequential()
        self.model['sede2ft3'].add(Merge([part1Model, part2Model], mode='concat'))
        self.model['sede2ft3'].add(Dense(self.yTrain['sede12'].shape[1], activation='softmax'))
        
        self.model['sede2ft3'].compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.history['sede2ft3'] = self.model['sede2ft3'].fit([self.yTrain['sede1'], self.XTrain], self.yTrain['sede12'], validation_split=0.2, nb_epoch=epochs, batch_size=batchSize, callbacks=[early_stopping])
 
    def saveModels(self):
        print("Save models")

        for task in self._tasks:
            #pickle.dump(self.lb[task], open(self._fileLb[task], "wb"))
            pickle.dump(self.history[task].history, open(self._fileDumpHistory[task], "wb"))
            self.model[task].save(self._fileModel[task])

    def saveFineTuningModel(self):
        print("Save fine tuning model")
        pickle.dump(self.history['sede2ft'].history, open(self._fileDumpHistory['sede2ft'], "wb"))
        self.model['sede2ft'].save(self._fileModel['sede2ft'])

    def saveFineTuning2Model(self):
        print("Save fine tuning 2 model")
        pickle.dump(self.history['sede2ft2'].history, open(self._fileDumpHistory['sede2ft2'], "wb"))
        self.model['sede2ft2'].save(self._fileModel['sede2ft2'])

    def saveFineTuning3Model(self):
        print("Save fine tuning 3 model")
        pickle.dump(self.history['sede2ft3'].history, open(self._fileDumpHistory['sede2ft3'], "wb"))
        self.model['sede2ft3'].save(self._fileModel['sede2ft3'])
 
    def loadModels(self):
        print("Load models")
        
        self.history = {}
        self.model = {}
        self.lb = {}
        for task in self._tasks:
            self.model[task] = load_model(self._fileModel[task])
            #self.lb[task] = pickle.load(open(self._fileLb[task], 'rb'))

    def loadFineTuningModel(self):
        print("Load fine tuning model")
        self.model['sede2ft'] = load_model(self._fileModel['sede2ft'])
 
    def loadFineTuning2Model(self):
        print("Load fine tuning 2 model")
        self.model['sede2ft2'] = load_model(self._fileModel['sede2ft2'])
 
    def loadFineTuning3Model(self):
        print("Load fine tuning 3 model")
        self.model['sede2ft3'] = load_model(self._fileModel['sede2ft3'])
 
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

    def evaluateFineTuning(self):
        print("Evaluate fine tuning")
        pp = self.model['sede2ft'].predict_classes([self.XTest, self.XTest])
        ppb = np.zeros(self.yTest['sede2'].shape, np.int)
        for i,j in enumerate(pp):            
            ppb[i][j] = 1            

        acc = accuracy_score(self.lb['sede2'].inverse_transform(self.yTest['sede2']), self.lb['sede2'].inverse_transform(ppb))
        kap = cohen_kappa_score(self.lb['sede2'].inverse_transform(self.yTest['sede2']), self.lb['sede2'].inverse_transform(ppb))
        
        with open(self._fileEvaluationFineTuning, 'w') as fe:
            fe.write("\t|  accuracy  |   kappa\n")
            fe.write("{}\t|    {:.2f}    |   {:.2f}\n".format('sede2ft', acc, kap))

        with open(self._fileEvaluationFineTuning, 'r') as fe:
            for l in fe.readlines():
                print(l, end='')

    def evaluateFineTuning2(self):
        print("Evaluate fine tuning 2")
        pp = self.model['sede2ft2'].predict_classes([self.yTest['sede1'], self.XTest])
        ppb = np.zeros(self.yTest['sede2'].shape, np.int)
        for i,j in enumerate(pp):            
            ppb[i][j] = 1            

        acc = accuracy_score(self.lb['sede2'].inverse_transform(self.yTest['sede2']), self.lb['sede2'].inverse_transform(ppb))
        kap = cohen_kappa_score(self.lb['sede2'].inverse_transform(self.yTest['sede2']), self.lb['sede2'].inverse_transform(ppb))
        
        with open(self._fileEvaluationFineTuning, 'w') as fe:
            fe.write("\t|  accuracy  |   kappa\n")
            fe.write("{}\t|    {:.2f}    |   {:.2f}\n".format('sede2ft2', acc, kap))

        with open(self._fileEvaluationFineTuning, 'r') as fe:
            for l in fe.readlines():
                print(l, end='')

    def evaluateFineTuning3(self):
        print("Evaluate fine tuning 3")
        pp = self.model['sede2ft3'].predict_classes([self.yTest['sede1'], self.XTest])
        ppb = np.zeros(self.yTest['sede12'].shape, np.int)
        for i,j in enumerate(pp):            
            ppb[i][j] = 1            

        acc = accuracy_score(self.lb['sede12'].inverse_transform(self.yTest['sede12']), self.lb['sede12'].inverse_transform(ppb))
        kap = cohen_kappa_score(self.lb['sede12'].inverse_transform(self.yTest['sede12']), self.lb['sede12'].inverse_transform(ppb))
        
        with open(self._fileEvaluationFineTuning, 'w') as fe:
            fe.write("\t|  accuracy  |   kappa\n")
            fe.write("{}\t|    {:.2f}    |   {:.2f}\n".format('sede2ft3', acc, kap))

        with open(self._fileEvaluationFineTuning, 'r') as fe:
            for l in fe.readlines():
                print(l, end='')

    def _evaluate(self, model, y, lb):
        pp = model.predict_classes(self.XTest)
        ppb = np.zeros(y.shape, np.int)
        for i,j in enumerate(pp):            
            ppb[i][j] = 1            


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


