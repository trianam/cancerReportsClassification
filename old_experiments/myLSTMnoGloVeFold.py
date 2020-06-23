import numpy as np
import pickle
import os
import math
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score, precision_recall_fscore_support, f1_score, accuracy_score, cohen_kappa_score
from tabulate import tabulate
from MAPScorer import MAPScorer
from scipy.interpolate import interp1d
from scipy import interp
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Masking
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Merge
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.pooling import GlobalAveragePooling1D
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras import initializers
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from keras.models import load_model
from keras.callbacks import EarlyStopping


class MyLSTM:
    _filesFolder = "./filesFolds-LSTMnoGloVe"
    _memmapFolder = "./memmapFolds-LSTMnoGloVe"
    
    def __init__(self, fold=0, tasks=['sede1', 'sede12', 'morfo1', 'morfo2'], corpusFolder="corpusLSTM_ICDO3", foldsFolder="folds10", fileWordsIndex="wordsIndex.p", phraseLen = 200, lstmCells=150, embeddingSize = 30, learningRate=0.001, learningRateDecay=0., dropout=0.5, patience=2, batchSize=65, epochs=100):
        self._tasks = tasks
        self._corpusFolder = "./"+corpusFolder
        self._fileWordsIndex = self._corpusFolder+"/"+fileWordsIndex
        self._lstmCells = lstmCells
        self._learningRate = learningRate
        self._learningRateDecay = learningRateDecay
        self._dropout = dropout
        self._patience = patience
        self._batchSize = batchSize
        self._epochs = epochs
        self._phraseLen = phraseLen
        self._embeddingSize = embeddingSize

        self.model = {}
        self.history = {}
        for task in self._tasks:
            self.model[task]=[]
            self.history[task]=[]

        self.XTrain = {}
        self.XTest = {}
        self.lb = {}
        self.yTrain = {}
        self.yTest = {}
        
        self._early_stopping = EarlyStopping(monitor='val_loss', patience=self._patience)

        shapesFolder = self._memmapFolder+"/"+str(fold)+"/shapes"
        if not os.path.exists(shapesFolder):
            os.makedirs(shapesFolder)
        self._fileShapes = shapesFolder+"/shapes.p"
       
        lbFolder = self._memmapFolder+"/"+str(fold)+"/binarizers/"
        if not os.path.exists(lbFolder):
            os.makedirs(lbFolder)
            
        self._fileLb = {}
        for task in self._tasks:
            self._fileLb[task] = lbFolder+"lb"+task.capitalize()+".p"
        
        dataFolder = self._memmapFolder+"/"+str(fold)+"/data/"
        if not os.path.exists(dataFolder):
            os.makedirs(dataFolder)
    
        self._fileMemmapXTrain = {}
        self._fileMemmapXTest = {}
        self._fileMemmapYTrain = {}
        self._fileMemmapYTest = {}
        for task in self._tasks:
            self._fileMemmapXTrain[task] = dataFolder+"XTrain"+task.capitalize()+".dat"
            self._fileMemmapXTest[task] = dataFolder+"XTest"+task.capitalize()+".dat"
            self._fileMemmapYTrain[task] = dataFolder+"yTrain"+task.capitalize()+".dat"
            self._fileMemmapYTest[task] = dataFolder+"yTest"+task.capitalize()+".dat"

        evaluationFolder = self._filesFolder+"/output/"+str(fold)
        if not os.path.exists(evaluationFolder):
            os.makedirs(evaluationFolder)
        self._fileEvaluationBase = evaluationFolder+"/evaluation"
        self._filePredictionBase = evaluationFolder+"/prediction"
        self._fileTableBase = evaluationFolder+"/table"

        historyFolder = self._filesFolder+"/history/"+str(fold)
        if not os.path.exists(historyFolder):
            os.makedirs(historyFolder)
        self._fileDumpHistoryBase = historyFolder+"/historyCat"
        
        modelsFolder = self._filesFolder+"/models/"+str(fold)
        if not os.path.exists(modelsFolder):
            os.makedirs(modelsFolder)
        self._fileModelBase = modelsFolder+"/modelCat"

        self._textFileTrain = {}
        self._textFileTest = {}
        self._fileYTrain = {}
        self._fileYTest = {}
        
        for task in self._tasks:
            corpusFoldFolder = self._corpusFolder+"/"+foldsFolder+"/"+task+"/"+str(fold)
            self._textFileTrain[task] = corpusFoldFolder+"/textTrain.txt"
            self._textFileTest[task] = corpusFoldFolder+"/textTest.txt"
            self._fileYTrain[task] = corpusFoldFolder+"/yTrain.txt"
            self._fileYTest[task] = corpusFoldFolder+"/yTest.txt"

    def _calculateFileModel(self, task, index):
        if index == -1:
            index = 0
            while os.path.isfile(self._calculateFileModel(task, index)):
                index += 1
            index -= 1

        return self._fileModelBase+task.capitalize()+"-"+str(index)+".h5"
        
    def _calculateFileDumpHistory(self, task, index):
        if index == -1:
            index = 0
            while os.path.isfile(self._calculateFileDumpHistory(task, index)):
                index += 1
            index -= 1

        return self._fileDumpHistoryBase+task.capitalize()+"-"+str(index)+".p"

    def _calculateFileEvaluation(self, task, index):
        if index == -1:
            index = 0
            while os.path.isfile(self._calculateFileEvaluation(task, index)):
                index += 1
            index -= 1

        return self._fileEvaluationBase+task.capitalize()+"-"+str(index)+".p"

    def _calculateFilePrediction(self, task, index):
        if index == -1:
            index = 0
            while os.path.isfile(self._calculateFilePrediction(task, index)):
                index += 1
            index -= 1

        #return self._filePredictionBase+task.capitalize()+"-"+str(index)+".p"
        return self._filePredictionBase+task.capitalize()+".p"

    def _calculateFileTable(self, task, index):
        if index == -1:
            index = 0
            while os.path.isfile(self._calculateFileTable(task, index)):
                index += 1
            index -= 1

        return self._fileTableBase+task.capitalize()+"-"+str(index)+".txt"

    def _processText(self, text, fileMemmap, wordsIndex):
        X = np.memmap(fileMemmap, mode='w+', shape=(len(text), self._phraseLen), dtype=np.float)

        textLen = len(text)
        for iDoc,l in enumerate(text):
            if iDoc%1000==0:
                print("         processed line {}/{}           ".format(iDoc,textLen), end='\r')

            words = l.split()

            iWordValid = 0
            for iWord in range(len(words)):
                try:
                    X[iDoc,iWordValid] = wordsIndex[words[iWord]]
                    iWordValid += 1
                except IndexError:
                    break
                except KeyError:
                    pass

        print("         processed line {}/{}           ".format(iDoc+1,textLen))
        X.flush()
        return X
                    
    def extractData(self):
        print("Extracting data")

        try:
            shapes = pickle.load(open(self._fileShapes, 'rb'))
        except FileNotFoundError:
            shapes = {}
            shapes['vecLen'] = {}
            
        shapes['phraseLen'] = self._phraseLen
        
        for task in self._tasks:
            print("   "+task)

            with open(self._textFileTrain[task]) as fid:
                textTrain = fid.readlines()

            with open(self._textFileTest[task]) as fid:
                textTest = fid.readlines()

            with open(self._fileYTrain[task]) as fid:
                yTrain = fid.readlines()

            with open(self._fileYTest[task]) as fid:
                yTest = fid.readlines()

            wordsIndex = pickle.load(open(self._fileWordsIndex, 'rb'))
 
            print("      train")
            self.XTrain[task] = self._processText(textTrain, self._fileMemmapXTrain[task], wordsIndex)
            print("      test")            
            self.XTest[task] = self._processText(textTest, self._fileMemmapXTest[task], wordsIndex)

            shapes['vecLen'][task] = len(wordsIndex)+1 #0 reserved for empty

            del textTrain
            del textTest

            self.lb[task] = LabelBinarizer()
            self.lb[task].fit(np.concatenate((yTrain, yTest)))
            #self.lb[task].fit(yTrain)

            shapes[task] = {}
            shapes[task]['XTrain'] = self.XTrain[task].shape
            shapes[task]['XTest'] = self.XTest[task].shape

            self.yTrain[task] = np.memmap(self._fileMemmapYTrain[task], mode='w+', shape=(len(yTrain), len(self.lb[task].classes_)), dtype=np.int)
            self.yTest[task] = np.memmap(self._fileMemmapYTest[task], mode='w+', shape=(len(yTest), len(self.lb[task].classes_)), dtype=np.int)
        
            self.yTrain[task][:] = self.lb[task].transform(yTrain)
            self.yTest[task][:] = self.lb[task].transform(yTest)
        
            self.yTrain[task].flush()
            self.yTest[task].flush()
            shapes[task]['yTrain'] = self.yTrain[task].shape
            shapes[task]['yTest'] = self.yTest[task].shape

            pickle.dump(self.lb[task], open(self._fileLb[task], "wb"))
 
           
        self._vecLen = shapes['vecLen']
        pickle.dump(shapes, open(self._fileShapes, "wb"))

    def loadData(self):
        shapes = pickle.load(open(self._fileShapes, 'rb'))
        self._phraseLen = shapes['phraseLen']
        self._vecLen = shapes['vecLen']

        for task in self._tasks:
            self.XTrain[task] = np.memmap(self._fileMemmapXTrain[task], mode='r', shape=shapes[task]['XTrain'], dtype=np.float)
            self.XTest[task] = np.memmap(self._fileMemmapXTest[task], mode='r', shape=shapes[task]['XTest'], dtype=np.float)

            self.yTrain[task] = np.memmap(self._fileMemmapYTrain[task], mode='r', shape=shapes[task]['yTrain'], dtype=np.int)
            self.yTest[task] = np.memmap(self._fileMemmapYTest[task], mode='r', shape=shapes[task]['yTest'], dtype=np.int)

            self.lb[task] = pickle.load(open(self._fileLb[task], 'rb'))

    def _createModel(self, outDim, vecLen):
        model = Sequential()
        #model.add(Masking(mask_value=0., input_shape=(self._phraseLen, vecLen)))
        model.add(Embedding(vecLen, self._embeddingSize, input_length=self._phraseLen))
        #model.add(Embedding(vecLen, self._embeddingSize))
        model.add(Bidirectional(LSTM(self._lstmCells, dropout=self._dropout, recurrent_dropout=self._dropout, return_sequences=True)))
        model.add(Bidirectional(LSTM(self._lstmCells, dropout=self._dropout, recurrent_dropout=self._dropout, return_sequences=True)))
        #model.add(Bidirectional(LSTM(outDim, dropout=self._dropout, recurrent_dropout=self._dropout, return_sequences=True), input_shape=(self._phraseLen, self._vecLen)))
        #model.add(Bidirectional(LSTM(outDim, dropout=self._dropout, recurrent_dropout=self._dropout, return_sequences=True)))
        model.add(GlobalAveragePooling1D())
        #model.add(Dense(outDim, activation='relu'))
        model.add(Dense(outDim, activation='relu'))
        model.add(Dense(outDim, activation='softmax'))

        #opt = RMSprop(lr=self._learningRate, decay=self._learningRateDecay)
        opt = Adam(lr=self._learningRate, decay=self._learningRateDecay)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

            
    def createModels(self):
        print("Creating models")

        for task in self._tasks:
            print("   "+task)
            self.model[task] = [self._createModel(self.yTrain[task].shape[1], self._vecLen[task])]
            self.history[task] = [self.model[task][0].fit(self.XTrain[task], self.yTrain[task], validation_split=0.2, epochs=self._epochs, batch_size=self._batchSize, callbacks=[self._early_stopping]).history]

    def continueTrainModels(self, fromModel=-1, newLearningRate=None, useEarlyStopping=False, epochs=None):
        print("Continue train models")
        for task in self._tasks:
            print("   "+task)

            #save and reload to make a copy
            fileModel = self._calculateFileModel(task, fromModel)
            self.model[task][fromModel].save(fileModel)
            self.model[task].append(load_model(fileModel))

            if newLearningRate != None:
                opt = RMSprop(lr = newLearningRate)
                self.model[task][-1].compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

            myCallbacks = []
            if useEarlyStopping:
                myCallbacks.append(self._early_stopping)

            if epochs != None:
                myEpochs = epochs
            else:
                myEpochs = self._epochs

            self.history[task].append(self.model[task][-1].fit(self.XTrain[task], self.yTrain[task], validation_split=0.2, epochs=myEpochs, batch_size=self._batchSize, callbacks=myCallbacks).history)

    def saveModels(self):
        print("Saving models")
        for task in self._tasks:
            for currIndex, currHistory in enumerate(self.history[task]):
                pickle.dump(currHistory, open(self._calculateFileDumpHistory(task, currIndex), "wb"))
            for currIndex, currModel in enumerate(self.model[task]):
                currModel.save(self._calculateFileModel(task, currIndex))
            
    def loadModels(self):
        print("loading models")
        for task in self._tasks:
            currIndex = 0
            while True:
                currHistoryFile = self._calculateFileDumpHistory(task, currIndex)
                currModelFile = self._calculateFileModel(task, currIndex)
                if not os.path.isfile(currHistoryFile) and not os.path.isfile(currModelFile):
                    break

                self.model[task].append(load_model(currModelFile))
                self.history[task].append(pickle.load(open(currHistoryFile, 'rb')))
                currIndex += 1

    def loadSpecificModels(self, nums):
        print("loading models")
        for task in self._tasks:
            currHistoryFile = self._calculateFileDumpHistory(task, nums[task])
            currModelFile = self._calculateFileModel(task, nums[task])

            self.model[task].append(load_model(currModelFile))
            self.history[task].append(pickle.load(open(currHistoryFile, 'rb')))

    def evaluate(self):
        print("Evaluating")
        na = ' '
        
        for task in self._tasks:
            print(task)
            for modelIndex, currModel in enumerate(self.model[task]):
                if not os.path.isfile(self._calculateFileEvaluation(task, modelIndex)) or not os.path.isfile(self._calculateFilePrediction(task, modelIndex)) or not os.path.isfile(self._calculateFileTable(task, modelIndex)):
                    table = [["task", "average", "MAPs", "MAPc", "accur.", "kappa", "prec.", "recall", "f1score"]]
                    table.append([" ", " ", " ", " ", " ", " ", " ", " "])
                    prediction = {}
                    yp = currModel.predict_proba(self.XTest[task])
                    yt = self.yTest[task]
                    prediction['yp'] = yp
                    prediction['yt'] = yt

                    ytn = self.lb[task].inverse_transform(yt)
                    yc = np.zeros(yt.shape, np.int)
                    for i,p in enumerate(yp):
                        yc[i][np.argmax(p)] = 1
                    ycn = self.lb[task].inverse_transform(yc)

                    metrics = {}
                    metrics['MAPs'] = MAPScorer().samplesScore(yt, yp)
                    metrics['MAPc'] = MAPScorer().classesScore(yt, yp)
                    metrics['accuracy'] = accuracy_score(yt, yc)
                    metrics['kappa'] = cohen_kappa_score(ytn, ycn)

                    metrics['precision'] = {}
                    metrics['recall'] = {}
                    metrics['f1score'] = {}
                    
                    table.append([task, na, "{:.3f}".format(metrics['MAPs']), "{:.3f}".format(metrics['MAPc']), "{:.3f}".format(metrics['accuracy']), "{:.3f}".format(metrics['kappa']), na, na, na])
                    for avg in ['micro', 'macro', 'weighted']:
                        metrics['precision'][avg], metrics['recall'][avg], metrics['f1score'][avg], _ = precision_recall_fscore_support(yt, yc, average=avg)
                        table.append([task, avg, na, na, na, na, "{:.3f}".format(metrics['precision'][avg]), "{:.3f}".format(metrics['recall'][avg]), "{:.3f}".format(metrics['f1score'][avg])])

                    metrics['pr-curve'] = {}
                    metrics['pr-curve']['x'], metrics['pr-curve']['y'], metrics['pr-curve']['auc'] = self._calculateMicroMacroCurve(lambda y,s: (lambda t: (t[1],t[0]))(precision_recall_curve(y,s)), yt, yp)
                    
                    metrics['roc-curve'] = {}
                    metrics['roc-curve']['x'], metrics['roc-curve']['y'], metrics['roc-curve']['auc'] = self._calculateMicroMacroCurve(lambda y,s: (lambda t: (t[0],t[1]))(roc_curve(y,s)), yt, yp)

                    pickle.dump(metrics, open(self._calculateFileEvaluation(task, modelIndex), "wb"))
                    pickle.dump(prediction, open(self._calculateFilePrediction(task, modelIndex), "wb"))
                    tableString = tabulate(table)
                    print(tableString)
                    with open(self._calculateFileTable(task, modelIndex), "w") as fid:
                        fid.write(tableString+"\n")

    def _calculateMicroMacroCurve(self, curveFunction, yt, yp):
        n_classes = yt.shape[1]
        abscissa = dict()
        ordinate = dict()
        area = dict()
        for i in range(n_classes):
            abscissa[i], ordinate[i] = curveFunction(yt[:, i], yp[:, i])
            area[i] = auc(abscissa[i], ordinate[i])
        abscissa["micro"], ordinate["micro"] = curveFunction(yt.ravel(), yp.ravel())
        area["micro"] = auc(abscissa["micro"], ordinate["micro"])
        # aggregate all
        all_rec = list(filter(lambda x: not math.isnan(x), np.unique(np.concatenate([abscissa[i] for i in range(n_classes)]))))

        # interpolate all prec/rec curves at this points
        mean_ordinate = np.zeros_like(all_rec)
        representedClasses = 0
        unrepresentedClasses = 0
        for i in range(n_classes):
            interp = interp1d(abscissa[i], ordinate[i])
            curr_ordinate = interp(all_rec)
            if not np.any([math.isnan(x) for x in abscissa[i]]) and not np.any([math.isnan(x) for x in ordinate[i]]):
                mean_ordinate += curr_ordinate
                representedClasses += 1
            else:
                unrepresentedClasses += 1

        # average it and compute AUC
        mean_ordinate /= representedClasses

        abscissa["macro"] = all_rec
        ordinate["macro"] = mean_ordinate
        area["macro"] = auc(abscissa["macro"], ordinate["macro"])

        return (abscissa, ordinate, area)

    def clearDataBinarizers(self):
        for task in self._tasks:
            os.remove(self._fileLb[task])
            os.remove(self._fileMemmapXTrain[task])
            os.remove(self._fileMemmapXTest[task])
            os.remove(self._fileMemmapYTrain[task])
            os.remove(self._fileMemmapYTest[task])


