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

from keras.layers import Layer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Masking
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.layers import TimeDistributed
from keras.layers import Reshape
from keras.layers import Bidirectional
from keras.layers import Dropout
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras import initializers
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from keras.models import load_model
from keras.callbacks import EarlyStopping


class NonMasking(Layer):   
    def __init__(self, **kwargs):   
        self.supports_masking = True  
        super(NonMasking, self).__init__(**kwargs)   
                                                  
    def build(self, input_shape):   
        input_shape = input_shape   
                                                                                
    def compute_mask(self, input, input_mask=None):   
        # do not pass the mask to the next layers   
        return None   
         
    def call(self, x, mask=None):   
        return x   
             
    def get_output_shape_for(self, input_shape):   
        return input_shape  


class MyMMI:
    
    def __init__(self, fold=0, tasks=['sede1', 'sede12', 'morfo1', 'morfo2'], 
            corpusFolder="corpusMMI", foldsFolder="folds10", fileVectors="vectors.txt", 
            numFields=3, numPhrases=10, phraseLen=100, featuresDim=100, 
            learningRate=0.001, learningRateDecay=0., dropout=0.2, 
            patience=2, batchSize=65, epochs=100, 
            useMemmap=False, featuresType = 'lstm', #'lstm', 'bilstm', or 'cnn'
            noLstm=False, noLstmP=False, 
            ngrams=0, #if 0 not use nGrams, choose factor of phraseLen
            cnnWindow=3,
            pooling='max', #'max', 'avg' or 'none'
            masking=False,
            featuresLayers=1, hiddenLayers=0, hiddenDim=None, noNotizie=False,
            afterFeaturesLayers=0,
            filesFolder="./filesFolds-MMI", memmapFolder= "./memmapFolds-MMI"):

        self._tasks = tasks
        self._corpusFolder = "./"+corpusFolder
        self._fileVectors = self._corpusFolder+"/"+fileVectors
        self._featuresDim = featuresDim
        self._learningRate = learningRate
        self._learningRateDecay = learningRateDecay
        self._dropout = dropout
        self._patience = patience
        self._batchSize = batchSize
        self._epochs = epochs
        self._numFields = numFields
        self._numPhrases = numPhrases
        self._phraseLen = phraseLen
        self._useMemmap = useMemmap
        self._noLstm = noLstm
        self._noLstmP = noLstmP
        self._featuresType = featuresType
        self._pooling = pooling
        self._masking = masking
        self._ngrams = ngrams
        self._cnnWindow = cnnWindow
        self._featuresLayers = featuresLayers
        self._afterFeaturesLayers = afterFeaturesLayers
        self._hiddenLayers = hiddenLayers
        self._hiddenDim = hiddenDim
        self._noNotizie = noNotizie
        self._filesFolder = filesFolder
        self._memmapFolder = memmapFolder

        self.model = {}
        self.history = {}
        for task in self._tasks:
            self.model[task]=[]
            self.history[task]=[]

        self.textTrain = {}
        self.textTest = {}
        self.textFTrain = {}
        self.textFTest = {}
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
        self._fileMistakenSamplesBase = evaluationFolder+"/mistakenSamples"

        historyFolder = self._filesFolder+"/history/"+str(fold)
        if not os.path.exists(historyFolder):
            os.makedirs(historyFolder)
        self._fileDumpHistoryBase = historyFolder+"/historyCat"
        
        modelsFolder = self._filesFolder+"/models/"+str(fold)
        if not os.path.exists(modelsFolder):
            os.makedirs(modelsFolder)
        self._fileModelBase = modelsFolder+"/modelCat"

        self._fileTrain = {}
        self._fileTest = {}
        
        for task in self._tasks:
            corpusFoldFolder = self._corpusFolder+"/"+foldsFolder+"/"+task+"/"+str(fold)
            self._fileTrain[task] = corpusFoldFolder+"/corpusTrain.p"
            self._fileTest[task] = corpusFoldFolder+"/corpusTest.p"

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

    def _calculateFileMistakenSamples(self, task, index, top):
        if index == -1:
            index = 0
            while os.path.isfile(self._calculateFileMistakenSamples(task, index, top)):
                index += 1
            index -= 1

        return self._fileMistakenSamplesBase+task.capitalize()+"-"+str(index)+"-top"+str(top)+".p"

 

    def _processText(self, text, fileMemmap, vectors):
        if self._useMemmap:
            X = np.memmap(fileMemmap, mode='w+', shape=(len(text), self._numFields, self._numPhrases, self._phraseLen, self._vecLen), dtype=np.float)
        else:
            X = np.zeros(shape=(len(text), self._numFields, self._numPhrases, self._phraseLen, self._vecLen), dtype=np.float)


        documents = []
        textLen = len(text)
        for iDoc,l in enumerate(text):
            if iDoc%1000==0:
                print("         processed line {}/{}           ".format(iDoc,textLen), end='\r')
            
            fields = []
            for iField, f in enumerate(l):
                if self._noNotizie and iField == 0:
                    continue

                phrases = []
                for iPhrase, p in enumerate(f):

                    words = p.split()

                    iWordValid = 0
                    wordsF = []
                    for iWord in range(len(words)):
                        try:
                            X[iDoc, iField, iPhrase, iWordValid] = vectors[words[iWord]]
                            iWordValid += 1
                            wordsF.append(words[iWord])
                        except IndexError:
                            break
                        except KeyError:
                            pass
                    phrases.append(" ".join(wordsF))
                fields.append(phrases)
            documents.append(fields)

        print("         processed line {}/{}           ".format(iDoc+1,textLen))
        if self._useMemmap:
            X.flush()
        return (X,documents)
    
    def _unprocessSingleText(self, doc, index):
        fields = []
        for iField, f in enumerate(doc):
            phrases = []
            for iPhrase, p in enumerate(f):
                words = []
                for iWord, vector in enumerate(p):
                    if not np.all(vector == 0):
                        words.append(index[tuple(vector)])
                phrases.append(" ".join(words))
            fields.append(phrases)

        return fields

    def _unprocessText(self, X, index):
        documents = []
        XLen = len(X)
        for iDoc,l in enumerate(X):
            if iDoc%1000==0:
                print("         processed line {}/{}           ".format(iDoc,XLen), end='\r')
            
            documents.append(self._unprocessSingleText(l))

        print("         processed line {}/{}           ".format(iDoc+1,XLen))
        return documents
                    
    def extractData(self):
        print("Extracting data")

        vectors = {}
        with open(self._fileVectors) as fid:
            for line in fid.readlines():
                sline = line.split()
                currVec = []
                for i in range(1,len(sline)):
                    currVec.append(sline[i])

                vectors[sline[0]] = currVec
            self._vecLen = len(currVec)

            
        try:
            shapes = pickle.load(open(self._fileShapes, 'rb'))
        except FileNotFoundError:
            shapes = {}
            
        shapes['numFields'] = self._numFields
        shapes['numPhrases'] = self._numPhrases
        shapes['phraseLen'] = self._phraseLen
        shapes['vecLen'] = self._vecLen
        
        for task in self._tasks:
            print("   "+task)

            corpusTrain = pickle.load(open(self._fileTrain[task], 'rb'))
            corpusTest = pickle.load(open(self._fileTest[task], 'rb'))

            self.textTrain[task] = corpusTrain['X']
            self.textTest[task] = corpusTest['X']

            yTrain = corpusTrain['y']
            yTest = corpusTest['y']

            print("      train")
            self.XTrain[task], self.textFTrain[task] = self._processText(self.textTrain[task], self._fileMemmapXTrain[task], vectors)
            print("      test")            
            self.XTest[task], self.textFTest[task] = self._processText(self.textTest[task], self._fileMemmapXTest[task], vectors)

            #del self.textTrain[task]
            #del self.textTest[task]

            self.lb[task] = LabelBinarizer()
            self.lb[task].fit(np.concatenate((yTrain, yTest)))
            #self.lb[task].fit(yTrain)

            shapes[task] = {}
            shapes[task]['XTrain'] = self.XTrain[task].shape
            shapes[task]['XTest'] = self.XTest[task].shape

            if self._useMemmap:
                self.yTrain[task] = np.memmap(self._fileMemmapYTrain[task], mode='w+', shape=(len(yTrain), len(self.lb[task].classes_)), dtype=np.int)
                self.yTest[task] = np.memmap(self._fileMemmapYTest[task], mode='w+', shape=(len(yTest), len(self.lb[task].classes_)), dtype=np.int)
            else:
                self.yTrain[task] = np.zeros(shape=(len(yTrain), len(self.lb[task].classes_)), dtype=np.int)
                self.yTest[task] = np.zeros(shape=(len(yTest), len(self.lb[task].classes_)), dtype=np.int)
        
            self.yTrain[task][:] = self.lb[task].transform(yTrain)
            self.yTest[task][:] = self.lb[task].transform(yTest)
        
            if self._useMemmap:
                self.yTrain[task].flush()
                self.yTest[task].flush()
            shapes[task]['yTrain'] = self.yTrain[task].shape
            shapes[task]['yTest'] = self.yTest[task].shape

            pickle.dump(self.lb[task], open(self._fileLb[task], "wb"))
            
        pickle.dump(shapes, open(self._fileShapes, "wb"))

    def loadData(self):
        if not self._useMemmap:
            self.extractData()
            return
            #raise Exception("loadData only with memmap")

        shapes = pickle.load(open(self._fileShapes, 'rb'))
        self._numPhrases = shapes['numPhrases']
        self._phraseLen = shapes['phraseLen']
        self._vecLen = shapes['vecLen']

        for task in self._tasks:
            self.XTrain[task] = np.memmap(self._fileMemmapXTrain[task], mode='r', shape=shapes[task]['XTrain'], dtype=np.float)
            self.XTest[task] = np.memmap(self._fileMemmapXTest[task], mode='r', shape=shapes[task]['XTest'], dtype=np.float)

            self.yTrain[task] = np.memmap(self._fileMemmapYTrain[task], mode='r', shape=shapes[task]['yTrain'], dtype=np.int)
            self.yTest[task] = np.memmap(self._fileMemmapYTest[task], mode='r', shape=shapes[task]['yTest'], dtype=np.int)

            self.lb[task] = pickle.load(open(self._fileLb[task], 'rb'))

    def _createModel(self, outDim):
        model = Sequential()
        
        if self._ngrams != 0 and (self._featuresType == 'lstm' or self._featuresType == 'bilstm'):
            model.add(Reshape(target_shape = (self._numFields * self._numPhrases * int(self._phraseLen / self._ngrams), self._ngrams, self._vecLen)))
            
            if self._masking:
                model.add(Masking(mask_value=0.))
        
            for layer in range(self._featuresLayers):
                if self._noLstm:
                    model.add(TimeDistributed(Dense(self._featuresDim)))
                else:
                    if self._featuresType == 'bilstm':
                        model.add(TimeDistributed(Bidirectional(LSTM(self._featuresDim, dropout=self._dropout, recurrent_dropout=self._dropout, return_sequences=(self._pooling != 'none')))))
                    else:
                        model.add(TimeDistributed(LSTM(self._featuresDim, dropout=self._dropout, recurrent_dropout=self._dropout, return_sequences=(self._pooling != 'none'))))

                for _ in range(self._afterFeaturesLayers):
                    if self._featuresType == 'bilstm':
                        model.add(TimeDistributed(Dense(self._featuresDim*2, activation='relu')))
                    else:
                        model.add(TimeDistributed(Dense(self._featuresDim, activation='relu')))

            if self._masking:
                model.add(NonMasking())
        
            if self._pooling == 'avg':
                model.add(TimeDistributed(GlobalAveragePooling1D()))
            elif self._pooling == 'max':
                model.add(TimeDistributed(GlobalMaxPooling1D()))

            if not self._noLstm and not self._noLstmP and self._featuresType == 'bilstm':
                model.add(Reshape(target_shape = (self._numFields * self._numPhrases, int(self._phraseLen / self._ngrams), self._featuresDim*2)))
            else:
                model.add(Reshape(target_shape = (self._numFields * self._numPhrases, int(self._phraseLen / self._ngrams), self._featuresDim)))
     
        else:
            model.add(Reshape(target_shape = (self._numFields * self._numPhrases, self._phraseLen, self._vecLen)))

        if self._masking:
            model.add(Masking(mask_value=0.))
        
        for layer in range(self._featuresLayers):
            if self._noLstm or (self._ngrams != 0 and self._noLstmP):
                model.add(TimeDistributed(Dense(self._featuresDim)))
            else:
                if self._featuresType == 'bilstm':
                    model.add(TimeDistributed(Bidirectional(LSTM(self._featuresDim, dropout=self._dropout, recurrent_dropout=self._dropout, return_sequences=(self._pooling != 'none')))))
                elif self._featuresType == 'lstm':
                    model.add(TimeDistributed(LSTM(self._featuresDim, dropout=self._dropout, recurrent_dropout=self._dropout, return_sequences=(self._pooling != 'none'))))
                elif self._featuresType == 'cnn':
                    model.add(TimeDistributed(Conv1D(self._featuresDim, kernel_size=self._cnnWindow, padding='same', activation='relu')))

            for _ in range(self._afterFeaturesLayers):
                if self._featuresType == 'bilstm':
                    model.add(TimeDistributed(Dense(self._featuresDim*2, activation='relu')))
                else:
                    model.add(TimeDistributed(Dense(self._featuresDim, activation='relu')))

            #if not (self._featuresType == 'cnn' and layer < self._featuresLayers -1):

        if self._masking:
            model.add(NonMasking())
        
        if self._pooling == 'avg':
            model.add(TimeDistributed(GlobalAveragePooling1D()))
        elif self._pooling == 'max':
            model.add(TimeDistributed(GlobalMaxPooling1D()))

        if not self._noLstm and not self._noLstmP and self._featuresType == 'bilstm':
            model.add(Reshape(target_shape = (self._numFields, self._numPhrases, self._featuresDim*2)))
        else:
            model.add(Reshape(target_shape = (self._numFields, self._numPhrases, self._featuresDim)))

        if self._masking:
            model.add(Masking(mask_value=0.))
        
        for layer in range(self._featuresLayers):
            if self._noLstm or self._noLstmP:
                model.add(TimeDistributed(Dense(self._featuresDim)))
            else:
                if self._featuresType == 'bilstm':
                    model.add(TimeDistributed(Bidirectional(LSTM(self._featuresDim, dropout=self._dropout, recurrent_dropout=self._dropout, return_sequences=(self._pooling != 'none')))))
                elif self._featuresType == 'lstm':
                    model.add(TimeDistributed(LSTM(self._featuresDim, dropout=self._dropout, recurrent_dropout=self._dropout, return_sequences=(self._pooling != 'none'))))
                elif self._featuresType == 'cnn':
                    model.add(TimeDistributed(Conv1D(self._featuresDim, kernel_size=self._cnnWindow, padding='same', activation='relu')))
            
            for _ in range(self._afterFeaturesLayers):
                if self._featuresType == 'bilstm':
                    model.add(TimeDistributed(Dense(self._featuresDim*2, activation='relu')))
                else:
                    model.add(TimeDistributed(Dense(self._featuresDim, activation='relu')))

            #if not (self._featuresType == 'cnn' and layer < self._featuresLayers -1):
        
        if self._masking:
            model.add(NonMasking())
        
        if self._pooling == 'avg':
            model.add(TimeDistributed(GlobalAveragePooling1D()))
        elif self._pooling == 'max':
            model.add(TimeDistributed(GlobalMaxPooling1D()))

        if self._masking:
            model.add(Masking(mask_value=0.))
        
        for layer in range(self._featuresLayers):
            if self._noLstm or self._noLstmP:
                model.add(Dense(self._featuresDim))
            else:
                if self._featuresType == 'bilstm':
                    model.add(Bidirectional(LSTM(self._featuresDim, dropout=self._dropout, recurrent_dropout=self._dropout, return_sequences=(self._pooling != 'none'))))
                elif self._featuresType == 'lstm':
                    model.add(LSTM(self._featuresDim, dropout=self._dropout, recurrent_dropout=self._dropout, return_sequences=(self._pooling != 'none')))
                elif self._featuresType == 'cnn':
                    model.add(Conv1D(self._featuresDim, kernel_size=self._cnnWindow, padding='same', activation='relu'))
            
            for _ in range(self._afterFeaturesLayers):
                if self._featuresType == 'bilstm':
                    model.add(Dense(self._featuresDim*2, activation='relu'))
                else:
                    model.add(Dense(self._featuresDim, activation='relu'))

            #if not (self._featuresType == 'cnn' and layer < self._featuresLayers -1):
        
        if self._masking:
            model.add(NonMasking())
        
        if self._pooling == 'avg':
            model.add(GlobalAveragePooling1D())
        elif self._pooling == 'max':
            model.add(GlobalMaxPooling1D())

        model.add(Dropout(self._dropout))
        for _ in range(self._hiddenLayers):
            if self._hiddenDim is None:
                model.add(Dense(outDim, activation='relu'))
            else:
                model.add(Dense(self._hiddenDim, activation='relu'))
        model.add(Dense(outDim, activation='softmax'))

        #opt = RMSprop(lr=self._learningRate, decay=self._learningRateDecay)
        opt = Adam(lr=self._learningRate, decay=self._learningRateDecay)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

            
    def createModels(self):
        print("Creating models")

        for task in self._tasks:
            print("   "+task)
            self.model[task] = [self._createModel(self.yTrain[task].shape[1])]
            self.history[task] = [self.model[task][0].fit(self.XTrain[task], self.yTrain[task], validation_split=0.2, epochs=self._epochs, batch_size=self._batchSize, callbacks=[self._early_stopping]).history]

    def continueTrainModels(self, fromModel=-1, newLearningRate=None, useEarlyStopping=False, epochs=None):
        print("Continue train models")
        for task in self._tasks:
            print("   "+task)

            #save and reload to make a copy
            fileModel = self._calculateFileModel(task, fromModel)
            self.model[task][fromModel].save(fileModel)
            if self._masking:
                self.model[task].append(load_model(fileModel, custom_objects={'NonMasking': NonMasking}))
            else:
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

    def plotSummary(self):
        print("Summary")
        for task in self._tasks:
            print("Task: {}".format(task))
            for currIndex, currModel in enumerate(self.model[task]):
                print("Model: {}".format(currIndex))
                currModel.summary()

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

                if self._masking:
                    self.model[task].append(load_model(currModelFile, custom_objects={'NonMasking': NonMasking}))
                else:
                    self.model[task].append(load_model(currModelFile))

                self.history[task].append(pickle.load(open(currHistoryFile, 'rb')))
                currIndex += 1

    def loadSpecificModels(self, nums):
        print("loading models")
        for task in self._tasks:
            currHistoryFile = self._calculateFileDumpHistory(task, nums[task])
            currModelFile = self._calculateFileModel(task, nums[task])

            if self._masking:
                self.model[task].append(load_model(currModelFile, custom_objects={'NonMasking': NonMasking}))
            else:
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

    def calculateMistakenSamples(self, top=1):
        print("Calculating mistaken samples")
        mistakens = {'index':[], 'text':[], 'textF':[], 'yTrue':[], 'yPredicted':[]}

        #index = {}
        #with open(self._fileVectors) as fid:
        #    for line in fid.readlines():
        #        sline = line.split()
        #        currVec = []
        #        for i in range(1,len(sline)):
        #            currVec.append(sline[i])
        #
        #        index[tuple(currVec)] = sline[0]

        

        for task in self._tasks:
            print(task)
            for modelIndex, currModel in enumerate(self.model[task]):
                if not os.path.isfile(self._calculateFileMistakenSamples(task, modelIndex, top)):
                    yp = currModel.predict_proba(self.XTest[task])
                    yt = self.yTest[task]

                    yc = np.zeros((top, yt.shape[0], yt.shape[1]), np.int)
                    for i,p in enumerate(yp):
                        #yc[i][np.argmax(p)] = 1
                        indTops = np.argpartition(p, - top)[- top:]
                        for j, ind in enumerate(indTops):
                            yc[j][i][ind] = 1
                    
                    #ypn = self.lb[task].inverse_transform(yc)
                    ypn = []
                    for i in range(top):
                        ypn.append(self.lb[task].inverse_transform(yc[i]))
                    ytn = self.lb[task].inverse_transform(yt)

                    for i in range(len(ytn)):
                        currYpn = []
                        for j in range(top):
                            currYpn.append(ypn[j][i])
                        #if ypn[i] != ytn[i]:
                        if ytn[i] not in currYpn:
                            mistakens['index'].append(i)
                            #mistakens['text'].append(self._unprocessSingleText(self.XTest[task][i], index))
                            mistakens['text'].append(self.textTest[task][i])
                            mistakens['textF'].append(self.textFTest[task][i])
                            mistakens['yTrue'].append(ytn[i])
                            #mistakens['yPredicted'].append(ypn[i])
                            mistakens['yPredicted'].append(currYpn)

                    pickle.dump(mistakens, open(self._calculateFileMistakenSamples(task, modelIndex, top), "wb"))

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
       

