import numpy as 
import pickle
import os
import math
from sklearn.eprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score, precision_recall_fscore_support, f1_score, accuracy_score, cohen_kappa_score
from tabulate import tabulate
from MAPScorer import MAPScorer
from scipy.interpolate import interp1d
from scipy import interp
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Masking
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Merge
from keras.layers import Dropout
from keras.layers.pooling import GlobalAveragePooling1D
from keras.optimizers import RMSprop
from keras import initializers
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from keras.models import load_model
from keras.callbacks import EarlyStopping


class MyLSTM:
    _filesFolder = "./filesFolds-LSTMbidirectional5b"
    _memmapFolder = "./memmapFolds-LSTMbidirectional"
    
    def __init__(self, fold=0, tasks=['sede1', 'sede12', 'morfo1', 'morfo2'], corpusFolder="corpusLSTM_ICDO3", foldsFolder="folds10", fileVectors="vectors.txt", phraseLen = 200, lstmCells=200, embeddingSize = 30, learningRate=0.01, dropout=0.5, patience=2, batchSize=65, epochs=100):
        self._tasks = tasks
        self._corpusFolder = "./"+corpusFolder
        self._fileVectors = self._corpusFolder+"/"+fileVectors
        self._lstmCells = lstmCells
        self._learningRate = learningRate
        self._dropout = dropout
        self._patience = patience
        self._batchSize = batchSize
        self._epochs = epochs
        self._phraseLen = phraseLen
        self._embeddingSize = embeddingSize

        self.model = {}
        self.history = {}
        self.historyContinue = {}

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
        self._fileEvaluation = evaluationFolder+"/evaluation.p"
        self._fileTable = evaluationFolder+"/table.txt"

        historyFolder = self._filesFolder+"/history/"+str(fold)
        if not os.path.exists(historyFolder):
            os.makedirs(historyFolder)
        self._fileDumpHistory = {}
        for task in self._tasks:
            self._fileDumpHistory[task] = historyFolder+"/historyCat"+task.capitalize()+".p"
        
        modelsFolder = self._filesFolder+"/models/"+str(fold)
        if not os.path.exists(modelsFolder):
            os.makedirs(modelsFolder)
        self._fileModel = {}
        for task in self._tasks:
            self._fileModel[task] = modelsFolder+"/modelCat"+task.capitalize()+".h5"

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


    def extractData(self):
        print("Extracting data")

        try:
            shapes = pickle.load(open(self._fileShapes, 'rb'))
        except FileNotFoundError:
            shapes = {}
            shapes['phraseLen'] = self._phraseLen
            shapes['vecLen'] = {}

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

            vectorizer = TfidfVectorizer(min_df=3, max_df=0.3, strip_accents='unicode', ngram_range=(1,2))
            #vectorizer = TfidfVectorizer(min_df=3, max_df=0.3, strip_accents='unicode', ngram_range=(1,1))
            #vectorizer.fit(textFull)
            vectorizer.fit(textTrain)
                
            self.XTrain[task] = vectorizer.transform(textTrain)
            self.XTest[task] = vectorizer.transform(textTest)

            shapes['vecLen'][task] = len(vectorizer.get_feature_names())

            del textTrain
            del textTest

            pickle.dump(self.XTrain[task], open(self._fileMemmapXTrain[task], 'wb'))
            pickle.dump(self.XTest[task], open(self._fileMemmapXTest[task], 'wb'))

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
            self.XTrain[task] = pickle.load(open(self._fileMemmapXTrain[task], 'rb'))
            self.XTest[task] = pickle.load(open(self._fileMemmapXTest[task], 'rb'))

            self.yTrain[task] = np.memmap(self._fileMemmapYTrain[task], mode='r', shape=shapes[task]['yTrain'], dtype=np.int)
            self.yTest[task] = np.memmap(self._fileMemmapYTest[task], mode='r', shape=shapes[task]['yTest'], dtype=np.int)

            self.lb[task] = pickle.load(open(self._fileLb[task], 'rb'))

    def _createModel(self, outDim, vecLen):
        model = Sequential()
        #model.add(Masking(mask_value=0., input_shape=(self._phraseLen, vecLen)))
        model.add(Embedding(vecLen, self._embeddingSize, input_length=self._phraseLen))
        model.add(Bidirectional(LSTM(self._lstmCells, dropout=self._dropout, recurrent_dropout=self._dropout, return_sequences=True)))
        model.add(Bidirectional(LSTM(self._lstmCells, dropout=self._dropout, recurrent_dropout=self._dropout, return_sequences=True)))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(outDim, activation='relu'))
        model.add(Dense(outDim, activation='relu'))
        model.add(Dense(outDim, activation='softmax'))

        opt = RMSprop(lr = self._learningRate)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

            
    def createModels(self):
        print("Creating models")

        for task in self._tasks:
            print("   "+task)
            self.model[task] = self._createModel(self.yTrain[task].shape[1], self._vecLen[task])
            self.history[task] = self.model[task].fit(self.XTrain[task], self.yTrain[task], validation_split=0.2, epochs=self._epochs, batch_size=self._batchSize, callbacks=[self._early_stopping]).history

    def continueTrainModels(self, newLearningRate):
        print("Continue train models")
        for task in self._tasks:
            print("   "+task)


            opt = RMSprop(lr = newLearningRate)
            self.model[task].compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

            if not task in self.historyContinue.keys():
                self.historyContinue[task] = []

            self.historyContinue[task].append(self.model[task].fit(self.XTrain[task], self.yTrain[task], validation_split=0.2, epochs=self._epochs, batch_size=self._batchSize, callbacks=[self._early_stopping]).history)

    def saveModels(self):
        print("Saving models")
        for task in self._tasks:
            pickle.dump(self.history[task], open(self._fileDumpHistory[task], "wb"))
            self.model[task].save(self._fileModel[task])
            if task in self.historyContinue:
                for indexHist, currHistory in enumerate(self.historyContinue[task]):
                    pickle.dump(currHistory, open(self._fileDumpHistory[task]+"-"+str(indexHist), "wb"))
            
    def loadModels(self):
        print("loading models")
        for task in self._tasks:
            self.model[task] = load_model(self._fileModel[task])
            self.history[task] = pickle.load(open(self._fileDumpHistory[task], 'rb'))
            
            indexHist = 0
            while True:
                currFile = self._fileDumpHistory[task]+"-"+str(indexHist)
                if not os.path.isfile(currFile):
                    break

                if not task in self.historyContinue.keys():
                    self.historyContinue[task] = []

                self.historyContinue[task].append(pickle.load(open(self._fileDumpHistory[task]+"-"+str(indexHist), 'rb')))
                indexHist += 1

    def evaluate(self):
        print("Evaluating")
        yp = {}
        ycn = {}
        yc = {}
        ytn = {}
        yt = {}

        try:
            metrics = pickle.load(open(self._fileEvaluation, 'rb'))
        except FileNotFoundError:
            metrics = {}

        table = [["task", "average", "MAPs", "MAPc", "accur.", "kappa", "prec.", "recall", "f1score"]]
        na = ' '
        
        for task in self._tasks:
            table.append([" ", " ", " ", " ", " ", " ", " ", " "])
            print(task)
            yp[task] = self.model[task].predict_proba(self.XTest[task])
            yt[task] = self.yTest[task]
            ytn[task] = self.lb[task].inverse_transform(yt[task])
            yc[task] = np.zeros(yt[task].shape, np.int)
            for i,p in enumerate(yp[task]):
                yc[task][i][np.argmax(p)] = 1
            ycn[task] = self.lb[task].inverse_transform(yc[task])

            metrics[task] = {}
            metrics[task]['MAPs'] = MAPScorer().samplesScore(yt[task], yp[task])
            metrics[task]['MAPc'] = MAPScorer().classesScore(yt[task], yp[task])
            metrics[task]['accuracy'] = accuracy_score(yt[task], yc[task])
            metrics[task]['kappa'] = cohen_kappa_score(ytn[task], ycn[task])

            metrics[task]['precision'] = {}
            metrics[task]['recall'] = {}
            metrics[task]['f1score'] = {}
            
            table.append([task, na, "{:.3f}".format(metrics[task]['MAPs']), "{:.3f}".format(metrics[task]['MAPc']), "{:.3f}".format(metrics[task]['accuracy']), "{:.3f}".format(metrics[task]['kappa']), na, na, na])
            for avg in ['micro', 'macro', 'weighted']:
                metrics[task]['precision'][avg], metrics[task]['recall'][avg], metrics[task]['f1score'][avg], _ = precision_recall_fscore_support(yt[task], yc[task], average=avg)
                table.append([task, avg, na, na, na, na, "{:.3f}".format(metrics[task]['precision'][avg]), "{:.3f}".format(metrics[task]['recall'][avg]), "{:.3f}".format(metrics[task]['f1score'][avg])])

            metrics[task]['pr-curve'] = {}
            metrics[task]['pr-curve']['x'], metrics[task]['pr-curve']['y'], metrics[task]['pr-curve']['auc'] = self._calculateMicroMacroCurve(lambda y,s: (lambda t: (t[1],t[0]))(precision_recall_curve(y,s)), yt[task], yp[task])
            
            metrics[task]['roc-curve'] = {}
            metrics[task]['roc-curve']['x'], metrics[task]['roc-curve']['y'], metrics[task]['roc-curve']['auc'] = self._calculateMicroMacroCurve(lambda y,s: (lambda t: (t[0],t[1]))(roc_curve(y,s)), yt[task], yp[task])


        pickle.dump(metrics, open(self._fileEvaluation, "wb"))
        
        tableString = tabulate(table)
        print(tableString)
        with open(self._fileTable, "w") as fid:
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

        

