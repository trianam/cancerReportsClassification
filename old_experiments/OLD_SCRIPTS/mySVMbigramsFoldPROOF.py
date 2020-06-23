import numpy as np
import pickle
import os
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

class MySVM:
    _filesFolder = "./filesFolds-SVMbigramsPROOF"
    _memmapFolder = "./memmapFolds-SVMbigramsPROOF"
    _tmpFolder = "./tmpFolds-SVMbigramsPROOF"
    
    _fileMemmapXTrain = {
        'sede1' : _tmpFolder+"/XTrainSede1.dat",
        'sede12' : _tmpFolder+"/XTrainSede12.dat",
        'morfo1' : _tmpFolder+"/XTrainMorfo1.dat",
        'morfo2' : _tmpFolder+"/XTrainMorfo2.dat",
    }
    _fileMemmapXTest = {
        'sede1' : _tmpFolder+"/XTestSede1.dat",
        'sede12' : _tmpFolder+"/XTestSede12.dat",
        'morfo1' : _tmpFolder+"/XTestMorfo1.dat",
        'morfo2' : _tmpFolder+"/XTestMorfo2.dat",
    }
    _fileMemmapYTrain = {
        'sede1' : _tmpFolder+"/yTrainSede1.dat",
        'sede12' : _tmpFolder+"/yTrainSede12.dat",
        'morfo1' : _tmpFolder+"/yTrainMorfo1.dat",
        'morfo2' : _tmpFolder+"/yTrainMorfo2.dat",
    }
    _fileMemmapYTest = {
        'sede1' : _tmpFolder+"/yTestSede1.dat",
        'sede12' : _tmpFolder+"/yTestSede12.dat",
        'morfo1' : _tmpFolder+"/yTestMorfo1.dat",
        'morfo2' : _tmpFolder+"/yTestMorfo2.dat",
    }

    def __init__(self, fold=0, tasks=['sede1', 'sede12', 'morfo1', 'morfo2'], corpusFolder="corpusLSTM_ICDO3", foldsFolder="randomSplit"):
        self._tasks = tasks
        self._corpusFolder = "./"+corpusFolder
        
        self._textFile = self._corpusFolder+"/text.txt"

        if not os.path.exists(self._tmpFolder):
            os.makedirs(self._tmpFolder)
        
        shapesFolder = self._memmapFolder+"/shapes/"+str(fold)
        if not os.path.exists(shapesFolder):
            os.makedirs(shapesFolder)
        self._fileShapes = shapesFolder+"/shapes.p"
       
        lbFolder = self._memmapFolder+"/binarizers/"+str(fold)
        if not os.path.exists(lbFolder):
            os.makedirs(lbFolder)
            
        self._fileLb = {
            'sede1' : lbFolder+"/lbSede1.p",
            'sede2' : lbFolder+"/lbSede2.p",
            'sede12' : lbFolder+"/lbSede12.p",
            'morfo1' : lbFolder+"/lbMorfo1.p",
            'morfo2' : lbFolder+"/lbMorfo2.p",
            'morfo12' : lbFolder+"/lbMorfo12.p"
        }
        evaluationFolder = self._filesFolder+"/output/"+str(fold)
        if not os.path.exists(evaluationFolder):
            os.makedirs(evaluationFolder)
        self._fileEvaluation = evaluationFolder+"/evaluation.p"

        modelsFolder = self._filesFolder+"/modelsSVM/"+str(fold)
        if not os.path.exists(modelsFolder):
            os.makedirs(modelsFolder)
        self._fileModel = {
            'sede1' : modelsFolder+"/modelCatSede1.h5",
            'sede2' : modelsFolder+"/modelCatSede2.h5",
            'sede12' : modelsFolder+"/modelCatSede12.h5",
            'morfo1' : modelsFolder+"/modelCatMorfo1.h5",
            'morfo2' : modelsFolder+"/modelCatMorfo2.h5",
            'morfo12' : modelsFolder+"/modelCatMorfo12.h5",
        }

        self._textFileTrain = {}
        self._textFileTest = {}
        self._fileYTrain = {}
        self._fileYTest = {}
        
        for task in self._tasks:
            corpusFoldFolder = self._corpusFolder+"/"+foldsFolder+"/"+task
            self._textFileTrain[task] = corpusFoldFolder+"/textTrain.txt"
            self._textFileTest[task] = corpusFoldFolder+"/textTest.txt"
            self._fileYTrain[task] = corpusFoldFolder+"/yTrain.txt"
            self._fileYTest[task] = corpusFoldFolder+"/yTest.txt"

    
    def extractData(self):
        print("Extracting data")

        with open(self._textFile) as fid:
            textFull = fid.readlines()

        self.XTrain = {}
        self.XTest = {}
        self.lb = {}
        self.yTrain = {}
        self.yTest = {}
        shapes = {}
        
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

            #vectorizer = TfidfVectorizer(min_df=3, max_df=0.3, strip_accents='unicode', ngram_range=(1,2))
            vectorizer = TfidfVectorizer(min_df=3, max_df=0.3, strip_accents='unicode', ngram_range=(1,1))
            #vectorizer.fit(textFull)
            vectorizer.fit(textTrain)
                
            self.XTrain[task] = vectorizer.transform(textTrain)
            self.XTest[task] = vectorizer.transform(textTest)

            del textTrain
            del textTest

            pickle.dump(self.XTrain[task], open(self._fileMemmapXTrain[task], 'wb'))
            pickle.dump(self.XTest[task], open(self._fileMemmapXTest[task], 'wb'))

            self.lb[task] = LabelBinarizer()
            self.lb[task].fit(np.concatenate((yTrain, yTest))) #0.70 acc 
            #self.lb[task].fit(yTrain) #0.70 acc (identico)            

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
            
        pickle.dump(shapes, open(self._fileShapes, "wb"))

    def loadData(self):
        shapes = pickle.load(open(self._fileShapes, 'rb'))

        self.XTrain = {}
        self.XTest = {}
        self.yTrain = {}
        self.yTest = {}
        self.lb = {}

        for task in self._tasks:
            self.XTrain[task] = pickle.load(open(self._fileMemmapXTrain[task], 'rb'))
            self.XTest[task] = pickle.load(open(self._fileMemmapXTest[task], 'rb'))

            self.yTrain[task] = np.memmap(self._fileMemmapYTrain[task], mode='r', shape=shapes[task]['yTrain'], dtype=np.int)
            self.yTest[task] = np.memmap(self._fileMemmapYTest[task], mode='r', shape=shapes[task]['yTest'], dtype=np.int)

            self.lb[task] = pickle.load(open(self._fileLb[task], 'rb'))

    def createModels(self):
        print("Creating models")

        self.model = {}

        for task in self._tasks:
            print("   "+task)
            self.model[task] = OneVsRestClassifier(LinearSVC())
            self.model[task].fit(self.XTrain[task], self.yTrain[task])

    def saveModels(self):
        print("Saving models")
        for task in self._tasks:
            with open(self._fileModel[task], 'wb') as fid:
                pickle.dump(self.model[task], fid)    

    def loadModels(self):
        print("loading models")
        self.model = {}
        for task in self._tasks:
            with open(self._fileModel[task], 'rb') as fid:
                self.model[task] = pickle.load(fid)

    def evaluate(self):
        print("Evaluating Test")
        self._evaluate(self.XTest, self.yTest)
        print("Evaluating Train")
        self._evaluate(self.XTrain, self.yTrain)
        
    def _evaluate(self, X, y):
        
        yp = {}
        ycn = {}
        yc = {}
        ytn = {}
        yt = {}

        metrics = {}

        table = [["task", "average", "MAPs", "MAPc", "accur.", "kappa", "prec.", "recall", "f1score"]]
        na = ' '
        
        for task in self._tasks:
            table.append([" ", " ", " ", " ", " ", " ", " ", " "])
            print(task)
            yp[task] = self.model[task].decision_function(X[task])
            yt[task] = y[task]
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
        print(tabulate(table))


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

        

