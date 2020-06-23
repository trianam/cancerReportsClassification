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
import scipy
import scipy.sparse

class MySVM:
    _filesFolder = "./filesFolds-SVMbigramsSeparated"
    _memmapFolder = "./memmapFolds-SVMbigramsSeparated"
    _tmpFolder = "./tmpFolds-SVMbigramsSeparated"
    #_filesFolder = "./filesFolds-SVM"
    #_memmapFolder = "./memmapFolds-SVM"
    #_tmpFolder = "./tmpFolds-SVM"
    
    _fileMemmapXNotizieTrain = {
        'sede1' : _tmpFolder+"/XNotizieTrainSede1.dat",
        'sede12' : _tmpFolder+"/XNotizieTrainSede12.dat",
        'morfo1' : _tmpFolder+"/XNotizieTrainMorfo1.dat",
        'morfo2' : _tmpFolder+"/XNotizieTrainMorfo2.dat",
    }
    _fileMemmapXNotizieTest = {
        'sede1' : _tmpFolder+"/XNotizieTestSede1.dat",
        'sede12' : _tmpFolder+"/XNotizieTestSede12.dat",
        'morfo1' : _tmpFolder+"/XNotizieTestMorfo1.dat",
        'morfo2' : _tmpFolder+"/XNotizieTestMorfo2.dat",
    }

    _fileMemmapXDiagnosiTrain = {
        'sede1' : _tmpFolder+"/XDiagnosiTrainSede1.dat",
        'sede12' : _tmpFolder+"/XDiagnosiTrainSede12.dat",
        'morfo1' : _tmpFolder+"/XDiagnosiTrainMorfo1.dat",
        'morfo2' : _tmpFolder+"/XDiagnosiTrainMorfo2.dat",
    }
    _fileMemmapXDiagnosiTest = {
        'sede1' : _tmpFolder+"/XDiagnosiTestSede1.dat",
        'sede12' : _tmpFolder+"/XDiagnosiTestSede12.dat",
        'morfo1' : _tmpFolder+"/XDiagnosiTestMorfo1.dat",
        'morfo2' : _tmpFolder+"/XDiagnosiTestMorfo2.dat",
    }

    _fileMemmapXMacroscopiaTrain = {
        'sede1' : _tmpFolder+"/XMacroscopiaTrainSede1.dat",
        'sede12' : _tmpFolder+"/XMacroscopiaTrainSede12.dat",
        'morfo1' : _tmpFolder+"/XMacroscopiaTrainMorfo1.dat",
        'morfo2' : _tmpFolder+"/XMacroscopiaTrainMorfo2.dat",
    }
    _fileMemmapXMacroscopiaTest = {
        'sede1' : _tmpFolder+"/XMacroscopiaTestSede1.dat",
        'sede12' : _tmpFolder+"/XMacroscopiaTestSede12.dat",
        'morfo1' : _tmpFolder+"/XMacroscopiaTestMorfo1.dat",
        'morfo2' : _tmpFolder+"/XMacroscopiaTestMorfo2.dat",
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
    
    def __init__(self, fold=0, tasks=['sede1', 'sede12', 'morfo1', 'morfo2'], corpusFolder="corpusLSTM_ICDO3-separated", foldsFolder="folds10"):
        self._tasks = tasks
        self._corpusFolder = "./"+corpusFolder
        
        #self._textFile = _corpusFolder+"/text.txt"

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
        self._filePredictionBase = evaluationFolder+"/prediction"

        modelsFolder = self._filesFolder+"/models/"+str(fold)
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

        self._textNotizieFileTrain = {}
        self._textNotizieFileTest = {}
        self._textDiagnosiFileTrain = {}
        self._textDiagnosiFileTest = {}
        self._textMacroscopiaFileTrain = {}
        self._textMacroscopiaFileTest = {}
        self._fileYTrain = {}
        self._fileYTest = {}
        
        for task in self._tasks:
            corpusFoldFolder = self._corpusFolder+"/"+foldsFolder+"/"+task+"/"+str(fold)
            self._textNotizieFileTrain[task] = corpusFoldFolder+"/textNotizieTrain.txt"
            self._textNotizieFileTest[task] = corpusFoldFolder+"/textNotizieTest.txt"
            self._textDiagnosiFileTrain[task] = corpusFoldFolder+"/textDiagnosiTrain.txt"
            self._textDiagnosiFileTest[task] = corpusFoldFolder+"/textDiagnosiTest.txt"
            self._textMacroscopiaFileTrain[task] = corpusFoldFolder+"/textMacroscopiaTrain.txt"
            self._textMacroscopiaFileTest[task] = corpusFoldFolder+"/textMacroscopiaTest.txt"
            self._fileYTrain[task] = corpusFoldFolder+"/yTrain.txt"
            self._fileYTest[task] = corpusFoldFolder+"/yTest.txt"

    def _calculateFilePrediction(self, task):
        return self._filePredictionBase+task.capitalize()+".p"

   
    def extractData(self):
        print("Extracting data")

        #with open(self._textFile) as fid:
        #    textFull = fid.readlines()

        self.XNotizieTrain = {}
        self.XNotizieTest = {}
        self.XDiagnosiTrain = {}
        self.XDiagnosiTest = {}
        self.XMacroscopiaTrain = {}
        self.XMacroscopiaTest = {}
        self.lb = {}
        self.yTrain = {}
        self.yTest = {}
        shapes = {}
        
        for task in self._tasks:
            print("   "+task)

            with open(self._textNotizieFileTrain[task]) as fid:
                textNotizieTrain = fid.readlines()

            with open(self._textNotizieFileTest[task]) as fid:
                textNotizieTest = fid.readlines()

            with open(self._textDiagnosiFileTrain[task]) as fid:
                textDiagnosiTrain = fid.readlines()

            with open(self._textDiagnosiFileTest[task]) as fid:
                textDiagnosiTest = fid.readlines()

            with open(self._textMacroscopiaFileTrain[task]) as fid:
                textMacroscopiaTrain = fid.readlines()

            with open(self._textMacroscopiaFileTest[task]) as fid:
                textMacroscopiaTest = fid.readlines()

            with open(self._fileYTrain[task]) as fid:
                yTrain = fid.readlines()

            with open(self._fileYTest[task]) as fid:
                yTest = fid.readlines()

            vectorizerNotizie = TfidfVectorizer(min_df=3, max_df=0.3, strip_accents='unicode', ngram_range=(1,2))
            #vectorizer = TfidfVectorizer(min_df=3, max_df=0.3, strip_accents='unicode', ngram_range=(1,1))
            #vectorizer.fit(textFull)
            vectorizerNotizie.fit(textNotizieTrain)
                
            vectorizerDiagnosi = TfidfVectorizer(min_df=3, max_df=0.3, strip_accents='unicode', ngram_range=(1,2))
            vectorizerDiagnosi.fit(textDiagnosiTrain)
            
            vectorizerMacroscopia = TfidfVectorizer(min_df=3, max_df=0.3, strip_accents='unicode', ngram_range=(1,2))
            vectorizerMacroscopia.fit(textMacroscopiaTrain)
            
            self.XNotizieTrain[task] = vectorizerNotizie.transform(textNotizieTrain)
            self.XNotizieTest[task] = vectorizerNotizie.transform(textNotizieTest)

            self.XDiagnosiTrain[task] = vectorizerDiagnosi.transform(textDiagnosiTrain)
            self.XDiagnosiTest[task] = vectorizerDiagnosi.transform(textDiagnosiTest)

            self.XMacroscopiaTrain[task] = vectorizerMacroscopia.transform(textMacroscopiaTrain)
            self.XMacroscopiaTest[task] = vectorizerMacroscopia.transform(textMacroscopiaTest)

            del textNotizieTrain
            del textNotizieTest

            del textDiagnosiTrain
            del textDiagnosiTest

            del textMacroscopiaTrain
            del textMacroscopiaTest

            pickle.dump(self.XNotizieTrain[task], open(self._fileMemmapXNotizieTrain[task], 'wb'))
            pickle.dump(self.XNotizieTest[task], open(self._fileMemmapXNotizieTest[task], 'wb'))

            pickle.dump(self.XDiagnosiTrain[task], open(self._fileMemmapXDiagnosiTrain[task], 'wb'))
            pickle.dump(self.XDiagnosiTest[task], open(self._fileMemmapXDiagnosiTest[task], 'wb'))

            pickle.dump(self.XMacroscopiaTrain[task], open(self._fileMemmapXMacroscopiaTrain[task], 'wb'))
            pickle.dump(self.XMacroscopiaTest[task], open(self._fileMemmapXMacroscopiaTest[task], 'wb'))

            self.lb[task] = LabelBinarizer()
            self.lb[task].fit(np.concatenate((yTrain, yTest))) #0.70 acc 
            #self.lb[task].fit(yTrain) #0.70 acc (identico)            

            shapes[task] = {}
            
            shapes[task]['XNotizieTrain'] = self.XNotizieTrain[task].shape
            shapes[task]['XNotizieTest'] = self.XNotizieTest[task].shape
            
            shapes[task]['XDiagnosiTrain'] = self.XDiagnosiTrain[task].shape
            shapes[task]['XDiagnosiTest'] = self.XDiagnosiTest[task].shape
            
            shapes[task]['XMacroscopiaTrain'] = self.XMacroscopiaTrain[task].shape
            shapes[task]['XMacroscopiaTest'] = self.XMacroscopiaTest[task].shape

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

        self.XNotizieTrain = {}
        self.XNotizieTest = {}
        self.XDiagnosiTrain = {}
        self.XDiagnosiTest = {}
        self.XMacroscopiaTrain = {}
        self.XMacroscopiaTest = {}
        self.yTrain = {}
        self.yTest = {}
        self.lb = {}

        for task in self._tasks:
            self.XNotizieTrain[task] = pickle.load(open(self._fileMemmapXNotizieTrain[task], 'rb'))
            self.XNotizieTest[task] = pickle.load(open(self._fileMemmapXNotizieTest[task], 'rb'))

            self.XDiagnosiTrain[task] = pickle.load(open(self._fileMemmapXDiagnosiTrain[task], 'rb'))
            self.XDiagnosiTest[task] = pickle.load(open(self._fileMemmapXDiagnosiTest[task], 'rb'))

            self.XMacroscopiaTrain[task] = pickle.load(open(self._fileMemmapXMacroscopiaTrain[task], 'rb'))
            self.XMacroscopiaTest[task] = pickle.load(open(self._fileMemmapXMacroscopiaTest[task], 'rb'))

            self.yTrain[task] = np.memmap(self._fileMemmapYTrain[task], mode='r', shape=shapes[task]['yTrain'], dtype=np.int)
            self.yTest[task] = np.memmap(self._fileMemmapYTest[task], mode='r', shape=shapes[task]['yTest'], dtype=np.int)

            self.lb[task] = pickle.load(open(self._fileLb[task], 'rb'))

    def createModels(self):
        print("Creating models")

        self.model = {}

        for task in self._tasks:
            print("   "+task)
            self.model[task] = OneVsRestClassifier(LinearSVC())
            self.model[task].fit(scipy.sparse.hstack([self.XNotizieTrain[task], self.XDiagnosiTrain[task], self.XMacroscopiaTrain[task]]), self.yTrain[task])

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
        print("Evaluating")
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
            prediction = {}
            yp[task] = self.model[task].decision_function(scipy.sparse.hstack([self.XNotizieTest[task], self.XDiagnosiTest[task], self.XMacroscopiaTest[task]]))
            yt[task] = self.yTest[task]
            prediction['yp'] = yp
            prediction['yt'] = yt
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
            
            pickle.dump(prediction, open(self._calculateFilePrediction(task), "wb"))


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

        

