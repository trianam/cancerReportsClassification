import os
import numpy as np 
import random
import pickle
import time

import spacy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import torchsummary
from tensorboardX import SummaryWriter
#from torch.autograd import Variable

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from MAPScorer import MAPScorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score


import modelMLP
import modelBase
import model1
import model1sent
import model1plain
import model1plainMultitask
import model1plainSoftmax
import model2
import modelPrimeTest
import modelPrimeTestSoftmax
import modelPrimeTestBase
import modelPrimeTestOrder
import modelPrimeTestOrderBeta
import modelCNN

def makeModel(conf, device):

    #REMEMBER plainmodels in processData
    if conf.modelType == 'model1':
        model = model1.MyModel(conf)
    elif conf.modelType == 'model1sent':
        model = model1sent.MyModel(conf)
    elif conf.modelType == 'model1plain':
        model = model1plain.MyModel(conf)
    elif conf.modelType == 'model1plainMultitask':
        model = model1plainMultitask.MyModel(conf)
    elif conf.modelType == 'model1plainSoftmax':
        model = model1plainSoftmax.MyModel(conf)
    elif conf.modelType == 'model2':
        model = model2.MyModel(conf)
    elif conf.modelType == 'modelMLP':
        model = modelMLP.MyModel(conf)
    elif conf.modelType == 'modelBase':
        model = modelBase.MyModel(conf)
    elif conf.modelType == 'modelPrimeTest':
        model = modelPrimeTest.MyModel(conf)
    elif conf.modelType == 'modelPrimeTestSoftmax':
        model = modelPrimeTestSoftmax.MyModel(conf)
    elif conf.modelType == 'modelPrimeTestBase':
        model = modelPrimeTestBase.MyModel(conf)
    elif conf.modelType == 'modelPrimeTestOrder':
        model = modelPrimeTestOrder.MyModel(conf)
    elif conf.modelType == 'modelPrimeTestOrderBeta':
        model = modelPrimeTestOrderBeta.MyModel(conf)
    elif conf.modelType == 'modelCNN':
        model = modelCNN.MyModel(conf)
    else:
        raise Exception("Model don't exists")

    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=conf.learningRate, weight_decay=conf.learningRateDecay)

    return model,optim

def summary(conf, model):
    torchsummary.summary(model, (conf.batchSize, conf.numFields, conf.numPhrases, conf.phraseLen, conf.vecLen))

def processVectors(conf):
    vectors = {}
    with open(conf.fileVectors) as fid:
        for line in fid.readlines():
            sline = line.split()
            currVec = []
            for i in range(1,len(sline)):
                currVec.append(sline[i])

            vectors[sline[0]] = currVec
        vecLen = len(currVec)

    return (vectors, vecLen)

def processCorpusAndSplit(conf):
    values = pickle.load(open(conf.fileValues, 'rb'))
    corpus = pickle.load(open(conf.fileCorpus, 'rb'))

    #corpus to split
    if 'text' in corpus.keys():
        if not conf.loadSplit:
            np.random.seed(42)

            skf = StratifiedKFold(n_splits=10, shuffle=True)
          
            train = []
            valid = []
            test = []
            #for currTrainValid, currTest in skf.split(np.zeros(len(corpus[conf.corpusKey])), corpus[conf.corpusKey]):
            #    currTrain, currValid = train_test_split(currTrainValid, test_size=conf.validSplit/100., shuffle=True, stratify=np.array(corpus[conf.corpusKey])[currTrainValid])
            #    train.append(currTrain)
            #    valid.append(currValid)
            #    test.append(currTest)

            alls = np.array(list(range(len(corpus[conf.corpusKey]))))
            for _, currTest in skf.split(np.zeros(len(corpus[conf.corpusKey])), corpus[conf.corpusKey]):
                test.append(currTest)

            for f in range(len(test)):
                valid.append(test[(f+1)%len(test)])
            
            for f in range(len(test)):
                train.append(np.setdiff1d(alls, np.union1d(valid[f], test[f])))

            train = np.array(train)
            valid = np.array(valid)
            test = np.array(test)
            
            if conf.saveSplit:
                pickle.dump({'train':train, 'valid':valid, 'test':test}, open(conf.fileSplit, 'wb'))

        else:
            split = pickle.load(open(conf.fileSplit, 'rb'))
            train = split['train']
            valid = split['valid']
            test = split['test']

        if conf.has("fold"):
            return (corpus, values, train[conf.fold], valid[conf.fold], test[conf.fold])
        else:
            return (corpus, values, train, valid, test)
    else: #corpus already splitted
        if conf.has("corpusKey") and type(conf.corpusKey) is list: #if multitask
            keys = conf.corpusKey.copy()
        else:
            keys = [conf.corpusKey]
        keys.append('text')

        newCorpus = {}
        for k in keys:
            newCorpus[k] = np.concatenate([corpus[d][k] for d in ['train', 'valid', 'test']])
        train = np.array(range(len(corpus['train']['text'])))
        valid = np.array(range(len(corpus['train']['text']), len(corpus['train']['text'])+len(corpus['valid']['text'])))
        test = np.array(range(len(corpus['train']['text'])+len(corpus['valid']['text']), len(corpus['train']['text'])+len(corpus['valid']['text'])+len(corpus['test']['text'])))

        return (newCorpus, values, train, valid, test)


def processDataMulticlassSent(conf, getText=False):
    corpus, values, train, valid, test = processCorpusAndSplit(conf) 
    vectors, veLen = processVectors(conf)

   
    if conf.preprocess == "save":
        Xlist = []
        textList = []

        #HACK to be size compatible with plain
        textLen = len(corpus['text'])
        for iDoc,l in enumerate(corpus['text']):
            if iDoc%1000==0:
                print("         preprocessed line {}/{}           ".format(iDoc,textLen), end='\r', flush=True)
            currText = []

            if type(l) is list: #if text splitted in fields
                l = " ".join([" ".join(s) for s in l])

            words = l.split()
            iWordValid = 0
            for iWord in range(len(words)):
                if iWordValid < conf.phraseLen:
                    try:
                        _ = vectors[words[iWord]]
                        currText.append(words[iWord])

                        iWordValid += 1
                    except KeyError:
                        pass


            textList.append(currText)

        print("         preprocessed line {}/{}           ".format(iDoc+1,textLen), flush=True)
     

        nlp = spacy.load('it_core_news_sm')

        textLen = len(textList)
        for iDoc,l in enumerate(textList):
            if iDoc%1000==0:
                print("         processed line {}/{}           ".format(iDoc,textLen), end='\r', flush=True)
            currX = []


            for iSent,s in enumerate(nlp(" ".join(l)).sents):
                if iSent < conf.numPhrases:
                    currS = []
                    words = str(s).split()
                    for iWord in range(len(words)):
                        try:
                            currS.append(vectors[words[iWord]])
                        except KeyError:
                            pass

                    currX.append(currS)

            Xlist.append(currX)

        print("         processed line {}/{}           ".format(iDoc+1,textLen), flush=True)

        toSave = {'Xlist':Xlist, 'textList':textList}
        pickle.dump(toSave, open(conf.preprocessFile, 'wb'))

    elif conf.preprocess == "load":
        toLoad = pickle.load(open(conf.preprocessFile, 'rb'))
        Xlist, textList = toLoad['Xlist'], toLoad['textList']

    else:
        raise ValueError("use save or load")

    X = np.zeros(shape=(len(corpus['text']), conf.numPhrases, conf.phraseLen, conf.vecLen), dtype=np.float32)
    if conf.outDim > 1:
        y = np.zeros(shape=(len(corpus['text'])), dtype=np.long)
    else:
        y = np.zeros(shape=(len(corpus['text']), 1), dtype=np.float32)

    textLen = len(Xlist)
    for iDoc,l in enumerate(Xlist):
        if iDoc%1000==0:
            print("         reprocessed line {}/{}           ".format(iDoc,textLen), end='\r', flush=True)
            
        y[iDoc] = values.index(corpus[conf.corpusKey][iDoc])
        for iSent,s in enumerate(l):
            for iWord,w in enumerate(s):
                for iVec,v in enumerate(w):
                    X[iDoc, iSent, iWord, iVec] = v
    print("         reprocessed line {}/{}           ".format(iDoc+1,textLen), flush=True)
        
    if getText:
        return(X, y, train, valid, test, textList)
    else:
        return(X, y, train, valid, test)



def processDataMulticlassPlain(conf, getText=False):
    corpus, values, train, valid, test = processCorpusAndSplit(conf) 
    vectors, veLen = processVectors(conf)

    Xlist = []
    textList = []
    
    if conf.outDim > 1:
        #TODO:implement also on other processors/loaders
        if conf.has("corpusKey") and type(conf.corpusKey) is list: #if multitask
            y = [np.zeros(shape=(len(corpus['text'])), dtype=np.long) for _ in range(len(conf.corpusKey))]
        else:
            y = np.zeros(shape=(len(corpus['text'])), dtype=np.long)
    else:
        if conf.has("corpusKey") and type(conf.corpusKey) is list: #if multitask
            y = [np.zeros(shape=(len(corpus['text'])), dtype=np.float32) for _ in range(len(conf.corpusKey))]
        else:
            y = np.zeros(shape=(len(corpus['text']), 1), dtype=np.float32)
    
    textLen = len(corpus['text'])
    for iDoc,l in enumerate(corpus['text']):
        if iDoc%1000==0:
            print("         processed line {}/{}           ".format(iDoc,textLen), end='\r', flush=True)
        currX = []
        currText = []

        if conf.has("corpusKey") and type(conf.corpusKey) is list: #if multitask
            for iKey in range(len(conf.corpusKey)):
                y[iKey][iDoc] = values[conf.corpusKey[iKey]].index(corpus[conf.corpusKey[iKey]][iDoc])
        else:
            y[iDoc] = values.index(corpus[conf.corpusKey][iDoc])

        #corpus to plainify
        if (not type(l) is str) and (not type(l) is np.str_): 
            for iField, f in enumerate(l):
                if iField < conf.numFields:
                    for iPhrase, p in enumerate(f):
                        if iPhrase < conf.numPhrases:
                            words = p.split()
                            iWordValid = 0
                            for iWord in range(len(words)):
                                if iWordValid < conf.phraseLen:
                                    try:
                                        currX.append(vectors[words[iWord]])
                                        if getText:
                                            currText.append(words[iWord])

                                        iWordValid += 1
                                    except KeyError:
                                        pass
        
        else:  #corpus already plain
            words = l.split()
            iWordValid = 0
            for iWord in range(len(words)):
                if iWordValid < conf.phraseLen:
                    try:
                        currX.append(vectors[words[iWord]])
                        if getText:
                            currText.append(words[iWord])

                        iWordValid += 1
                    except KeyError:
                        pass


        Xlist.append(currX)
        if getText:
            textList.append(currText)

    print("         processed line {}/{}           ".format(iDoc+1,textLen), flush=True)
    
    X = np.zeros(shape=(len(corpus['text']), max(map(len, Xlist)), conf.vecLen), dtype=np.float32)

    for iDoc,l in enumerate(Xlist):
        if iDoc%1000==0:
            print("         reprocessed line {}/{}           ".format(iDoc,textLen), end='\r', flush=True)
        for iWord,w in enumerate(l):
            for iVec,v in enumerate(w):
                X[iDoc, iWord, iVec] = v
    print("         reprocessed line {}/{}           ".format(iDoc+1,textLen), flush=True)
        
    if getText:
        return(X, y, train, valid, test, textList)
    else:
        return(X, y, train, valid, test)

def processDataMulticlassStructured(conf, getText=False):
    #TODO: implement getText

    corpus, values, train, valid, test = processCorpusAndSplit(conf) 
    vectors, veLen = processVectors(conf)

    X = np.zeros(shape=(len(corpus['text']), conf.numFields, conf.numPhrases, conf.phraseLen, conf.vecLen), dtype=np.float32)

    #y = np.zeros(shape=(len(corpus['text']), len(values)), dtype=np.float32)
    if conf.outDim > 1:
        y = np.zeros(shape=(len(corpus['text'])), dtype=np.long)
    else:
        y = np.zeros(shape=(len(corpus['text']), 1), dtype=np.float32)

    #documents = []
    textLen = len(corpus['text'])
    for iDoc,l in enumerate(corpus['text']):
        if iDoc%1000==0:
            print("         processed line {}/{}           ".format(iDoc,textLen), end='\r', flush=True)
       
        #y one hot
        #y[iDoc][values.index(corpus[conf.corpusKey][iDoc])] = 1.
        #y position
        y[iDoc] = values.index(corpus[conf.corpusKey][iDoc])

        #fields = []
        for iField, f in enumerate(l):
            #if self._noNotizie and iField == 0:
            #    continue

            #phrases = []
            for iPhrase, p in enumerate(f):

                words = p.split()

                iWordValid = 0
                #wordsF = []
                for iWord in range(len(words)):
                    try:
                        X[iDoc, iField, iPhrase, iWordValid] = vectors[words[iWord]]
                        iWordValid += 1
                        #wordsF.append(words[iWord])
                    except IndexError:
                        break
                    except KeyError:
                        pass
                #phrases.append(" ".join(wordsF))
            #fields.append(phrases)
        #documents.append(fields)

    print("         processed line {}/{}           ".format(iDoc+1,textLen), flush=True)

    return(X, y, train, valid, test)

def loadData(conf, getText=False):
    #TODO: implement getText

    if hasattr(conf, 'fileX'):
        X = np.load(conf.fileX)
        y = np.load(conf.fileY)

        X = X.astype('float32')
        y = y.astype('float32')

        y = y.reshape((len(y),1))
    else:
        corpus = np.load(conf.fileCorpus)
        X = corpus['X']
        y = corpus['y']
        y = y.astype('float32')
    
    if conf.outDim == 1:
        #binary dataset
        if conf.loadSplit:
            split = pickle.load(open(conf.fileSplit, 'rb'))
            train = split['train']
            valid = split['valid']
            test = split['test']
        else:
            i0 = []
            i1 = []
            for i in range(len(y)):
                if y[i] == 0.:
                    i0.append(i)
                else:
                    i1.append(i)
            random.shuffle(i0)
            random.shuffle(i1)

            trainSplit0 = int(len(i0)/100*conf.trainSplit)
            trainSplit1 = int(len(i0)/100*conf.trainSplit)

            validSplit0 = trainSplit0 + int((len(i0) - trainSplit0)/100*conf.validSplit)
            validSplit1 = trainSplit1 + int((len(i1) - trainSplit1)/100*conf.validSplit)

            train = i0[:trainSplit0] + i1[:trainSplit1]
            valid = i0[trainSplit0:validSplit0] + i1[trainSplit1:validSplit1]
            test = i0[validSplit0:] + i1[validSplit1:]

            random.shuffle(train)
            random.shuffle(valid)
            random.shuffle(test)

            if conf.saveSplit:
                split = {
                    "train":train,
                    "valid":valid,
                    "test":test,
                }

                pickle.dump(split, open(conf.fileSplit, 'wb'))

    #TODO: implement multiclass
    #else:
        #multiclass
        #if conf.loadSplit:
        #else:

    return(X,y,np.array(train),np.array(valid),np.array(test))

def processData(conf, getText=False):
    if conf.dataMethod == 'process':
        plainModels = ['modelMLP', 'modelBase', 'model1plain', 'model1plainMultitask', 'model1plainSoftmax', 'modelCNN']
        sentModels = ['model1sent']

        if conf.modelType in plainModels:
            return processDataMulticlassPlain(conf, getText)
        elif conf.modelType in sentModels:
            return processDataMulticlassSent(conf, getText)
        else:
            return processDataMulticlassStructured(conf, getText)

    elif conf.dataMethod == 'load':
        return loadData(conf, getText)

def loadCorpus(conf):
    _, extension = os.path.splitext(conf.fileCorpus)
    if extension == '.p':
        return pickle.load(open(conf.fileCorpus, 'rb'))
    elif extension == '.npz':
        return np.load(conf.fileCorpus)

def evaluate(model, lossFun, dataloader, getWrongIndexes=False, getPredictions=False):
    device = next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0
    if getWrongIndexes:
        wrongIndexes = []
    if getPredictions:
        ybs = []
        yps = []
        ypbs = []
    with torch.no_grad():
        runningLoss = 0.
        for data in dataloader:
            x,y,i = data
            x,y = x.to(device), y.to(device)

            yp = model(x)
            loss = lossFun(yp, y)

            total += y.size(0)
            if yp.size(1) > 1: #multiclass
                yb = torch.argmax(yp, 1)
            else: #binary
                yb = (yp > 0.5).float()

            if getWrongIndexes:
                wrongIndexes.extend(list(i[(yb != y)]))

            if getPredictions:
                ybs.extend(y.detach().cpu().numpy())
                yps.extend(yp.detach().cpu().numpy())
                ypbs.extend(yb.detach().cpu().numpy())

            correct += (yb == y).sum().item()
            runningLoss += loss.item()

    toReturn = [(runningLoss / len(dataloader)), (100.*correct/total)]

    if getWrongIndexes:
        toReturn.append(np.array(wrongIndexes))

    if getPredictions:
        ybs = np.array(ybs)
        yps = np.array(yps)
        ypbs = np.array(ypbs)

        ys = np.zeros_like(yps)
        for i in range(len(ys)):
            ys[i][ybs[i]] = 1.

        toReturn.append(ys)
        toReturn.append(ybs)
        toReturn.append(yps)
        toReturn.append(ypbs)

    return toReturn

def evaluateMultitask(model, lossFun, dataloader, getWrongIndexes=False, getPredictions=False):
    device = next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0
    if getWrongIndexes:
        wrongIndexes = []
    if getPredictions:
        ybs0 = []
        yps0 = []
        ypbs0 = []
        ybs1 = []
        yps1 = []
        ypbs1 = []
    with torch.no_grad():
        runningLoss = 0.
        for data in dataloader:
            x,y0,y1,i = data
            x,y0,y1 = x.to(device), y0.to(device), y1.to(device)


            yp = model(x)
            loss = lossFun(yp[0], y0)+lossFun(yp[1], y1)


            total += y0.size(0) + y1.size(0)

            if yp[0].size(1) > 1: #multiclass
                yb0 = torch.argmax(yp[0], 1)
            else: #binary
                yb0 = (yp[0] > 0.5).float()

            if getWrongIndexes:
                wrongIndexes.extend(list(i[(yb0 != y0)]))

            if getPredictions:
                ybs0.extend(y0.detach().cpu().numpy())
                yps0.extend(yp[0].detach().cpu().numpy())
                ypbs0.extend(yb0.detach().cpu().numpy())

            
            if yp[1].size(1) > 1: #multiclass
                yb1 = torch.argmax(yp[1], 1)
            else: #binary
                yb1 = (yp[1] > 0.5).float()

            if getWrongIndexes:
                wrongIndexes.extend(list(i[(yb1 != y1)]))

            if getPredictions:
                ybs1.extend(y1.detach().cpu().numpy())
                yps1.extend(yp[1].detach().cpu().numpy())
                ypbs1.extend(yb1.detach().cpu().numpy())

            correct += (yb0 == y0).sum().item() + (yb1 == y1).sum().item()
            runningLoss += loss.item()

    toReturn = [(runningLoss / len(dataloader)), (100.*correct/total)]

    if getWrongIndexes:
        toReturn.append(np.array(wrongIndexes))

    if getPredictions:
        ybs0 = np.array(ybs0)
        yps0 = np.array(yps0)
        ypbs0 = np.array(ypbs0)

        ys0 = np.zeros_like(yps0)
        for i in range(len(ys0)):
            ys0[i][ybs0[i]] = 1.

        toReturn.append(ys0)
        toReturn.append(ybs0)
        toReturn.append(yps0)
        toReturn.append(ypbs0)

        ybs1 = np.array(ybs1)
        yps1 = np.array(yps1)
        ypbs1 = np.array(ypbs1)

        ys1 = np.zeros_like(yps1)
        for i in range(len(ys1)):
            ys1[i][ybs1[i]] = 1.

        toReturn.append(ys1)
        toReturn.append(ybs1)
        toReturn.append(yps1)
        toReturn.append(ypbs1)


    return toReturn

def runTrain(conf, model, optim, X, y, train, valid, test=None):
    device = next(model.parameters()).device
   
    if conf.has("corpusKey") and type(conf.corpusKey) is list: #if multitask
        #TODO:change for more than 2
        trainDataset = TensorDataset(torch.from_numpy(X[train]), torch.from_numpy(y[0][train]), torch.from_numpy(y[1][train]), torch.from_numpy(train.reshape(-1,1)))
        validDataset = TensorDataset(torch.from_numpy(X[valid]), torch.from_numpy(y[0][valid]), torch.from_numpy(y[1][valid]), torch.from_numpy(valid.reshape(-1,1)))
        if not test is None:
            testDataset = TensorDataset(torch.from_numpy(X[test]), torch.from_numpy(y[0][test]), torch.from_numpy(y[1][test]), torch.from_numpy(test.reshape(-1,1)))
    else:
        trainDataset = TensorDataset(torch.from_numpy(X[train]), torch.from_numpy(y[train]), torch.from_numpy(train.reshape(-1,1)))
        validDataset = TensorDataset(torch.from_numpy(X[valid]), torch.from_numpy(y[valid]), torch.from_numpy(valid.reshape(-1,1)))
        if not test is None:
            testDataset = TensorDataset(torch.from_numpy(X[test]), torch.from_numpy(y[test]), torch.from_numpy(test.reshape(-1,1)))
    trainDataloader = DataLoader(trainDataset, batch_size=conf.batchSize)
    validDataloader = DataLoader(validDataset, batch_size=conf.batchSize)
    if not test is None:
        testDataloader = DataLoader(testDataset, batch_size=conf.batchSize)

    if not conf.tensorboard is None:
        writer = SummaryWriter(conf.tensorboard, flush_secs=60)

    if conf.loss == 'binary_crossentropy':
        lossFun = nn.BCELoss()
    elif conf.loss == 'categorical_crossentropy':
        #lossFun = nn.NLLLoss()
        lossFun = nn.CrossEntropyLoss()

    if not conf.earlyStopping is None:
        maxAccuracy = 0.
        currPatience = 0.

    #if not conf.earlyStopping is None:
    #    previousAccuracy = 0.
    #    currPatience = 0

    #if not conf.earlyStopping is None:
    #    previousAccuracies = np.zeros(conf.earlyStopping)

    if not os.path.exists(os.path.dirname(conf.modelFile)):
        os.makedirs(os.path.dirname(conf.modelFile))

    bestValidAcc = -1
    for epoch in range(conf.startEpoch, conf.startEpoch+conf.epochs):
        print("epoch {}: ".format(epoch), end='', flush=True)

        model.train()
        for batchIndex, data in enumerate(trainDataloader):
            if conf.has("corpusKey") and type(conf.corpusKey) is list: #if multitask
                x,y0,y1,_ = data
                x,y0,y1 = x.to(device), y0.to(device), y1.to(device)
            else:
                x,y,_ = data
                x,y = x.to(device), y.to(device)

            model.zero_grad()
            yp = model(x, batchIndex=batchIndex)

            if conf.has("corpusKey") and type(conf.corpusKey) is list: #if multitask
                loss = lossFun(yp[0], y0)+lossFun(yp[1],y1)
            else:
                loss = lossFun(yp, y)
            loss.backward()
            optim.step()

        if conf.has("corpusKey") and type(conf.corpusKey) is list: #if multitask
            trainLoss, trainAcc = evaluateMultitask(model, lossFun, trainDataloader)
            validLoss, validAcc = evaluateMultitask(model, lossFun, validDataloader)
        else:
            trainLoss, trainAcc = evaluate(model, lossFun, trainDataloader)
            validLoss, validAcc = evaluate(model, lossFun, validDataloader)

        writerDictLoss = {
            'train': trainLoss,
            'valid': validLoss,
            }
        writerDictAcc = {
            'train': trainAcc,
            'valid': validAcc,
            }

        if conf.modelSave == "all":
            torch.save({
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'epoch': epoch
            }, conf.modelFile.format(epoch))
        elif conf.modelSave == "best":
            if validAcc > bestValidAcc:
                bestValidAcc = validAcc
                
                if os.path.isfile(conf.modelFile):
                    os.remove(conf.modelFile)

                torch.save({
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optim.state_dict(),
                'epoch': epoch
                }, conf.modelFile)


        printString = "train loss {:0.2f}, train acc {:0.1f} - valid loss {:0.2f}, valid acc {:0.1f}".format(trainLoss, trainAcc, validLoss, validAcc)

        if not test is None:
            if conf.has("corpusKey") and type(conf.corpusKey) is list: #if multitask
                testLoss, testAcc = evaluateMultitask(model, lossFun, testDataloader)
            else:
                testLoss, testAcc = evaluate(model, lossFun, testDataloader)
            writerDictLoss['test'] = testLoss
            writerDictAcc['test'] = testAcc
            printString += " - test loss {:0.2f}, test acc {:0.1f}".format(testLoss, testAcc)

        if not conf.tensorboard is None:
            writer.add_scalars('loss', writerDictLoss, epoch)

            writer.add_scalars('accuracy', writerDictAcc, epoch)

        print(printString, flush=True)
        
        if not conf.earlyStopping is None:
            if validAcc < maxAccuracy:
                if currPatience >= conf.earlyStopping:
                    break
                currPatience += 1
            else:
                maxAccuracy = validAcc
                currPatience = 0
            
        #if not conf.earlyStopping is None:
        #    if validAcc < previousAccuracy:
        #        if currPatience >= conf.earlyStopping:
        #            break
        #        currPatience += 1
        #    else:
        #        currPatience = 0

        #if not conf.earlyStopping is None:
        #    if epoch >= conf.earlyStopping:
        #        if (validAcc <= previousAccuracies).all():
        #            break
        #    previousAccuracies[epoch%conf.earlyStopping] = validAcc

    time.sleep(120) #time to write tensorboard

def runTest(conf, model, X, y, test, verbose=True):
    if conf.has("corpusKey") and type(conf.corpusKey) is list: #if multitask
        dataset = TensorDataset(torch.from_numpy(X[test]), torch.from_numpy(y[0][test]), torch.from_numpy(y[1][test]), torch.from_numpy(test.reshape(-1,1)))
    else:
        dataset = TensorDataset(torch.from_numpy(X[test]), torch.from_numpy(y[test]), torch.from_numpy(test.reshape(-1,1)))
        
    dataloader = DataLoader(dataset, batch_size=conf.batchSize)
    
    if conf.loss == 'binary_crossentropy':
        lossFun = nn.BCELoss()
    elif conf.loss == 'categorical_crossentropy':
        #lossFun = nn.NLLLoss()
        lossFun = nn.CrossEntropyLoss()
   
    if conf.has("corpusKey") and type(conf.corpusKey) is list: #if multitask
        loss, acc, wrongIndexes, y0, yb0, yp0, ypb0, y1, yb1, yp1, ypb1 = evaluateMultitask(model, lossFun, dataloader, getWrongIndexes=True, getPredictions=True)

        ypb30 = np.argsort(yp0,1)[:,-3:][:,::-1] #top 3
        ypb50 = np.argsort(yp0,1)[:,-5:][:,::-1] #top 5

        maps0 = MAPScorer().samplesScore(y0, yp0)
        mapc0 = MAPScorer().classesScore(y0, yp0)
        acc20 = accuracy_score(yb0, ypb0)
        kappa0 = cohen_kappa_score(yb0, ypb0)

        total0=0
        correct30=0
        correct50=0
        for i in range(len(yb0)):
            total0 += 1
            if yb0[i] in ypb30[i]:
                correct30 += 1
            if yb0[i] in ypb50[i]:
                correct50 += 1

        accT30 = correct30/total0
        accT50 = correct50/total0
 
        ypb31 = np.argsort(yp1,1)[:,-3:][:,::-1] #top 3
        ypb51 = np.argsort(yp1,1)[:,-5:][:,::-1] #top 5

        maps1 = MAPScorer().samplesScore(y1, yp1)
        mapc1 = MAPScorer().classesScore(y1, yp1)
        acc21 = accuracy_score(yb1, ypb1)
        kappa1 = cohen_kappa_score(yb1, ypb1)

        total1=0
        correct31=0
        correct51=0
        for i in range(len(yb1)):
            total1 += 1
            if yb1[i] in ypb31[i]:
                correct31 += 1
            if yb1[i] in ypb51[i]:
                correct51 += 1

        accT31 = correct31/total1
        accT51 = correct51/total1
       
        if verbose:
            print("test_loss: {:0.4f} - test_acc: {:0.4f}".format(loss, acc), flush=True)
            print("=== Site", flush=True)
            print("MAPs: {:0.4f} - MAPc: {:0.4f}".format(maps0, mapc0), flush=True)
            print("acc: {:0.4f} - kappa: {:0.4f}".format(acc20, kappa0), flush=True)
            print("accT3: {:0.4f} - accT5: {:0.4f}".format(accT30, accT50), flush=True)
            print("=== Morpho", flush=True)
            print("MAPs: {:0.4f} - MAPc: {:0.4f}".format(maps1, mapc1), flush=True)
            print("acc: {:0.4f} - kappa: {:0.4f}".format(acc21, kappa1), flush=True)
            print("accT3: {:0.4f} - accT5: {:0.4f}".format(accT31, accT51), flush=True)


    else:
        loss, acc, wrongIndexes, y, yb, yp, ypb = evaluate(model, lossFun, dataloader, getWrongIndexes=True, getPredictions=True)
        

        ypb3 = np.argsort(yp,1)[:,-3:][:,::-1] #top 3
        ypb5 = np.argsort(yp,1)[:,-5:][:,::-1] #top 5

        maps = MAPScorer().samplesScore(y, yp)
        mapc = MAPScorer().classesScore(y, yp)
        acc2 = accuracy_score(yb, ypb)
        kappa = cohen_kappa_score(yb, ypb)
        f1m = f1_score(yb, ypb, average="micro")
        f1M = f1_score(yb, ypb, average="macro")
        f1A = f1_score(yb, ypb, average=None)

        total=0
        correct3=0
        correct5=0
        for i in range(len(yb)):
            total += 1
            if yb[i] in ypb3[i]:
                correct3 += 1
            if yb[i] in ypb5[i]:
                correct5 += 1

        accT3 = correct3/total
        accT5 = correct5/total
       
        if verbose:
            print("test_loss: {:0.4f} - test_acc: {:0.4f}".format(loss, acc), flush=True)
            print("MAPs: {:0.4f} - MAPc: {:0.4f}".format(maps, mapc), flush=True)
            print("acc: {:0.4f} - kappa: {:0.4f}".format(acc2, kappa), flush=True)
            print("accT3: {:0.4f} - accT5: {:0.4f}".format(accT3, accT5), flush=True)
            print("f1micro: {:0.4f} - f1macro: {:0.4f}".format(f1m, f1M), flush=True)

    return wrongIndexes, acc2, accT3, accT5, kappa, f1m, f1M, f1A, y, yp

def getAttention(conf, model, X, y, test):
    device = next(model.parameters()).device
    dataset = TensorDataset(torch.from_numpy(X[test]), torch.from_numpy(y[test]), torch.from_numpy(test.reshape(-1,1)))
    dataloader = DataLoader(dataset, batch_size=conf.batchSize)
    attentionsDoc = []
    attentionsField = []
    attentionsPhrase = []
    predictions = []
    with torch.no_grad():
        for data in dataloader:
            x,y,i = data
            x,y = x.to(device), y.to(device)

            res = model(x, getAttention=True)
            
            yp = res[0]
            for currYp in np.array(yp.cpu()):
                predictions.append(currYp)

            if len(res) > 2:
                attentionDoc, attentionField, attentionPhrase = res[1:]

                for currAtt in np.array(attentionDoc.cpu()):
                    attentionsDoc.append(currAtt)
                for currAtt in np.array(attentionField.cpu()):
                    attentionsField.append(currAtt)
                for currAtt in np.array(attentionPhrase.cpu()):
                    attentionsPhrase.append(currAtt)

            else:
                attentionPhrase = res[1]
                
                for currAtt in np.array(attentionPhrase.cpu()):
                    attentionsPhrase.append(currAtt)

                
            #attentions.append(np.array(attention.cpu()))

    if len(attentionsField) > 0:
        return np.array(predictions), np.array(attentionsDoc), np.array(attentionsField), np.array(attentionsPhrase)
    else:
        return np.array(predictions), np.array(attentionsPhrase)

        
def saveModel(conf, model, optim):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optim.state_dict(), 
        }, conf.modelFile)

def loadModel(conf, device):
    if conf.modelSave == "best":
        fileToLoad = conf.modelFile
    else:
        fileToLoad = conf.modelFileLoad

    print("Loading {}".format(fileToLoad), flush=True)

    model, optim = makeModel(conf, device)

    checkpoint = torch.load(fileToLoad)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optim_state_dict'])
    return model,optim

def getLearningCurves(conf, metric='accuracy'):
    import tensorflow as tf
    
    sets = ['train', 'valid', 'test']

    path = {}
    for s in sets:
        path[s] = os.path.join(conf.tensorboard, metric, s)

    logFiles = {}
    for s in sets:
        logFiles[s] = list(map(lambda f: os.path.join(path[s],f), sorted(os.listdir(path[s]))))

    values = {}
    for s in sets:
        values[s] = []
        for f in logFiles[s]:
            for e in tf.compat.v1.train.summary_iterator(f):
                for v in e.summary.value:
                    if v.tag == 'loss' or v.tag == 'accuracy':
                        values[s].append(v.simple_value)

    return values

def getMaxValidEpoch(conf):
    values = getLearningCurves(conf)
    return np.argmax(values['valid'])

def getMaxValid(conf):
    try:
        values = getLearningCurves(conf)
        return values['valid'][np.argmax(values['valid'])]
    except (FileNotFoundError, ValueError):
        return -1

def getMaxTest(conf):
    try:
        values = getLearningCurves(conf)
        return values['test'][np.argmax(values['valid'])]
    except (FileNotFoundError, ValueError):
        return -1

def alreadyLaunched(conf):
    return os.path.isdir(conf.tensorboard)

def main(conf):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("======= CREATE MODEL")
    model,optim = makeModel(conf, device)
    print("======= LOAD DATA")
    X, y, train, valid, test = loadData(conf)
    print("======= TRAIN MODEL")
    runTrain(conf, model, optim, X, y, train, valid) 
    print("======= TEST MODEL")
    runTest(conf, model, X, y, test)
    print("======= SAVE MODEL")
    saveModel(conf, model, optim)


if __name__== "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("======= CREATE MODEL")
    model,optim = makeModel(config, device)
    print("======= LOAD DATA")
    X, y, train, valid, test = loadData(config)
    print("======= TRAIN MODEL")
    runTrain(config, model, optim, X, y, train, valid) 
    print("======= TEST MODEL")
    runTest(config, model, X, y, test)
    print("======= SAVE MODEL")
    saveModel(conf, model, optim)


