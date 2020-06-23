import os
import re
import numpy as np
import pickle
import collections
import sklearn
from sklearn.model_selection import StratifiedKFold

tasks = ['sede1', 'sede12', 'morfo1', 'morfo2']
    
corpusFolder = "./corpusMMI"
foldsFolder = corpusFolder+"/folds10"

corpusFile = corpusFolder+"/corpusClean.p"
fileIndexes = foldsFolder+"/indexes.p"

corpus = pickle.load(open(corpusFile, 'rb'))

text = corpus['text']

sedeClean = corpus['sedeICDO3']
morfoClean = corpus['morfoICDO3']

y = {}

y['sede1'] = np.zeros(shape=(len(sedeClean)), dtype=np.int)
y['sede12'] = np.zeros(shape=(len(sedeClean)), dtype=np.int)
for i,c in enumerate(sedeClean):
    y['sede1'][i], temp = c.split()
    try:
        y['sede12'][i] = y['sede1'][i] * 10 + int(temp)
    except ValueError as e:
        print(i)
        raise e

y['morfo1'] = np.zeros(shape=(len(morfoClean)), dtype=np.int)
y['morfo2'] = np.zeros(shape=(len(morfoClean)), dtype=np.int)
for i,c in enumerate(morfoClean):
    y['morfo1'][i], y['morfo2'][i] = c.split()

skf = StratifiedKFold(n_splits=10, random_state=42)
#indexes = {'train':{}, 'test':{}}

for task in tasks:
    print("Task {}".format(task))
    #indexes['train'][task] = {}
    #indexes['test'][task] = {}

    currText = []
    currY = []
    col = collections.Counter(y[task])
    for i, c in enumerate(y[task]):
        if col[c] >= 10:
        #if True:
            currText.append(text[i])
            currY.append(y[task][i])

    for f, (train, test) in enumerate(skf.split(np.zeros(len(currY)), currY)):
        print("    fold {}".format(f))
        
        #indexes['train'][task][f] = train
        #indexes['test'][task][f] = test

        currFolder = foldsFolder+"/"+task+"/"+str(f)

        corpusFileTrain = currFolder+"/corpusTrain.p"
        corpusFileTest = currFolder+"/corpusTest.p"

        if not os.path.exists(currFolder):
            os.makedirs(currFolder)

        corpusTrain = {
            'X': [],
            'y': [],
        }
        corpusTest = {
            'X': [],
            'y': [],
        }

        for i in train:
            corpusTrain['X'].append(currText[i])
            corpusTrain['y'].append(currY[i])
        for i in test:
            corpusTest['X'].append(currText[i])
            corpusTest['y'].append(currY[i])
        
        pickle.dump(corpusTrain, open(corpusFileTrain, 'wb'))
        pickle.dump(corpusTest, open(corpusFileTest, 'wb'))


#pickle.dump(indexes, open(fileIndexes, "wb"))
