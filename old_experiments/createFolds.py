import os
import re
import numpy as np
import pickle
import collections
from sklearn.model_selection import StratifiedKFold

tasks = ['sede1', 'sede12', 'morfo1', 'morfo2']
    
corpusFolder = "./corpusLSTM_ICDO3-bis"
#foldsFolder = corpusFolder+"/foldsS"
foldsFolder = corpusFolder+"/folds10"
#foldsFolder = corpusFolder+"/foldsA"

#textFile = corpusFolder+"/textS1000.txt"
#fileSedeClean = corpusFolder+"/sedeCleanS1000.txt"
#fileMorfoClean = corpusFolder+"/morfoCleanS1000.txt"
textFile = corpusFolder+"/text.txt"
fileSedeClean = corpusFolder+"/sedeClean.txt"
fileMorfoClean = corpusFolder+"/morfoClean.txt"
fileOperatoreIns = corpusFolder+"/operatoreIns.txt"
fileOperatoreAgg = corpusFolder+"/operatoreAgg.txt"
fileIndexes = foldsFolder+"/indexes.p"

with open(textFile) as fid:
    text = fid.readlines()

with open(fileSedeClean) as fid:
    sedeClean = fid.readlines()
    
with open(fileMorfoClean) as fid:
    morfoClean = fid.readlines()

with open(fileOperatoreIns) as fid:
    operatoreIns = fid.readlines()

with open(fileOperatoreAgg) as fid:
    operatoreAgg = fid.readlines()

y = {}

y['sede1'] = np.zeros(shape=(len(sedeClean)), dtype=np.int)
y['sede12'] = np.zeros(shape=(len(sedeClean)), dtype=np.int)
for i,c in enumerate(sedeClean):
    y['sede1'][i], temp = c.split()
    y['sede12'][i] = y['sede1'][i] * 10 + int(temp)

y['morfo1'] = np.zeros(shape=(len(morfoClean)), dtype=np.int)
y['morfo2'] = np.zeros(shape=(len(morfoClean)), dtype=np.int)
for i,c in enumerate(morfoClean):
    y['morfo1'][i], y['morfo2'][i] = c.split()

#opIns = np.zeros(shape=(len(operatoreIns)), dtype=np.str)
#for i,c in enumerate(operatoreIns):
#    opIns[i] = c

#opAgg = np.zeros(shape=(len(operatoreAgg)), dtype=np.str)
#for i,c in enumerate(operatoreAgg):
#    opAgg[i] = c
    
opIns = []
for c in operatoreIns:
    opIns.append(re.sub('\n','',c))

opAgg = []
for c in operatoreAgg:
    opAgg.append(re.sub('\n','',c))

skf = StratifiedKFold(n_splits=10, random_state=42)
#indexes = {'train':{}, 'test':{}}

for task in tasks:
    print("Task {}".format(task))
    #indexes['train'][task] = {}
    #indexes['test'][task] = {}

    currText = []
    currY = []
    currOpIns = []
    currOpAgg = []
    col = collections.Counter(y[task])
    for i, c in enumerate(y[task]):
        if col[c] >= 10:
        #if True:
            currText.append(text[i])
            currY.append(y[task][i])
            currOpIns.append(opIns[i])
            currOpAgg.append(opAgg[i])

    for f, (train, test) in enumerate(skf.split(np.zeros(len(currY)), currY)):
        print("    fold {}".format(f))
        
        #indexes['train'][task][f] = train
        #indexes['test'][task][f] = test

        currFolder = foldsFolder+"/"+task+"/"+str(f)

        textFileTrain = currFolder+"/textTrain.txt"
        fileYTrain = currFolder+"/yTrain.txt"
        textFileTest = currFolder+"/textTest.txt"
        fileYTest = currFolder+"/yTest.txt"
        fileOpInsTrain = currFolder+"/opInsTrain.txt"
        fileOpAggTrain = currFolder+"/opAggTrain.txt"
        fileOpInsTest = currFolder+"/opInsTest.txt"
        fileOpAggTest = currFolder+"/opAggTest.txt"

        if not os.path.exists(currFolder):
            os.makedirs(currFolder)

        with open(textFileTrain, 'w') as fidText:
            with open(fileYTrain, 'w') as fidY:
                with open(fileOpInsTrain, 'w') as fidOpIns:
                    with open(fileOpAggTrain, 'w') as fidOpAgg:
                        for i in train:
                            fidText.write("{}".format(currText[i]))
                            if currText[i][-1] != '\n':
                                fidText.write("\n")
                            fidY.write("{}\n".format(currY[i]))
                            fidOpIns.write("{}\n".format(currOpIns[i]))
                            fidOpAgg.write("{}\n".format(currOpAgg[i]))
        with open(textFileTest, 'w') as fidText:
            with open(fileYTest, 'w') as fidY:
                with open(fileOpInsTest, 'w') as fidOpIns:
                    with open(fileOpAggTest, 'w') as fidOpAgg:
                        for i in test:
                            fidText.write("{}".format(currText[i]))
                            if currText[i][-1] != '\n':
                                fidText.write("\n")
                            fidY.write("{}\n".format(currY[i]))
                            fidOpIns.write("{}\n".format(currOpIns[i]))
                            fidOpAgg.write("{}\n".format(currOpAgg[i]))

#pickle.dump(indexes, open(fileIndexes, "wb"))
