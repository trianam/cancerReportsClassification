import os
import numpy as np
import pickle
import collections
from sklearn.model_selection import train_test_split


tasks = ['sede1', 'sede12', 'morfo1', 'morfo2']
    
corpusFolder = "./corpusLSTM_ICDO3"
foldsFolder = corpusFolder+"/randomSplit"

textFile = corpusFolder+"/text.txt"
fileSedeClean = corpusFolder+"/sedeClean.txt"
fileMorfoClean = corpusFolder+"/morfoClean.txt"
fileIndexes = foldsFolder+"/indexes.p"

with open(textFile) as fid:
    text = fid.readlines()

with open(fileSedeClean) as fid:
    sedeClean = fid.readlines()
    
with open(fileMorfoClean) as fid:
    morfoClean = fid.readlines()

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


indexTrain, indexTest = train_test_split(np.arange(len(text)), test_size=0.2, random_state=42)

#indexes = {'train':indexTrain, 'test':indexTest}

for task in tasks:
    print("Task {}".format(task))
    #indexes['train'][task] = {}
    #indexes['test'][task] = {}

    currText = []
    currY = []
    col = collections.Counter(y[task])
    for i, c in enumerate(y[task]):
        #if col[c] >= 10:
        if True:
            currText.append(text[i])
            currY.append(y[task][i])

    #indexes['train'][task][f] = train
    #indexes['test'][task][f] = test

    currFolder = foldsFolder+"/"+task

    textFileTrain = currFolder+"/textTrain.txt"
    fileYTrain = currFolder+"/yTrain.txt"
    textFileTest = currFolder+"/textTest.txt"
    fileYTest = currFolder+"/yTest.txt"

    if not os.path.exists(currFolder):
        os.makedirs(currFolder)

    with open(textFileTrain, 'w') as fidText:
        with open(fileYTrain, 'w') as fidY:
            for i in indexTrain:
                fidText.write("{}".format(currText[i]))
                if currText[i][-1] != '\n':
                    fidText.write("\n")
                fidY.write("{}\n".format(currY[i]))
    
    with open(textFileTest, 'w') as fidText:
        with open(fileYTest, 'w') as fidY:
            for i in indexTest:
                fidText.write("{}".format(currText[i]))
                if currText[i][-1] != '\n':
                    fidText.write("\n")
                fidY.write("{}\n".format(currY[i]))

#pickle.dump(indexes, open(fileIndexes, "wb"))
