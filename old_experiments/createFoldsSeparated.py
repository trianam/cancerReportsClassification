import os
import re
import numpy as np
import pickle
import collections
from sklearn.model_selection import StratifiedKFold

tasks = ['sede1', 'sede12', 'morfo1', 'morfo2']
    
corpusFolder = "./corpusLSTM_ICDO3-separated"
#foldsFolder = corpusFolder+"/foldsS"
foldsFolder = corpusFolder+"/folds10"
#foldsFolder = corpusFolder+"/foldsA"

#textFile = corpusFolder+"/textS1000.txt"
#fileSedeClean = corpusFolder+"/sedeCleanS1000.txt"
#fileMorfoClean = corpusFolder+"/morfoCleanS1000.txt"
textFile = corpusFolder+"/text.txt"
textNotizieFile = corpusFolder+"/textNotizie.txt"
textDiagnosiFile = corpusFolder+"/textDiagnosi.txt"
textMacroscopiaFile = corpusFolder+"/textMacroscopia.txt"
fileSedeClean = corpusFolder+"/sedeClean.txt"
fileMorfoClean = corpusFolder+"/morfoClean.txt"
fileIndexes = foldsFolder+"/indexes.p"

with open(textFile) as fid:
    text = fid.readlines()

with open(textNotizieFile) as fid:
    textNotizie = fid.readlines()

with open(textDiagnosiFile) as fid:
    textDiagnosi = fid.readlines()

with open(textMacroscopiaFile) as fid:
    textMacroscopia = fid.readlines()

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

skf = StratifiedKFold(n_splits=10, random_state=42)
#indexes = {'train':{}, 'test':{}}

for task in tasks:
    print("Task {}".format(task))
    #indexes['train'][task] = {}
    #indexes['test'][task] = {}

    currText = []
    currTextNotizie = []
    currTextDiagnosi = []
    currTextMacroscopia = []
    currY = []
    currOpIns = []
    currOpAgg = []
    col = collections.Counter(y[task])
    for i, c in enumerate(y[task]):
        if col[c] >= 10:
        #if True:
            currText.append(text[i])
            currTextNotizie.append(textNotizie[i])
            currTextDiagnosi.append(textDiagnosi[i])
            currTextMacroscopia.append(textMacroscopia[i])
            currY.append(y[task][i])

    for f, (train, test) in enumerate(skf.split(np.zeros(len(currY)), currY)):
        print("    fold {}".format(f))
        
        #indexes['train'][task][f] = train
        #indexes['test'][task][f] = test

        currFolder = foldsFolder+"/"+task+"/"+str(f)

        textFileTrain = currFolder+"/textTrain.txt"
        textNotizieFileTrain = currFolder+"/textNotizieTrain.txt"
        textDiagnosiFileTrain = currFolder+"/textDiagnosiTrain.txt"
        textMacroscopiaFileTrain = currFolder+"/textMacroscopiaTrain.txt"
        fileYTrain = currFolder+"/yTrain.txt"
        textFileTest = currFolder+"/textTest.txt"
        textNotizieFileTest = currFolder+"/textNotizieTest.txt"
        textDiagnosiFileTest = currFolder+"/textDiagnosiTest.txt"
        textMacroscopiaFileTest = currFolder+"/textMacroscopiaTest.txt"
        fileYTest = currFolder+"/yTest.txt"

        if not os.path.exists(currFolder):
            os.makedirs(currFolder)

        with open(textFileTrain, 'w') as fidText:
            with open(textNotizieFileTrain, 'w') as fidTextNotizie:
                with open(textDiagnosiFileTrain, 'w') as fidTextDiagnosi:
                    with open(textMacroscopiaFileTrain, 'w') as fidTextMacroscopia:
                        with open(fileYTrain, 'w') as fidY:
                            for i in train:
                                fidText.write("{}".format(currText[i]))
                                if currText[i][-1] != '\n':
                                    fidText.write("\n")
                                fidTextNotizie.write("{}".format(currTextNotizie[i]))
                                if currTextNotizie[i][-1] != '\n':
                                    fidTextNotizie.write("\n")
                                fidTextDiagnosi.write("{}".format(currTextDiagnosi[i]))
                                if currTextDiagnosi[i][-1] != '\n':
                                    fidTextDiagnosi.write("\n")
                                fidTextMacroscopia.write("{}".format(currTextMacroscopia[i]))
                                if currTextMacroscopia[i][-1] != '\n':
                                    fidTextMacroscopia.write("\n")
                                fidY.write("{}\n".format(currY[i]))
        with open(textFileTest, 'w') as fidText:
            with open(textNotizieFileTest, 'w') as fidTextNotizie:
                with open(textDiagnosiFileTest, 'w') as fidTextDiagnosi:
                    with open(textMacroscopiaFileTest, 'w') as fidTextMacroscopia:
                        with open(fileYTest, 'w') as fidY:
                            for i in test:
                                fidText.write("{}".format(currText[i]))
                                if currText[i][-1] != '\n':
                                    fidText.write("\n")
                                fidTextNotizie.write("{}".format(currTextNotizie[i]))
                                if currTextNotizie[i][-1] != '\n':
                                    fidTextNotizie.write("\n")
                                fidTextDiagnosi.write("{}".format(currTextDiagnosi[i]))
                                if currTextDiagnosi[i][-1] != '\n':
                                    fidTextDiagnosi.write("\n")
                                fidTextMacroscopia.write("{}".format(currTextMacroscopia[i]))
                                if currTextMacroscopia[i][-1] != '\n':
                                    fidTextMacroscopia.write("\n")
                                fidY.write("{}\n".format(currY[i]))

#pickle.dump(indexes, open(fileIndexes, "wb"))
