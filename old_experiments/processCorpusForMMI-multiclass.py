#! /usr/bin/env python

import pickle
import collections
import numpy as np
import h5py

outDir = "./corpusMMI"
fileInCorpus = outDir+"/corpusFiltered-multiclass-noRare.p"
#fileOutHdf5 = outDir+"/corpusProcessed.hdf5"
fileInValues = outDir+"/sede1-values.p"
fileOutX = outDir+"/X-multiclass.npy"
fileOutY = outDir+"/y-multiclass.npy"
fileVectors = outDir+"/vectors.txt"

numFields = 3
numPhrases = 10
phraseLen = 100

balanceKey = 'sede1'
values = pickle.load(open(fileInValues, 'rb'))


corpus = pickle.load(open(fileInCorpus, 'rb'))


vectors = {}
with open(fileVectors) as fid:
    for line in fid.readlines():
        sline = line.split()
        currVec = []
        for i in range(1,len(sline)):
            currVec.append(sline[i])

        vectors[sline[0]] = currVec
    vecLen = len(currVec)


X = np.zeros(shape=(len(corpus['text']), numFields, numPhrases, phraseLen, vecLen), dtype=np.float)
y = np.zeros(shape=(len(corpus['text']), len(values)), dtype=np.float)


documents = []
textLen = len(corpus['text'])
for iDoc,l in enumerate(corpus['text']):
    if iDoc%1000==0:
        print("         processed line {}/{}           ".format(iDoc,textLen), end='\r')
   
    y[iDoc][values.index(corpus[balanceKey][iDoc])] = 1.

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

print("         processed line {}/{}           ".format(iDoc+1,textLen))

#dataFile = h5py.File(fileOutHdf5, 'w')
#dataFile.create_dataset('X', data=X)
#dataFile.create_dataset('y', data=y)
#dataFile.close()

np.save(fileOutX, X)
np.save(fileOutY, y)

