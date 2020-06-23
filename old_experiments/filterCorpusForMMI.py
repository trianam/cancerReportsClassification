#! /usr/bin/env python

import pickle
import collections
import numpy as np

outDir = "./corpusMMI"
fileInCorpus = outDir+"/corpusClean.p"
fileOutCorpus = outDir+"/corpusFiltered.p"
fileOutCorpusBalanced = outDir+"/corpusFilteredBalanced.p"

keepFirst = False
filters={
    'active':{
        'sede1':True,
        'sede2':False,
        'morfo1':False,
        'morfo2':False,
    },
    'sede1':[18,20],
    'sede2':[],
    'morfo1':[],
    'morfo2':[],
}

balanceKey = 'sede1'


corpus = pickle.load(open(fileInCorpus, 'rb'))

#filter
newCorpus = {
    'text': [],
    'sedeICDO3': [],
    'morfoICDO3': [],
}

for i in range(len(corpus['text'])):
    currValues = {}
    currValues['sede1'], currValues['sede2'] = map(int, corpus['sedeICDO3'][i].split())
    currValues['morfo1'], currValues['morfo2'] = map(int, corpus['morfoICDO3'][i].split())

    includeRecord = True
    for k in ['sede1', 'sede2', 'morfo1', 'morfo2']:
        if filters['active'][k] and not currValues[k] in filters[k]:
            includeRecord = False 
            break

    if includeRecord:
        newCorpus['text'].append(corpus['text'][i])
        newCorpus['sedeICDO3'].append(corpus['sedeICDO3'][i])
        newCorpus['morfoICDO3'].append(corpus['morfoICDO3'][i])
        
corpus = newCorpus

#remove duplicates

newCorpus = {
    'text': [],
    'sedeICDO3': [],
    'morfoICDO3': [],
    'sede1': [],
    'sede2': [],
    'morfo1': [],
    'morfo2': [],
}

merge = lambda r: " ".join([ i for j in r for i in j ])
text = corpus['text'] 
duplicates = {} 
for i in range(len(text)):
    if i%10 == 0:
        print("{}               ".format(i), end="\r", flush=True)
    for j in range(i+1,len(text)):
        if text[i] == text[j]:
            k = merge(text[i])
            if k not in duplicates.keys():
                duplicates[k] = set()
            duplicates[k].add(i)
            duplicates[k].add(j)

print("{}               ".format(i), flush=True)
print("FOUND {} duplicates".format(len(duplicates)))

indicesToRemove = []
for t in duplicates:
    if keepFirst:
        for i in list(duplicates[t])[1:]:
            indicesToRemove.append(i)
    else:
        for i in duplicates[t]:
            indicesToRemove.append(i)

for i in range(len(text)):
    if not i in indicesToRemove:
        sede = corpus['sedeICDO3'][i]
        morfo = corpus['morfoICDO3'][i]
        newCorpus['text'].append(text[i])
        newCorpus['sedeICDO3'].append(sede)
        newCorpus['morfoICDO3'].append(morfo)
        newCorpus['sede1'].append(int(sede.split()[0]))
        newCorpus['sede2'].append(int(sede.split()[1]))
        newCorpus['morfo1'].append(int(morfo.split()[0]))
        newCorpus['morfo2'].append(int(morfo.split()[1]))


pickle.dump(newCorpus, open(fileOutCorpus, 'wb'))

corpus = newCorpus

p = np.random.permutation(len(corpus['text']))
text = [ corpus['text'][i] for i in p ]
y = [ corpus[balanceKey][i] for i in p ]

newCorpus = {
    'text': [],
    balanceKey: [],
}

counter = collections.Counter(y)
numRecords = counter.most_common()[-1][1]

for k in counter.keys():
    found = 0
    for i in range(len(text)):
        if y[i] == k:
            newCorpus['text'].append(text[i])
            newCorpus[balanceKey].append(k)
            found += 1
        if found == numRecords:
            break

pickle.dump(newCorpus, open(fileOutCorpusBalanced, 'wb'))

