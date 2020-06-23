#!/usr/bin/env python
# coding: utf-8


import sys
import pandas as pd
import csv
import collections
import re
import pickle
import numpy as np
#import nltk
import edlib



fileNeop = "../data/RTRT_NEOPLASI_corr.csv"
fileIsto = "../data/ISTOLOGIE_corr.csv" 
fileIcdo = "icdo-3-dict-en.p"

fileLoadDistances = "editDistances-{}.npz"
fileSaveDistances = "editDistances.npz"

encoding = 'iso-8859-1'

icdo = pickle.load(open(fileIcdo, 'rb'))


# # Load neop

dfNeop = pd.read_csv(fileNeop, header = 0, encoding=encoding, quoting=csv.QUOTE_NONNUMERIC, low_memory=False,
                     usecols = ['id_neopl', 'sede_icdo3',
                                'morfologia_icdo3', 'inserimento_data_neoplasi'],
                     dtype={'id_neopl':pd.np.float64, 'sede_icdo3':pd.np.str, 
                            'morfologia_icdo3':pd.np.str, 'inserimento_data_neoplasi':pd.np.str},
                    #parse_dates=['inserimento_data_neoplasi'], 
                    #date_parser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
                    )


dfNeop = dfNeop.dropna('index')
dfNeop['inserimento_data_neoplasi'] =  pd.to_datetime(dfNeop['inserimento_data_neoplasi'], format='%Y-%m-%d %H:%M:%S')
dfNeop = dfNeop.sort_values(by=['inserimento_data_neoplasi'])


# remove invalid sites
dfNeop['correctSede'] = dfNeop.sede_icdo3.map(lambda s: len(s)==4 and s[0] == "C" and int(s[1:3]) in icdo.keys())
dfNeop = dfNeop[dfNeop.correctSede]


dfNeop['y'] = dfNeop.sede_icdo3.map(lambda s: int(s[1:3]))
dfNeop['yM'] = dfNeop.morfologia_icdo3.map(lambda m: int(m[:4]))

# # Load isto

dfIsto = pd.read_csv(fileIsto, header = 0, encoding=encoding, quoting=csv.QUOTE_NONNUMERIC, low_memory=False, 
                     usecols = ['notizie', 'diagnosi', 'macroscopia', 'id_neopl', 'id_fonte'],
                     dtype={'id_neopl':pd.np.float64, 'id_fonte':pd.np.float64})

nullToEmpty = lambda s: "" if pd.isnull(s) else str(s)
dfIsto['notizie'] = dfIsto.notizie.map(nullToEmpty)
dfIsto['diagnosi'] = dfIsto.diagnosi.map(nullToEmpty)
dfIsto['macroscopia'] = dfIsto.macroscopia.map(nullToEmpty)


dfIsto = dfIsto.dropna('index')


dfIsto['text'] = dfIsto[['notizie' ,'diagnosi', 'macroscopia']].apply(lambda t: " ".join(t).upper(), axis=1)
dfIsto['text'] = dfIsto['text'].astype('unicode')

dfIsto['text'] = dfIsto['text'].apply(lambda t: " ".join(re.findall(r"\w+|[^\w\s]", t, re.UNICODE)))

dfIsto

dfIsto['correctText'] = dfIsto.text.map(lambda t: len(t) > 0)

dfIsto = dfIsto[dfIsto.correctText]

dfIsto = dfIsto.drop_duplicates(['text'])


dfIsto = dfIsto.drop_duplicates(['id_neopl'], keep=False)

# # Merge

dfMerge = pd.merge(dfIsto, dfNeop, on='id_neopl', validate="one_to_one")

# # Final

dfFinal = dfMerge[['inserimento_data_neoplasi','text', 'y', 'yM']]
dfFinal = dfFinal.rename(columns={"inserimento_data_neoplasi": "data"})

dfFinal = dfFinal.sort_values(by=['data'])

# # Calculate distance


corpus = {
        'text':dfFinal['text'].values,
        'sede1':dfFinal['y'].values,
        'morfo1':dfFinal['yM'].values,
    }

dist = np.zeros((len(corpus['text']), len(corpus['text'])), dtype=np.uint16)
numDocs = len(corpus['text'])

splits = []
for s in range(10):
    splits.append(numDocs//10*s)
splits.append(numDocs)

dataset = []
testLen = int(len(dfFinal)/100*20)
datasetSplits = [len(dfFinal) - testLen - testLen, len(dfFinal) - testLen] 
for i in range(len(corpus['text'])):
    if i<datasetSplits[0]:
        dataset.append("train")
    elif i<datasetSplits[1]:
        dataset.append("valid")
    else:
        dataset.append("test")

corpus['dataset'] = np.array(dataset)



for s in range(10):
    print("=============== {}".format(fileLoadDistances.format(s)))
    currFile = np.load(fileLoadDistances.format(s))
    currDist = currFile['distance']
    for i in range(splits[s], splits[s+1]):
        for j in range(i+1, numDocs):
            if j % 1000 == 0:
                print("{}/{} - {}/{}          ".format(i, splits[s+1]-1, j, numDocs), end='\r', flush=True)
            #dist[i,j] = dist[j,i] = nltk.edit_distance(corpus['text'][i], corpus['text'][j])
            dist[i,j] = dist[j,i] = currDist[i,j]
    print("{}/{} - {}/{}        ".format(i, splits[s+1]-1, j, numDocs), flush=True)

corpus['distance'] = dist

# # write

np.savez(fileSaveDistances, text=corpus['text'], sede1=corpus['sede1'], morfo1=corpus['morfo1'], dataset=corpus['dataset'], distance=corpus['distance'])

