#! /usr/bin/env python

import pickle

outDir = "./corpusMMI"
fileCorpus = outDir+"/corpus.p"
fileCorpusClean = outDir+"/corpusClean.p"

corpus = pickle.load(open(fileCorpus, 'rb'))

for k in corpus.keys():
    del corpus[k][51946] #row with mistaken morfo code

pickle.dump(corpus, open(fileCorpusClean, 'wb'))

