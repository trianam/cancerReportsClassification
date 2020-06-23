#! /usr/bin/env python

import corpusProcesser

outDir = "./corpusMMI"
fileIsto = "./data/ISTOLOGIE_corr.csv"
fileNeop = "./data/RTRT_NEOPLASI_corr.csv"
fileCorpus = outDir+"/corpus.p"

cc = corpusProcesser.CorpusProcesser(" ", "\n", 1, 1)

print("Import CSV")
cc.importCsvMerge(fileIsto, fileNeop)
print("Drop NA")
#cc.dropNaICDO1()
cc.dropNaICDO3()
print("Create corpus")
cc.createCorpusMMI()
print("Write corpus file")
cc.writeCorpusMMI(fileCorpus)

