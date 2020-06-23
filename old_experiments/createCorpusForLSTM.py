#! /usr/bin/env python

import corpusProcesser

#outDir = "./corpusLSTM_ICDO1"
outDir = "./corpusLSTM_ICDO3-separated"
fileIsto = "./data/ISTOLOGIE_corr.csv"
fileNeop = "./data/RTRT_NEOPLASI_corr.csv"
fileText = outDir+"/text.txt"
fileTextNotizie = outDir+"/textNotizie.txt"
fileTextDiagnosi = outDir+"/textDiagnosi.txt"
fileTextMacroscopia = outDir+"/textMacroscopia.txt"
fileSedeICDO1 = outDir+"/sedeICDO1.txt"
fileMorfoICDO1 = outDir+"/morfoICDO1.txt"
fileSedeICDO3 = outDir+"/sedeICDO3.txt"
fileMorfoICDO3 = outDir+"/morfoICDO3.txt"
fileSedeICDO1clean = outDir+"/sedeICDO1clean.txt"
fileMorfoICDO1clean = outDir+"/morfoICDO1clean.txt"
fileSedeICDO3clean = outDir+"/sedeICDO3clean.txt"
fileMorfoICDO3clean = outDir+"/morfoICDO3clean.txt"
fileOperatoreIns = outDir+"/operatoreIns.txt"
fileOperatoreAgg = outDir+"/operatoreAgg.txt"

cc = corpusProcesser.CorpusProcesser(" ", "\n", 1, 1)

print("Import CSV")
cc.importCsvMerge(fileIsto, fileNeop)
print("Drop NA")
#cc.dropNaICDO1()
cc.dropNaICDO3()
print("Create text")
cc.createText()
cc.createTextSeparated()
print("Create codes")
#cc.createCodesICDO1()
cc.createCodesICDO3()
#print("Create operatore")
#cc.createOperatore()
print("Substitute pre")
cc.substitutePre()
cc.substitutePreSeparated()
print("Tokenize")
cc.tokenize()
cc.tokenizeSeparated()
print("Substitute post")
cc.substitutePost()
cc.substitutePostSeparated()
print("Write text file")
cc.writeText(fileText)
cc.writeTextSeparated(fileTextNotizie, fileTextDiagnosi, fileTextMacroscopia)
print("Write codes files")
#cc.writeCodesICDO1(fileSedeICDO1, fileMorfoICDO1, fileSedeICDO1clean, fileMorfoICDO1clean)
cc.writeCodesICDO3(fileSedeICDO3, fileMorfoICDO3, fileSedeICDO3clean, fileMorfoICDO3clean)
#print("Write operatore")
#cc.writeOperatore(fileOperatoreIns, fileOperatoreAgg)

