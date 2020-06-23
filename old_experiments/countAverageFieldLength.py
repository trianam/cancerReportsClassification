#! /usr/bin/env python

import corpusProcesser

def main():
    fileIsto = "./data/ISTOLOGIE_corr.csv"
    fileText = "./text.txt"

    cc = corpusProcesser.CorpusProcesser(" ", "\n", 1, 1)

    print("Import CSV")
    cc.importCsv(fileIsto)
    print("Create text")
    cc.createText()
    print("Substitute pre")
    cc.substitutePre()
    print("Tokenize")
    cc.tokenize()
    print("Substitute post")
    cc.substitutePost()
    print("Calculate average field length")
    cc.calculateFieldLength()
    print("WriteText")
    cc.writeText(fileText)

main()

