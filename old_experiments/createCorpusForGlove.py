#! /usr/bin/env python

import corpusProcesser

def main():
    fileIsto = "./data/ISTOLOGIE_corr.csv"
    fileNeop = "./data/RTRT_NEOPLASI_corr.csv"
    fileTextRaw = "./rawText.txt"
    fileTextTok = "./tokText.txt"

    #cc = corpusProcesser.CorpusProcesser(eot="#eotr #eotr #eotr #eotr #eotr #eotr #eotr #eotr #eotr #eotr #eotr #eotr #eotr #eotr #eotr #eotr #eotr #eotr #eotr #eotr #eotl #eotl #eotl #eotl #eotl #eotl #eotl #eotl #eotl #eotl #eotl #eotl #eotl #eotl #eotl #eotl #eotl #eotl #eotl #eotl", eor="", repEot=1, repEor=0)
    #cc = corpusProcesser.CorpusProcesser(eot="", eor="", repEot=0, repEor=0)
    cc = corpusProcesser.CorpusProcesser(eot="\n", eor="\n", repEot=1, repEor=1)

    print("Import CSV")
    cc.importCsv(fileIsto)
    print("RemoveMergeable")
    cc.removeMergeable(fileNeop)
    print("Create text")
    cc.createText()
    #print("Write raw file")
    #cc.writeText(fileTextRaw)
    print("Substitute pre")
    cc.substitutePre()
    print("Tokenize")
    cc.tokenize()
    print("Substitute post")
    cc.substitutePost()
    cc.addExtremesEot()
    print("Remove empty lines")
    cc.removeEmptyLines()
    print("Write tok file")
    cc.writeText(fileTextTok)

main()

