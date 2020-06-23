#! /usr/bin/env python

import numpy as np
import pandas as pd
import csv
import nltk
import nltk.data
import nltk.tokenize.stanford
import random
import re
import pickle

class CorpusProcesser:
    #encoding = 'iso-8859-1' or encoding = 'cp1252' or encoding = 'latin1'
    _encoding = 'iso-8859-1'
    #_tokenPattern = r'(?u)\b\w*[a-zA-Z_][a-zA-Z_]+\w*\b'

    def __init__(self, eot="#eot#", eor="#eor#", repEot=5, repEor=10, num="#num#", sur="#sur#", vol="#vol#", div="#div#", date="#date#"):
        #self._tokenizer = nltk.tokenize.RegexpTokenizer(r'\. |, |: |; | |\'|\.$', gaps=True)
        self._tokenizer = nltk.tokenize.stanford.StanfordTokenizer("/home/trianam/stanford-postagger-full-2016-10-31/stanford-postagger.jar", options={'ptb3Escaping':False, 'normalizeFractions':False, 'normalizeParentheses':False, 'normalizeOtherBrackets':False})
        self._tokenizerSplitPhrases = nltk.data.load('tokenizers/punkt/italian.pickle')

        self._eot = ("tttextenddd", eot)
        self._eor = ("rrrecordenddd", eor)
        self._repEot = repEot
        self._repEor = repEor
        self._num = num
        self._sur = sur
        self._vol = vol
        self._div = div
        self._date = date

    def importCsv(self, fileIsto):
        self._df = pd.read_csv(fileIsto, header = 0, encoding=self._encoding, quoting=csv.QUOTE_NONNUMERIC, low_memory=False, usecols = ['notizie', 'diagnosi', 'macroscopia', 'id_neopl'], dtype={'id_neopl':pd.np.float64})

        nullToEmpty = lambda s: "" if pd.isnull(s) else str(s)
        self._df['notizie'] = self._df.notizie.map(nullToEmpty)
        self._df['diagnosi'] = self._df.diagnosi.map(nullToEmpty)
        self._df['macroscopia'] = self._df.macroscopia.map(nullToEmpty)

        self._df['text'] = self._df[['notizie' ,'diagnosi', 'macroscopia']].apply(lambda t: (" "+self._eot[0]+" ").join(t).lower(), axis=1)

    def removeMergeable(self, fileNeop):
        dfNeop = pd.read_csv(fileNeop, header = 0, encoding=self._encoding, quoting=csv.QUOTE_NONNUMERIC, low_memory=False, usecols = ['id_neopl'], dtype={'id_neopl':pd.np.float64})
        availableIds = np.unique(dfNeop.id_neopl)
        #self._df = self._df.drop(self._df[self._df.id_neopl in availableIds].index)
        #self._df = self._df[list(map(lambda x: x not in availableIds, self._df.id_neopl))]
        keeps = list(map(lambda x: x not in availableIds, self._df.id_neopl))
        self._df = self._df[keeps]

    def importCsvMerge(self, fileIsto, fileNeop):
        dfIsto = pd.read_csv(fileIsto, header = 0, encoding=self._encoding, quoting=csv.QUOTE_NONNUMERIC, low_memory=False, usecols = ['notizie', 'diagnosi', 'macroscopia', 'ICDO1_T', 'ICDO1_M', 'ICDO3_T', 'ICDO3_M', 'ICD_IX', 'id_neopl'], dtype={'id_neopl':pd.np.float64, 'ICDO1_T':pd.np.str, 'ICDO1_M':pd.np.str, 'ICDO3_T':pd.np.str, 'ICDO3_M':pd.np.str, 'ICD_IX':pd.np.str})#, converters={'ICDO1_T':str, 'ICDO1_M':str})
        dfNeop = pd.read_csv(fileNeop, header = 0, encoding=self._encoding, quoting=csv.QUOTE_NONNUMERIC, low_memory=False, usecols = ['id_neopl', 'sede', 'morfologia', 'sede_icdo3', 'morfologia_icdo3', 'ICD_IX', 'inserimento_operatore_neoplasi', 'aggiornamento_operatore'], dtype={'id_neopl':pd.np.float64, 'sede':pd.np.str, 'morfologia':pd.np.str, 'sede_icdo3':pd.np.str, 'morfologia_icdo3':pd.np.str, 'ICD_IX':pd.np.str, 'inserimento_operatore_neoplasi':pd.np.str, 'aggiornamento_operatore':pd.np.str})

        self._df = pd.merge(dfIsto, dfNeop, on='id_neopl')

        nullToEmpty = lambda s: "" if pd.isnull(s) else str(s)
        self._df['notizie'] = self._df.notizie.map(nullToEmpty)
        self._df['diagnosi'] = self._df.diagnosi.map(nullToEmpty)
        self._df['macroscopia'] = self._df.macroscopia.map(nullToEmpty)

        self._df['text'] = self._df[['notizie' ,'diagnosi', 'macroscopia']].apply(lambda t: (" "+self._eot[0]+" ").join(t).lower(), axis=1)
        self._df['textNotizie'] = self._df.notizie.map(str.lower)
        self._df['textDiagnosi'] = self._df.diagnosi.map(str.lower)
        self._df['textMacroscopia'] = self._df.macroscopia.map(str.lower)
        
        #nullToEmptyStrip = lambda s: nullToEmpty(s).strip().upper()
        nullToEmptyStrip = lambda s: str(s).strip().upper()
        self._df['sedeICDO1'] = self._df.sede.map(nullToEmptyStrip)
        self._df['morfoICDO1'] = self._df.morfologia.map(nullToEmptyStrip)
        self._df['sedeICDO3'] = self._df.sede_icdo3.map(nullToEmptyStrip)
        self._df['morfoICDO3'] = self._df.morfologia_icdo3.map(nullToEmptyStrip)
        self._df['operatoreIns'] = self._df.inserimento_operatore_neoplasi.map(nullToEmptyStrip)
        self._df['operatoreAgg'] = self._df.aggiornamento_operatore.map(nullToEmptyStrip)

 
    def importCsvMergeAllFields(self, fileIsto, fileNeop):
        dfIsto = pd.read_csv(fileIsto, header = 0, encoding=self._encoding, quoting=csv.QUOTE_NONNUMERIC, low_memory=False, usecols = ['notizie', 'diagnosi', 'macroscopia', 'ICDO1_T', 'ICDO1_M', 'ICDO3_T', 'ICDO3_M', 'ICD_IX', 'id_neopl'], dtype={'id_neopl':pd.np.float64, 'ICDO1_T':pd.np.str, 'ICDO1_M':pd.np.str, 'ICDO3_T':pd.np.str, 'ICDO3_M':pd.np.str, 'ICD_IX':pd.np.str})#, converters={'ICDO1_T':str, 'ICDO1_M':str})
        dfNeop = pd.read_csv(fileNeop, header = 0, encoding=self._encoding, quoting=csv.QUOTE_NONNUMERIC, low_memory=False, dtype={'id_neopl':pd.np.float64, 'sede':pd.np.str, 'morfologia':pd.np.str, 'sede_icdo3':pd.np.str, 'morfologia_icdo3':pd.np.str, 'ICD_IX':pd.np.str})

        self._df = pd.merge(dfIsto, dfNeop, on='id_neopl')

        nullToEmpty = lambda s: "" if pd.isnull(s) else str(s)
        self._df['notizie'] = self._df.notizie.map(nullToEmpty)
        self._df['diagnosi'] = self._df.diagnosi.map(nullToEmpty)
        self._df['macroscopia'] = self._df.macroscopia.map(nullToEmpty)

        self._df['text'] = self._df[['notizie' ,'diagnosi', 'macroscopia']].apply(lambda t: (" "+self._eot[0]+" ").join(t).lower(), axis=1)
        
        #nullToEmptyStrip = lambda s: nullToEmpty(s).strip().upper()
        nullToEmptyStrip = lambda s: str(s).strip().upper()
        self._df['sedeICDO1'] = self._df.sede.map(nullToEmptyStrip)
        self._df['morfoICDO1'] = self._df.morfologia.map(nullToEmptyStrip)
        self._df['sedeICDO3'] = self._df.sede_icdo3.map(nullToEmptyStrip)
        self._df['morfoICDO3'] = self._df.morfologia_icdo3.map(nullToEmptyStrip)

    def importCsvIds(self, fileIsto):
        self._dfIds = pd.read_csv(fileIsto, header=0, encoding=self._encoding, quoting=csv.QUOTE_NONNUMERIC,
                             low_memory=False,
                             usecols=['id_tot', 'id_fonte', 'id_neopl'],
                             dtype={'id_tot': pd.np.float64, 'id_fonte': pd.np.float64, 'id_neopl': pd.np.float64})

    def writeIds(self, filename):
        self._dfIds.to_csv(filename, float_format='%.0f', index=False)

    def writeId_fonte(self, filename):
        self._dfIds.to_csv(filename, float_format='%.0f', index=False, columns=['id_fonte'])


    def dropNaICDO1(self):
        for i,r in self._df.iterrows():
            if len(r['sedeICDO1']) != 4:
                self._df.set_value(i, 'sede', np.NaN)
        self._df = self._df.dropna(subset = ['sede', 'morfologia'])

    def dropNaICDO3(self):
        self._df = self._df.dropna(subset = ['sede_icdo3', 'morfologia_icdo3'])

    def dropNa(self):
        self._df = self._df.dropna(subset = ['sede', 'morfologia', 'sede_icdo3', 'morfologia_icdo3'])

    def cutDataset(self, numSamples):
        self._df = self._df.ix[random.sample(list(self._df.index), numSamples)]
 
    def createText(self):
        self._text = " ".join((" "+self._eor[0]+" ").join(self._df['text']).split())

    def createTextSeparated(self):
        self._textNotizie = " ".join((" "+self._eor[0]+" ").join(self._df['textNotizie']).split())
        self._textDiagnosi = " ".join((" "+self._eor[0]+" ").join(self._df['textDiagnosi']).split())
        self._textMacroscopia = " ".join((" "+self._eor[0]+" ").join(self._df['textMacroscopia']).split())

    def createCorpusMMIOld(self):
        cleanSede3 = lambda s: s[1:3]+" "+s[3]
        cleanMorfo3 = lambda s: s[0:4]+" "+s[4]
        
        self._corpusMMI = {
            'text': [],
            'sedeICDO3': [],
            'morfoICDO3': [],
        }

        totRows = self._df.shape[0]
        for index, row in self._df.iterrows():
            if index%10 == 0:
                print("processed {}/{}        ".format(index, totRows), end="\r", flush=True)

            currRecord = []
            for field in ['textNotizie', 'textDiagnosi', 'textMacroscopia']:
                phrases = self._tokenizerSplitPhrases.tokenize(row[field])
                phrasesTok = []
                for phrase in phrases:
                    phrasesTok.append(
                        self._substitutePost(self._tokenize(self._substitutePre(phrase))))
                currRecord.append(phrasesTok)
            
            self._corpusMMI['text'].append(currRecord)
            
            self._corpusMMI['sedeICDO3'].append(cleanSede3(row['sedeICDO3']))
            self._corpusMMI['morfoICDO3'].append(cleanMorfo3(row['morfoICDO3']))


        print("processed {}/{}        ".format(index, totRows), flush=True)


    def createCorpusMMI(self, filters={
                    'active':{
                        'sede1':False,
                        'sede2':False,
                        'morfo1':False,
                        'morfo2':False,
                        },
                    'sede1':[],
                    'sede2':[],
                    'morfo1':[],
                    'morfo2':[]}):
        cleanSede3 = lambda s: s[1:3]+" "+s[3]
        cleanMorfo3 = lambda s: s[0:4]+" "+s[4]
        
        self._corpusMMI = {
            'text': [],
            'sedeICDO3': [],
            'morfoICDO3': [],
        }

        totRows = self._df.shape[0]
        for index, row in self._df.iterrows():
            if index%10 == 0:
                print("processed {}/{}        ".format(index, totRows), end="\r", flush=True)

            sede = cleanSede3(row['sedeICDO3'])
            morfo = cleanMorfo3(row['morfoICDO3'])
            currValues = {}
            currValues['sede1'], currValues['sede2'] = map(int, sede.split())
            currValues['morfo1'], currValues['morfo2'] = map(int, morfo.split())

            includeRecord = True
            for k in ['sede1', 'sede2', 'morfo1', 'morfo2']:
                if filters['active'][k] and not currValues[k] in filters[k]:
                    includeRecord = False
                    break

            if includeRecord:
                currRecord = []
                for field in ['textNotizie', 'textDiagnosi', 'textMacroscopia']:
                    phrases = self._tokenizerSplitPhrases.tokenize(row[field])
                    phrasesTok = []
                    for phrase in phrases:
                        phrasesTok.append(
                            self._substitutePost(self._tokenize(self._substitutePre(phrase))))
                    currRecord.append(phrasesTok)
                
                self._corpusMMI['text'].append(currRecord)
                
                self._corpusMMI['sedeICDO3'].append(sede)
                self._corpusMMI['morfoICDO3'].append(morfo)


        print("processed {}/{}        ".format(index, totRows), flush=True)

    def removeDuplicatesMMI(self, keepFirst=False):
        text = self._corpusMMI['text']
        duplicates = {}
        for i in range(len(text)):
            if i%10 == 0:
                print("{}               ".format(i), end="\r", flush=True)
            for j in range(i+1,len(text)):
                if text[i] == text[j]:
                    if text[i] not in duplicates.keys():
                        duplicates[text[i]] = set()
                    duplicates[text[i]].add(i)
                    duplicates[text[i]].add(j)

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
        
        newCorpusMMI = {
            'text': [],
            'sedeICDO3': [],
            'morfoICDO3': [],
        }

        for i in range(len(text)):
            if not i in indicesToRemove:
                newCorpusMMI['text'].append(text[i])
                newCorpusMMI['sedeICDO3'].append(self._corpusMMI['sedeICDO3'])
                newCorpusMMI['morfoICDO3'].append(self._corpusMMI['morfoICDO3'])

        self._corpusMMI = newCorpusMMI

    def writeCorpusMMI(self, filename):
        pickle.dump(self._corpusMMI, open(filename, 'wb'))
    
    def createCodesICDO1(self):
        self._sedeICDO1 = "\n".join(self._df['sedeICDO1'])
        self._morfoICDO1 = "\n".join(self._df['morfoICDO1'])

        cleanSede1 = lambda s: s[0:3]+" "+s[3]
        cleanMorfo1 = lambda s: s[0:4]+" "+s[4]
        self._sedeICDO1clean = "\n".join(self._df['sedeICDO1'].map(cleanSede1))
        self._morfoICDO1clean = "\n".join(self._df['morfoICDO1'].map(cleanMorfo1))

    def createOperatore(self):
        self._operatoreIns = "\n".join(self._df['operatoreIns'])
        self._operatoreAgg = "\n".join(self._df['operatoreAgg'])

    def createCodesICDO3(self):
        self._sedeICDO3 = "\n".join(self._df['sedeICDO3'])
        self._morfoICDO3 = "\n".join(self._df['morfoICDO3'])

        cleanSede3 = lambda s: s[1:3]+" "+s[3]
        cleanMorfo3 = lambda s: s[0:4]+" "+s[4]
        self._sedeICDO3clean = "\n".join(self._df['sedeICDO3'].map(cleanSede3))
        self._morfoICDO3clean = "\n".join(self._df['morfoICDO3'].map(cleanMorfo3))

    def _substitutePre(self, text):
        text = re.sub('(\\\\|/|-|_|\<|\>)', ' \\1 ', text)
        text = re.sub('([0-9])(cm|mm)', '\\1 \\2', text)
        text = re.sub('(cm|mm)\.([0-9])', '\\1. \\2', text)
        text = re.sub('(cm|mm)([0-9])', '\\1 \\2', text)
        text = re.sub("([0-9])(x)([0-9])","\\1 \\2 \\3", text)
        text = re.sub("([0-9])(x)([0-9])","\\1 \\2 \\3", text)
        text = re.sub(self._eot[0]+"( "+self._eot[0]+")+", self._eot[0], text)
        #text = re.sub(self._eor[0]+"( "+self._eor[0]+")+", self._eor[0], text)
        text = re.sub(self._eor[0]+" "+self._eot[0], self._eor[0], text)
        text = re.sub(self._eot[0]+" "+self._eor[0], self._eor[0], text)
        return text

    def _tokenize(self, text):
        text = ' '.join(self._tokenizer.tokenize(text))
        return text

    def _substitutePost(self, text):
        text = re.sub(' (\. *\. *\. *(\. *)+|_ *_ *_ *(_ *)+|- *- *- *(- *)+|= *= *= *(= *)+|\* *\* *\* *(\* *)+|\+ *\+ *\+ *(\+ *)+) ', " "+self._div+" ", text)
        text = re.sub(' [0-9][0-9]? *(,|\.|-|\\\\|/) *[0-9][0-9]? *(,|\.|-|\\\\|/) *[0-9][0-9][0-9]?[0-9]? ', " "+self._date+" ", text)
        text = re.sub(' [0-9]+(,|\.)[0-9]+ ', " "+self._num+" ", text)
        text = re.sub(' [0-9]+ ', " "+self._num+" ", text)
        text = re.sub(' '+self._num+' x '+self._num+' x '+self._num+' ', " "+self._vol+" ", text)
        text = re.sub(' '+self._num+' x '+self._num+' ', " "+self._sur+" ", text)
        text = text.replace(self._eot[0], " "+(self._eot[1]+" ")*self._repEot)
        text = text.replace(self._eor[0], " "+(self._eor[1]+" ")*self._repEor)
        #text = re.sub("( +\n *| *\n +)", "\n", text)
        text = re.sub("  +", " ", text)
        return text
 
    def substitutePre(self):
        self._text = self._substitutePre(self._text)

    def tokenize(self):
        self._text = self._tokenize(self._text)
       
    def substitutePost(self):
        self._text = self._substitutePost(self._text)

    def substitutePreSeparated(self):
        self._textNotizie = self._substitutePre(self._textNotizie)
        self._textDiagnosi = self._substitutePre(self._textDiagnosi)
        self._textMacroscopia = self._substitutePre(self._textMacroscopia)

    def tokenizeSeparated(self):
        self._textNotizie = self._tokenize(self._textNotizie)
        self._textDiagnosi = self._tokenize(self._textDiagnosi)
        self._textMacroscopia = self._tokenize(self._textMacroscopia)
       
    def substitutePostSeparated(self):
        self._textNotizie = self._substitutePost(self._textNotizie)
        self._textDiagnosi = self._substitutePost(self._textDiagnosi)
        self._textMacroscopia = self._substitutePost(self._textMacroscopia)

    def addExtremesEot(self):
        self._text =  " "+(self._eor[1]+" ")*self._repEor + " "+(self._eot[1]+" ")*self._repEot + self._text + " "+(self._eot[1]+" ")*self._repEot + " "+(self._eor[1]+" ")*self._repEor


    def calculateFieldLength(self):
        maxWordLen = 0
        maxWordLenLineNum = 0
        avgWordLen = 0
        varWord = 0
        numWords = 0

        maxLineLen = 0
        maxLineLenNum = 0
        avgLineLen = 0
        varLine = 0
        numLines = 0
        for l in self._text.split(self._eor[1]):
            numLines += 1
            currLen = len(l.split())
            if currLen > maxLineLen:
                maxLineLen = currLen
                maxLineLenNum = numLines
            avgLineLen += currLen
            varLine += currLen * currLen

            for w in l.split():
                numWords += 1
                currLen = len(w)
                if currLen > maxWordLen:
                    maxWordLen = currLen
                    maxWordLenLineNum = numLines
                avgWordLen += currLen
                varWord += currLen * currLen

        avgLineLen /= numLines
        varLine = (varLine/numLines) - (avgLineLen*avgLineLen)

        avgWordLen /= numWords
        varWord = (varWord/numWords) - (avgWordLen*avgWordLen)

        print("Max field length (line#): {} ({})".format(maxLineLen, maxLineLenNum))
        print("Average field length (var): {} ({})".format(avgLineLen, varLine))

        print("Max word length (line#): {} ({})".format(maxWordLen, maxWordLenLineNum))
        print("Average word length (var): {} ({})".format(avgWordLen, varWord))
    
    def removeEmptyLines(self):
        self._text = '\n'.join([line for line in self._text.split('\n') if line.strip() != ''])

    def writeText(self, filename):
        with open(filename, 'w') as fid:
            fid.write(self._text)

    def writeTextSeparated(self, filenameNotizie, filenameDiagnosi, filenameMacroscopia):
        for filename, text in [(filenameNotizie, self._textNotizie), (filenameDiagnosi, self._textDiagnosi), (filenameMacroscopia, self._textMacroscopia)]:
            with open(filename, 'w') as fid:
                fid.write(text)

    def writeCodesICDO1(self, filenameSedeICDO1, filenameMorfoICDO1, filenameSedeICDO1clean, filenameMorfoICDO1clean):
        filesTexts = [
            (filenameSedeICDO1, self._sedeICDO1),
            (filenameMorfoICDO1, self._morfoICDO1),
            (filenameSedeICDO1clean, self._sedeICDO1clean),
            (filenameMorfoICDO1clean, self._morfoICDO1clean)
            ]

        self._writeCodes(filesTexts)

    def writeCodesICDO3(self, filenameSedeICDO3, filenameMorfoICDO3, filenameSedeICDO3clean, filenameMorfoICDO3clean):
        filesTexts = [
            (filenameSedeICDO3, self._sedeICDO3),
            (filenameMorfoICDO3, self._morfoICDO3),
            (filenameSedeICDO3clean, self._sedeICDO3clean),
            (filenameMorfoICDO3clean, self._morfoICDO3clean)
            ]

        self._writeCodes(filesTexts)

    def writeOperatore(self, filenameOperatoreIns, filenameOperatoreAgg):
        filesTexts = [
            (filenameOperatoreIns, self._operatoreIns),
            (filenameOperatoreAgg, self._operatoreAgg),
            ]

        self._writeCodes(filesTexts)

    def _writeCodes(self, filesTexts):
        for filename, text in filesTexts:
            with open(filename, 'w') as fid:
                fid.write(text)

