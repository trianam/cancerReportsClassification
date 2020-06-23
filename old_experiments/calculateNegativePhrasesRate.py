from myLSTMbidirectionalFold5NoNegation import MyLSTM

def calculate(text, vectors, negations, phraseLen):
    numPhrases = 0
    numWords = 0
    numPhrasesNeg = 0
    numWordsNeg = 0
    
    textLen = len(text)
    for iDoc,l in enumerate(text):
        thisPhraseNeg = False
        numPhrases += 1
        
        #if iDoc%1000==0:
        #    print("         processed line {}/{}           ".format(iDoc,textLen), end='\r')

        words = l.split()

        iWordWithNeg = 0
        for iWord in range(len(words)):
            try:
                w = words[iWord]
                appo = vectors[w]
                iWordWithNeg += 1
                if iWordWithNeg >= phraseLen:
                    break
                numWords += 1
                if w in negations:
                    numWordsNeg += 1
                    if not thisPhraseNeg:
                        thisPhraseNeg = True
                        numPhrasesNeg += 1
            except IndexError:
                break
            except KeyError:
                pass

    #print("")
    return (numPhrases, numWords, numPhrasesNeg, numWordsNeg)


for fold in range(1):
    print("########################################### Fold {}".format(fold))

    totNumPhrases = 0
    totNumWords = 0
    totNumPhrasesNeg = 0
    totNumWordsNeg = 0
    
    for task in ['sede1', 'sede12', 'morfo1', 'morfo2']:
        print("   "+task)
        
        lstm = MyLSTM(fold, tasks=[task], epochs = 5, patience = 5)
        negations = lstm._negations
        phraseLen = lstm._phraseLen

        vectors = {}
        with open(lstm._fileVectors) as fid:
            for line in fid.readlines():
                sline = line.split()
                currVec = []
                for i in range(1,len(sline)):
                    currVec.append(sline[i])

                vectors[sline[0]] = currVec
                
        with open(lstm._textFileTrain[task]) as fid:
            textTrain = fid.readlines()

        with open(lstm._textFileTest[task]) as fid:
            textTest = fid.readlines()

        trainPhrases, trainWords, trainPhrasesNeg, trainWordsNeg = calculate(textTrain, vectors , lstm._negations, lstm._phraseLen)
        testPhrases, testWords, testPhrasesNeg, testWordsNeg = calculate(textTest, vectors , lstm._negations, lstm._phraseLen)

        numPhrases = trainPhrases + testPhrases
        numWords = trainWords + testWords
        numPhrasesNeg = trainPhrasesNeg + testPhrasesNeg
        numWordsNeg = trainWordsNeg + testWordsNeg

        print("         Negative Phrases Rate = {}   ({}/{})".format(numPhrasesNeg/numPhrases, numPhrasesNeg, numPhrases))
        print("         Negative Words Rate = {}   ({}/{})".format(numWordsNeg/numWords, numWordsNeg, numWords))

        totNumPhrases += numPhrases
        totNumWords += numWords
        totNumPhrasesNeg += numPhrasesNeg
        totNumWordsNeg += numWordsNeg

    print("    Negative Phrases Rate = {}   ({}/{})".format(totNumPhrasesNeg/totNumPhrases, totNumPhrasesNeg, totNumPhrases))
    print("    Negative Words Rate = {}   ({}/{})".format(totNumWordsNeg/totNumWords, totNumWordsNeg, totNumWords))
