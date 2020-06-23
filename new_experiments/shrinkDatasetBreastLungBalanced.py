import pickle

inputCorpusFile = "corpusTemporalBreastLung/corpusTemporal.p"
outputFile = "corpusTemporalBreastLung/corpusTemporal-1000b.p"

dim = {'train':800, 'valid':100, 'test':100}
icdo = [34,50]
icdoKey = 'sede1'

corpus = pickle.load(open(inputCorpusFile, 'rb'))

newCorpus = {d:{k:[] for k in corpus[d]} for d in corpus}
for  d in corpus:
    count = 0
    i = [len(corpus[d][icdoKey]) -1, len(corpus[d][icdoKey]) -1]
    while count < dim[d]:
        while corpus[d][icdoKey][i[0]] != icdo[0]:
            i[0] -= 1
        newCorpus[d]['text'].append(corpus[d]['text'][i[0]])
        newCorpus[d][icdoKey].append(corpus[d][icdoKey][i[0]])
        i[0] -= 1
        count += 1

        while corpus[d][icdoKey][i[1]] != icdo[1]:
            i[1] -= 1
        newCorpus[d]['text'].append(corpus[d]['text'][i[1]])
        newCorpus[d][icdoKey].append(corpus[d][icdoKey][i[1]])
        i[1] -= 1
        count += 1

    for k in corpus[d]:
        newCorpus[d][k] = list(reversed(newCorpus[d][k]))   

pickle.dump(newCorpus, open(outputFile, 'wb'))

