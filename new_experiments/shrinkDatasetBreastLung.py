import pickle

inputCorpusFile = "corpusTemporalBreastLung/corpusTemporal.p"
outputFile = "corpusTemporalBreastLung/corpusTemporal-1000.p"

dim = {'train':800, 'valid':100, 'test':100}

corpus = pickle.load(open(inputCorpusFile, 'rb'))

newCorpus = {d:{k:corpus[d][k][-dim[d]:] for k in corpus[d]} for d in corpus}

pickle.dump(newCorpus, open(outputFile, 'wb'))

