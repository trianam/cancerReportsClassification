import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

fileWordsIndex = "corpusLSTM_ICDO3/wordsIndex.p"
fileText = "corpusLSTM_ICDO3/text.txt"

with open(fileText) as fid:
    text = fid.readlines()

#vectorizer = TfidfVectorizer(min_df=3, max_df=0.3, strip_accents='unicode', ngram_range=(1,2))
vectorizer = TfidfVectorizer(min_df=3, max_df=0.3, strip_accents='unicode', ngram_range=(1,1))
vectorizer.fit(text)

wordsIndex = {}
for i,word in enumerate(vectorizer.get_feature_names()):
    wordsIndex[word] = i+1 #0 reserved for empty
 
pickle.dump(wordsIndex, open(fileWordsIndex, "wb"))

