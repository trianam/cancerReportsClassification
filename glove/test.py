import numpy as np
import sklearn as sk
import sklearn.metrics.pairwise

d = dict()
index = dict()

with open('vectors.txt') as f:
    lines = f.readlines()
    words = np.empty(len(lines), dtype=object)
    vectors = np.zeros((len(lines),50))

    i = 0
    for l in lines:
        sl = l.split()
        currWord = sl[0]
        currVec = np.array(list(map(float,sl[1:])))

        d[currWord] = currVec
        index[currWord] = i
        words[i] = currWord
        vectors[i] = currVec

        i += 1

#distances = sk.metrics.pairwise.cosine_similarity(vectors)

#neighb = np.argsort(distances[index['tessuto']])
#list(words[neighb])[0:15]
#list(distances[index['tessuto']][neighb])[0:15]

#list(words[np.argsort(sk.metrics.pairwise.euclidean_distances(vectors, (d['king']-d['man']+d['woman']).reshape(1,-1)).T[0])])[0:15]

