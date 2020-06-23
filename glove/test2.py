import numpy as np
import numpy.linalg
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

#distances = sk.metrics.pairwise.euclidean_distances(vectors)

#neighb = np.argsort(distances[index['tessuto']])
#list(words[neighb])[0:15]
#list(distances[index['tessuto']][neighb])[0:15]
#np.linalg.norm((d['adenocarcinoma']-d['epiteliale']+d['linfonodo'])- d['linfoma'])

#list(words[np.argsort(sk.metrics.pairwise.euclidean_distances(vectors, (d['linfoma']-d['linfonodo']+d['cute']).reshape(1,-1)).T[0])])[0:15]



import sys
import codecs
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import TruncatedSVD as Projector
#from sklearn.manifold import TSNE
 
subset = ['slow', 'slower', 'slowest', 'short', 'shorter', 'shortest', 'strong', 'stronger', 'strongest', 'loud', 'louder', 'loudest']

pairs = {
    'b':[
        ('slow','slower'),
        ('short', 'shorter'),
        ('strong', 'stronger'),
        ('loud', 'louder')
    ],
    'r':[
        ('slower', 'slowest'),
        ('shorter', 'shortest'),
        ('stronger', 'strongest'),
        ('louder', 'loudest')
    ]}
                                   
proj = Projector(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
Y = proj.fit_transform(vectors[[index[k] for k in subset],:])
labels = words[[index[k] for k in subset]]

plt.scatter(Y[:, 0], Y[:, 1])
for label, x, y in zip(labels, Y[:, 0], Y[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

for color in pairs:
    for pair in pairs[color]:
        plt.plot([Y[subset.index(pair[0]),0], Y[subset.index(pair[1]),0]], [Y[subset.index(pair[0]),1], Y[subset.index(pair[1]),1]], '-', color=color)

plt.show()
                                                                                                                                                        
                                                                                                                                                         
