import pickle
import numpy as np
from collections import Counter
import sklearn.metrics

models = [
        "MMI-cnn3x60-1hidd",
        "MMI-bi70cell1hidd-3",
        "LSTMbidirectional5e",
        "SVMbigrams",
        "LSTMconvolutional",
        ]
intModels = [models[0], models[1]]

predictionsRaw = {
        "MMI-cnn3x60-1hidd" : pickle.load(open("filesFolds-MMI-cnn3x60-1hidd/output/0/predictionSede1.p", 'rb')),
        "MMI-bi70cell1hidd-3" : pickle.load(open("filesFolds-MMI-bi70cell1hidd-3/output/0/predictionSede1.p", 'rb')),
        "LSTMbidirectional5e" : pickle.load(open("filesFolds-LSTMbidirectional5e/output/0/predictionSede1.p", 'rb')),
        "SVMbigrams" : pickle.load(open("filesFolds-SVMbigrams/output/0/predictionSede1.p", 'rb')),
        "LSTMconvolutional" : pickle.load(open("filesFolds-LSTMconvolutional2/output/0/predictionSede1.p", 'rb')),
        }

binarizers = {
        "MMI-cnn3x60-1hidd" : pickle.load(open("memmapFolds-MMI-cnn3x60-1hidd/0/binarizers/lbSede1.p", 'rb')),
        "MMI-bi70cell1hidd-3" : pickle.load(open("memmapFolds-MMI-bi70cell1hidd-3/0/binarizers/lbSede1.p", 'rb')),
        "LSTMbidirectional5e" : pickle.load(open("memmapFolds-LSTMbidirectional5e/0/binarizers/lbSede1.p", 'rb')),
        "SVMbigrams" : pickle.load(open("memmapFolds-SVMbigrams/binarizers/0/lbSede1.p", 'rb')),
        "LSTMconvolutional" : pickle.load(open("memmapFolds-LSTMconvolutional/0/binarizers/lbSede1.p", 'rb')),
        }

#calculate predictions
predictions = {}
for m in models:
    predictions[m] = []
    if type(predictionsRaw[m]['yp']) is dict:
        currPredictions = binarizers[m].inverse_transform(predictionsRaw[m]['yp']['sede1'], 0).astype(int)
    else:
        currPredictions = binarizers[m].inverse_transform(predictionsRaw[m]['yp'], 0).astype(int)

    for i in range(len(currPredictions)):
        if len(currPredictions) == 9432 or i != 9336:
            predictions[m].append(currPredictions[i])

    predictions[m] = np.array(predictions[m])

#majority vote
predictionsVote = []
for i in range(len(predictions[models[0]])):
    count = Counter([predictions[m][i] for m in models])
    predictionsVote.append(count.most_common(1)[0][0])

predictionsVote = np.array(predictionsVote)

#true values
m = models[0]
true = []
if type(predictionsRaw[m]['yt']) is dict:
    currTrue = binarizers[m].inverse_transform(predictionsRaw[m]['yt']['sede1'], 0).astype(int)
else:
    currTrue = binarizers[m].inverse_transform(predictionsRaw[m]['yt'], 0).astype(int)

for i in range(len(currTrue)):
    if len(currTrue) == 9432 or i != 9336:
        true.append(currTrue[i])

true = np.array(true)

#average vote
labels = binarizers[models[0]].classes_.astype(int)

index = {}
for m in models:
    if m in intModels:
        index[m] = np.argmax(binarizers[m].transform(labels),1)
    else:
        index[m] = np.argmax(binarizers[m].transform(np.array(list(map(lambda e:str(e)+"\n", labels)))), 1)

#reverseIndex = {}
#for m in models:
#    reverseIndex[m] = np.argsort(index[m])

predictionsOrdered = {}
for m in models:
    predictionsOrdered[m] = []
    if type(predictionsRaw[m]['yp']) is dict:
        currPredictions = predictionsRaw[m]['yp']['sede1'][:,index[m]]
    else:
        currPredictions = predictionsRaw[m]['yp'][:,index[m]]

    for i in range(len(currPredictions)):
        if len(currPredictions) == 9432 or i != 9336:
            predictionsOrdered[m].append(currPredictions[i])

    predictionsOrdered[m] = np.array(predictionsOrdered[m])

for m in models:
    arr1 = np.array([labels[e] for e in np.argmax(predictionsOrdered[m], 1)])
    arr2 = np.array(predictions[m])
    #print(m)
    #print(np.array_equal(arr1, arr2))

predictionsAverage = np.zeros(predictionsOrdered[models[0]].shape)
for m in models:
    predictionsAverage += predictionsOrdered[m]

predictionsAverage /= len(models)

predictionsAverageTop1 = np.array([labels[e] for e in np.argmax(predictionsAverage, 1)])

#top N
topValues = [2,3,4,5]

predictionsTopN = {}
for m in models:
    predictionsTopN[m] = {}
    for n in topValues:
        predictionsTopN[m][n] = []
        for i in range(len(predictionsOrdered[m])):
            tops = np.array([labels[e] for e in np.argpartition(predictionsOrdered[m][i], -n)[-n:]])
            if true[i] in tops:
                predictionsTopN[m][n].append(true[i])
            else:
                predictionsTopN[m][n].append(labels[np.argmax(predictionsOrdered[m][i])])
        predictionsTopN[m][n] = np.array(predictionsTopN[m][n])

predictionsAverageTopN = {}
for n in topValues:
    predictionsAverageTopN[n] = []
    for i in range(len(predictionsAverage)):
        tops = np.array([labels[e] for e in np.argpartition(predictionsAverage[i], -n)[-n:]])
        if true[i] in tops:
            predictionsAverageTopN[n].append(true[i])
        else:
            predictionsAverageTopN[n].append(labels[np.argmax(predictionsAverage[i])])
    predictionsAverageTopN[n] = np.array(predictionsAverageTopN[n])

#print stuff
print("======= single models")
for m in models:
    print(m)
    print(sklearn.metrics.cohen_kappa_score(predictions[m], true))

print("\n======= max vote")
print(sklearn.metrics.cohen_kappa_score(predictionsVote, true))

print("\n======= avg vote")
print(sklearn.metrics.cohen_kappa_score(predictionsAverageTop1, true))

for n in topValues:
    print("\n======= top{} models".format(n))
    for m in models:
        print(m)
        print(sklearn.metrics.cohen_kappa_score(predictionsTopN[m][n], true))

    print("\n======= top{} avg vote".format(n))
    print(sklearn.metrics.cohen_kappa_score(predictionsAverageTopN[n], true))

