import pickle

with open("y.txt",'rt') as yFile:
    y = yFile.readlines()

duplicates = pickle.load(open("results.p", 'rb'))

for k in duplicates:
    duplicates[k]['indicesY'] = []
    for i in sorted(duplicates[k]['indices']):
        duplicates[k]['indicesY'].append((i,y[i]))

pickle.dump(duplicates, open("resultsIntegrated.p", 'wb'))

