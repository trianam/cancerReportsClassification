import pickle

with open("text.txt",'rt') as textFile:
    text = textFile.readlines()

with open("y.txt",'rt') as yFile:
    y = yFile.readlines()

duplicates = {}
for i in range(len(text)):
    print("{}               ".format(i), end="\r")
    for j in range(i+1,len(text)):
        if text[i] == text[j]:
            print("FOUND {},{}".format(i,j))
            if text[i] not in duplicates.keys():
                duplicates[text[i]] = {
                    "indices":set(),
                    "y":set(),
                }
            duplicates[text[i]]['indices'].add(i)
            duplicates[text[i]]['indices'].add(j)
            duplicates[text[i]]['y'].add(y[i])
            duplicates[text[i]]['y'].add(y[j])

pickle.dump(duplicates, open("results.p", 'wb'))

