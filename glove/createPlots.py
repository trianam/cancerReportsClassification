import numpy as np
import re
import sys

iterXmaxs = [(15,10), (100,100)]
corpuses = ["aladarV2", "aladarWiki", "aladarV3"]
vectorSizes = [50, 100, 200, 300]
windowSizes = [5, 10, 15, 20]
topNumbers = [1, 5, 10, 15]
questions = ["benigno-tessuto", "maligno-tessuto", "benigno-maligno", "gram8-plurali", "gram8-plurali-short", "total"]


hypercube = np.ones((len(iterXmaxs), len(corpuses), len(vectorSizes), len(windowSizes), len(topNumbers), len(questions)))*(-1)

for ii,i in enumerate(iterXmaxs):
    path1 = "auto/iter"+str(i[0])+"-xMax"+str(i[1])+"/"
    for cc,c in enumerate(corpuses):
        for vv,v in enumerate(vectorSizes):
            for ww,w in enumerate(windowSizes):
                path2 = "out-"+str(c)+"-"+str(v)+"-"+str(w)+"/"
                for tt,t in enumerate(topNumbers):
                    filename = "evalTOP"+str(t)+".txt"
                    try:
                        with open(path1+path2+filename) as f:
                            while True:
                                l = f.readline()
                                quest = re.sub("\.txt:\n", "", l)
                                l = f.readline()
                                val = float(re.sub(".*: (.+)%.*", "\\1", l))

                                found = False
                                for qq in range(len(questions) - 1):
                                    if quest == questions[qq]:
                                        found = True
                                        hypercube[ii,cc,vv,ww,tt,qq] = val
                                        break
                                if not found:
                                    hypercube[ii,cc,vv,ww,tt,len(questions)-1] = val
                                    break
                    except IOError:
                        pass

                                
                            
                
