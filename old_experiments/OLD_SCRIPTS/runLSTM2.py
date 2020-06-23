import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import datetime
import notifier
import sys
from myLSTM2 import MyLSTM

startTime = str(datetime.datetime.now())

lstm = MyLSTM()

lstm.extractData()
#lstm.loadData()
lstm.createModels()
lstm.saveModels()
#lstm.loadModels()
#lstm.createFineTuningModel()
#lstm.createFineTuning2Model()
#lstm.createFineTuning3Model()
#lstm.createFineTuning4Model()
#lstm.createFineTuning5Model()
#lstm.createFineTuning6Model()
#lstm.saveFineTuningModel()
#lstm.saveFineTuning2Model()
#lstm.saveFineTuning3Model()
#lstm.saveFineTuning4Model()
#lstm.saveFineTuning5Model()
#lstm.saveFineTuning6Model()
#lstm.loadFineTuningModel()
#lstm.loadFineTuning2Model()
#lstm.loadFineTuning3Model()
#lstm.evaluate()
#lstm.evaluateFineTuning3()

endTime = str(datetime.datetime.now())
notifier.sendMessage("myLSTM finished", "start: "+startTime+"  -  end: "+endTime)

sys.exit()

lstm.extractData()
lstm.createModels()
lstm.saveModels()
lstm.evaluate()

sys.exit()


pp = modelBinMorfo2.predict_classes(X)
ppb = np.zeros(yMorfo2.shape, np.int)
for i,j in enumerate(pp):            
    ppb[i][j] = 1            

media = 0
mini = np.inf
maxi = 0
for i in range(ppb.shape[1]):
    currAcc = sk.metrics.accuracy_score(yMorfo2[:,i], ppb[:,i])
    media += currAcc
    if currAcc < mini:
        mini = currAcc
    if currAcc > maxi:
        maxi = currAcc
media /= ppb.shape[1]
multi = sk.metrics.accuracy_score(yMorfo2, ppb)

sk.metrics.cohen_kappa_score(lstm.lbMorfo1.inverse_transform(lstm.yMorfo1Test), lstm.lbMorfo1.inverse_transform(ppb))


with open("./corpusLSTM/sedeICDO3clean.txt") as fid:
    sedeICDO3clean = fid.readlines()

yUnSede1 = np.zeros((len(sedeICDO3clean)), np.int)
yUnSede2 = np.zeros((len(sedeICDO3clean)), np.int)
for i,c in enumerate(sedeICDO3clean):
    yUnSede1[i], yUnSede2[i] = c.split()

ySede1 = lb.fit_transform(yUnSede1)
pp = modelCatSede1.predict_classes(X)
ppb = np.zeros(ySede1.shape, np.int)
for i,j in enumerate(pp):
    ppb[i][j] = 1

ppu = lb.inverse_transform(ppb)
sk.metrics.cohen_kappa_score(ppu, yUnSede1)

