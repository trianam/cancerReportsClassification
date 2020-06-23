import datetime
import notifier
import sys
import gc
from myLSTMbidirectionalFold5NoNegation import MyLSTM

goodModelNumbers = {
    "sede1" : 7,
    "sede12" : 9,
    "morfo1" : 6,
    "morfo2" : 3,
}


startTime = str(datetime.datetime.now())

for fold in range(10):
    print("########################################### Fold {} ({})".format(fold, str(datetime.datetime.now())))
 
    #for task in ['sede1', 'sede12', 'morfo1', 'morfo2']:
    for task in ['sede1']:
        startPartTime = str(datetime.datetime.now())
        lstm = MyLSTM(fold, tasks=[task], epochs = 5, patience = 5)

        lstm.loadData()
        lstm.createModels()
        lstm.saveModels()
        lstm.evaluate()

        for _ in range(goodModelNumbers[task]):
            lstm.continueTrainModels(epochs=5)
            lstm.saveModels()
            lstm.evaluate()

        del(lstm)
        gc.collect()

        endPartTime = str(datetime.datetime.now())
        notifier.sendMessage("myLSTM partial", "on spallina\nstart: "+startPartTime+"  -  end: "+endPartTime)

endTime = str(datetime.datetime.now())
print("########################################### Finish ({})".format(endTime))
notifier.sendMessage("myLSTM finished", "on spallina\nstart: "+startTime+"  -  end: "+endTime)

