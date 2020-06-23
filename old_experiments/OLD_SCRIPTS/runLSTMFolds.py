import datetime
import notifier
import sys
from myLSTMFold import MyLSTM

startTime = str(datetime.datetime.now())

#for fold in range(8, 10):
for fold in range(10):
    print("########################################### Fold {}".format(fold))
    lstm = MyLSTM(fold)

    lstm.extractData()
    #lstm.loadData()
    
    lstm.createModels()
    lstm.saveModels()
    #lstm.loadModels()
    
    lstm.evaluate()

endTime = str(datetime.datetime.now())
notifier.sendMessage("myLSTM finished", "start: "+startTime+"  -  end: "+endTime)

