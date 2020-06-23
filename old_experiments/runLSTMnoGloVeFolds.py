import datetime
import notifier
import sys
from myLSTMnoGloVeFold import MyLSTM

startTime = str(datetime.datetime.now())

#for fold in range(10):
#for fold in range(5, 8):
for fold in range(1,2):
    print("########################################### Fold {} ({})".format(fold, str(datetime.datetime.now())))
    #for task in ['sede1', 'sede12', 'morfo1', 'morfo2']:
    #for task in ['sede12', 'sede1', 'morfo1', 'morfo2']:
    for task in ['morfo2']:
        startPartTime = str(datetime.datetime.now())
        #lstm = MyLSTM(fold, tasks=[task], foldsFolder="foldsS")
        lstm = MyLSTM(fold, tasks=[task], epochs = 5, patience = 5)

        #if fold == 0:
        if False:
            lstm.loadData()
            lstm.loadModels()
            lstm.continueTrainModels(epochs=5)
            lstm.saveModels()
            lstm.evaluate()
        else:
            lstm.extractData()
            lstm.createModels()
            lstm.saveModels()
            lstm.evaluate()
        
            for _ in range(3):
                lstm.continueTrainModels(epochs=5)
                lstm.saveModels()
                lstm.evaluate()
        
        lstm.clearDataBinarizers()
        endPartTime = str(datetime.datetime.now())
        notifier.sendMessage("myLSTMnoGloVe partial", "on uriel\nstart: "+startPartTime+"  -  end: "+endPartTime)

endTime = str(datetime.datetime.now())
print("########################################### Finish ({})".format(endTime))
notifier.sendMessage("myLSTMnoGloVe finished", "on uriel\nstart: "+startTime+"  -  end: "+endTime)

