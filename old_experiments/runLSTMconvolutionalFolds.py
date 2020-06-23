import datetime
import notifier
import sys
from myLSTMconvolutionalFold import MyLSTM

startTime = str(datetime.datetime.now())

#for fold in range(1):
#for fold in range(10):
for fold in range(6, 10):
    print("########################################### Fold {} ({})".format(fold, str(datetime.datetime.now())))
    #for task in ['sede1', 'sede12', 'morfo1', 'morfo2']:
    #for task in ['sede12', 'sede1', 'morfo1', 'morfo2']:
    for task in ['morfo2']:
        startPartTime = str(datetime.datetime.now())
        #lstm = MyLSTM(fold, tasks=[task], foldsFolder="foldsS")
        lstm = MyLSTM(fold, tasks=[task], epochs = 5, patience = 5)

        #if fold == 0:
        #    lstm.loadData()
        #else:
        #    lstm.extractData()
        
        lstm.extractData()
        #lstm.loadData()

        lstm.createModels()
        lstm.saveModels()
        lstm.evaluate()
        #lstm.loadModels()

        for _ in range(9):
            lstm.continueTrainModels(epochs=5)
            lstm.saveModels()
            lstm.evaluate()

        endPartTime = str(datetime.datetime.now())
        #notifier.sendFile("myLSTM partial", lstm._fileTable, "start: "+startPartTime+"  -  end: "+endPartTime)
        notifier.sendMessage("myLSTMconvolutional partial", "on logos\nstart: "+startPartTime+"  -  end: "+endPartTime)

endTime = str(datetime.datetime.now())
print("########################################### Finish ({})".format(endTime))
notifier.sendMessage("myLSTM convolutional finished", "on logos\nstart: "+startTime+"  -  end: "+endTime)

