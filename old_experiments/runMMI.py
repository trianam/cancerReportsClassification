import datetime
import notifier
import sys
from myMMI import MyMMI

startTime = str(datetime.datetime.now())

#for fold in range(1):
#for fold in range(2, 10):
for fold in range(10):
    print("########################################### Fold {} ({})".format(fold, str(datetime.datetime.now())))
    for task in ['sede1', 'sede12', 'morfo1', 'morfo2']:
        startPartTime = str(datetime.datetime.now())
        mmi = MyMMI(fold, tasks=[task], epochs = 5, patience = 5)

        #if fold == 7:
        if False:
            mmi.loadData()
            mmi.loadModels()
            for _ in range(4):
                mmi.continueTrainModels(epochs=5)
                mmi.saveModels()
                mmi.evaluate()
        else:
            mmi.extractData()

            mmi.createModels()
            mmi.saveModels()
            mmi.evaluate()

            for _ in range(7):
                mmi.continueTrainModels(epochs=5)
                mmi.saveModels()
                mmi.evaluate()

        endPartTime = str(datetime.datetime.now())
        #notifier.sendFile("myMMI partial", mmi._fileTable, "start: "+startPartTime+"  -  end: "+endPartTime)
        notifier.sendMessage("myMMI partial", "on mari\nstart: "+startPartTime+"  -  end: "+endPartTime)

endTime = str(datetime.datetime.now())
print("########################################### Finish ({})".format(endTime))
notifier.sendMessage("myMMI finished", "on mari\nstart: "+startTime+"  -  end: "+endTime)

