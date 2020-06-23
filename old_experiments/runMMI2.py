import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1";
import datetime
import notifier
import sys
import time
from myMMI import MyMMI
#from keras import backend as K

startTime = str(datetime.datetime.now())

#for featuresDim in [2,3,5,8,13,22,36,60,100]:
for featuresDim in [200]:
    if featuresDim == 60:
        featuresLayersList = [8]
    else:
        #featuresLayersList = [1,2,3,5,8,13,22,36,60]
        featuresLayersList = [2]

    for featuresLayers in featuresLayersList:
        #for fold in range(10):
        #for fold in range(8, 10):
        for fold in range(1):
            print("########################################### Fold {},{} ({})".format(featuresDim, featuresLayers, str(datetime.datetime.now())))
            for task in ['sede1']:
                startPartTime = str(datetime.datetime.now())
                mmi = MyMMI(fold, tasks=[task], epochs = 5, patience = 5, featuresLayers=featuresLayers, featuresDim=featuresDim, hiddenLayers=1, featuresType='cnn', filesFolder="./filesFolds-MMI-cnn{}x{}-1hidd".format(featuresLayers, featuresDim), memmapFolder="./memmapFolds-MMI-cnn{}x{}-1hidd".format(featuresLayers, featuresDim), useMemmap=False)
                time.sleep(10)

                #if fold == 7:
                if False:
                    mmi.loadData()
                    #mmi.extractData()
                    mmi.loadModels()
                    for _ in range(7):
                        mmi.continueTrainModels(epochs=5)
                        mmi.saveModels()
                        mmi.evaluate()
                else:
                    tentatives = 0
                    while True:
                        try:
                            mmi.extractData()
                        except MemoryError as err:
                            tentatives += 1
                            if tentatives < 10:
                                print("handling MemoryError")
                                del mmi
                                #K.clear_session()
                                time.sleep(60)
                                mmi = MyMMI(fold, tasks=[task], epochs = 5, patience = 5, featuresLayers=featuresLayers, featuresDim=featuresDim, hiddenLayers=1, featuresType='cnn', filesFolder="./filesFolds-MMI-cnn{}x{}-1hidd".format(featuresLayers, featuresDim), memmapFolder="./memmapFolds-MMI-cnn{}x{}-1hidd".format(featuresLayers, featuresDim), useMemmap=False)
                                time.sleep(10)
                            else:
                                endTime = str(datetime.datetime.now())
                                notifier.sendMessage("MEMORY ERROR on runMMI2.py on sugaar2 finished", "start: "+startTime+"  -  end: "+endTime)
                                time.sleep(60)
                                raise err
                        else:
                            break

                    mmi.createModels()
                    mmi.saveModels()
                    mmi.evaluate()
                    mmi.plotSummary()

                    for _ in range(10):
                        mmi.continueTrainModels(epochs=5)
                        mmi.saveModels()
                        mmi.evaluate()

                del mmi
                time.sleep(30)

                endPartTime = str(datetime.datetime.now())
                #notifier.sendFile("myMMI partial", mmi._fileTable, "start: "+startPartTime+"  -  end: "+endPartTime)
                notifier.sendMessage("cnn{}x{}-1hidd on sugaar2 partial".format(featuresLayers, featuresDim), "start: "+startPartTime+"  -  end: "+endPartTime)

endTime = str(datetime.datetime.now())
print("########################################### Finish ({})".format(endTime))
notifier.sendMessage("runMMI2 on sugaar2 finished", "start: "+startTime+"  -  end: "+endTime)

