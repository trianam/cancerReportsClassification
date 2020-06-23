#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import datetime
import notifier
import sys
from myLSTM import MyLSTM

startTime = str(datetime.datetime.now())

vectorFiles = [
    "vectors1.txt",
    "vectors2.txt",
    "vectors3.txt",
    "vectors4.txt",
    "vectors5.txt",
    "vectors6.txt",
    "vectors7.txt",
    "vectors8.txt",
    "vectors9.txt",
    "vectors10.txt",
    "vectors11.txt",
    "vectors12.txt",
    "vectors13.txt",
    "vectors14.txt",
    "vectors15.txt",
    "vectors16.txt",
]

for vectorFile in vectorFiles:
    print("########################## "+vectorFile)

    lstm = MyLSTM(vectorFile)

    lstm.extractData()
    lstm.createModels()
    lstm.evaluate()

endTime = str(datetime.datetime.now())
notifier.sendMessage("myLSTMauto finished", "start: "+startTime+"  -  end: "+endTime)


