import datetime
import notifier
import sys
from myLSTMKfold import MyLSTM

startTime = str(datetime.datetime.now())

lstm = MyLSTM()

lstm.extractData()
#lstm.loadData()
lstm.createModels()
lstm.saveModels()
#lstm.loadModels()

endTime = str(datetime.datetime.now())
notifier.sendMessage("myLSTM finished", "start: "+startTime+"  -  end: "+endTime)

