#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
#os.environ["CUDA_VISIBLE_DEVICES"]="0";
import configurationsMulticlass
import funPytorch as fun

conf = configurationsMulticlass.configFolds2

device = "cuda:1"

#model.main(configurations.configRun1)

first = True
for f in conf:
    print("################# FOLD {}".format(f)) 
    if first:
        print("======= LOAD DATA")
        X, y, train, valid, test = fun.processData(conf[f])
        first = False
    else:
        _,_, train, valid, test = fun.processCorpusAndSplit(conf[f])
    print("======= CREATE MODEL") 
    model,optim = fun.makeModel(conf[f], device)
    print("======= TRAIN MODEL")
    fun.runTrain(conf[f], model, optim, X, y, train, valid, test) 


