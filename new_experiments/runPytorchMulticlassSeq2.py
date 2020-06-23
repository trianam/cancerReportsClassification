#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
#os.environ["CUDA_VISIBLE_DEVICES"]="0";
import configurationsMulticlass
import funPytorch as fun
import time

confSeq = configurationsMulticlass.configRunSeq2

device = "cuda:1"

#model.main(configurations.configRun1)

for conf,name in confSeq:
    print("##################### CONF {}".format(name))
    print("======= CREATE MODEL") 
    model,optim = fun.makeModel(conf, device)
    print("======= LOAD DATA")
    X, y, train, valid, test = fun.processData(conf)
    print("======= TRAIN MODEL")
    fun.runTrain(conf, model, optim, X, y, train, valid, test) 
    #print("======= TEST MODEL") 
    #fun.runTest(conf, model, X, y, test) 
    #print("======= SAVE MODEL")
    #fun.saveModel(conf, model, optim)

    del model
    del optim
    del X
    del y
    del train
    del valid
    del test

    time.sleep(30)

