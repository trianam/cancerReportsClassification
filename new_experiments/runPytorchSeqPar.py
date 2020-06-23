#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
#os.environ["CUDA_VISIBLE_DEVICES"]="0";
import configurations
import funPytorch as fun
import time
import sys
import notifier
import platform

confSeq = configurations.configRunSeqPar

device = "cuda:{}".format(sys.argv[2])

#model.main(configurations.configRun1)

skipped = []
times = []
for conf,name in confSeq[int(sys.argv[1])]:
    times.append("CONF {}  -  {}".format(name, time.ctime()))
    print("##################### CONF {}".format(name))
    if fun.alreadyLaunched(conf):
        skipped.append("CONF {}".format(name))
        print("###################################### SKIP, ALREADY LAUNCHED!")
        continue
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

times.append("END  -  {}".format(time.ctime()))
if len(skipped)>0:
    print("=============================")
    print("ATTENTION! SKIPPED SOMETHING!")

notifier.sendMessage("Training finished", "Finished task {} on {}.\n\n{}\n\nskipped:\n{}".format(sys.argv[1], platform.node(), "\n".join(times), "\n".join(skipped)))
time.sleep(10)

