#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
#os.environ["CUDA_VISIBLE_DEVICES"]="0";
import configurations
import funPytorch as fun

conf = configurations.configRun1

device = "cuda:0"

#model.main(configurations.configRun1)

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


