#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
#os.environ["CUDA_VISIBLE_DEVICES"]="1";
import configurationsPrimeTest as configurations
import funPytorch as fun

confContinue = configurations.configRun3Continue
#confSave = configurations.configRun1Save

device = "cuda:0"

#model.main(configurations.configRun1)

print("======= CREATE MODEL") 
model,optim = fun.loadModel(confContinue, device)
print("======= LOAD DATA")
X, y, train, valid, test = fun.processData(confContinue)
print("======= TRAIN MODEL")
fun.runTrain(confContinue, model, optim, X, y, train, valid, test) 
#print("======= TEST MODEL") 
#fun.runTest(confContinue, model, X, y, test) 
#print("======= SAVE MODEL")
#fun.saveModel(confSave, model, optim)


