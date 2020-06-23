import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0";
import configurations
import funKeras as fun

conf = configurations.configRun1

#model.main(configurations.configRun1)

print("======= CREATE MODEL") 
model = fun.makeModel(conf)
print("======= LOAD DATA")
X, y, train, valid, test = fun.loadData(conf)
print("======= TRAIN MODEL")
fun.runTrain(conf, model, X[train], y[train], X[valid], y[valid]) 
print("======= TEST MODEL") 
fun.runTest(conf, model, X[test], y[test]) 
print("======= SAVE MODEL")
fun.save(conf, model)


