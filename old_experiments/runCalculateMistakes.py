import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1";

from myMMI import MyMMI

for var in range(3):
    mmi = MyMMI(0, tasks=['sede1'], epochs = 5, patience = 5, lstmLayers=1, lstmCells=70, hiddenLayers=1, bidirectional=True, filesFolder="./filesFolds-MMI-bi70cell1hidd-"+str(var), memmapFolder="./memmapFolds-MMI-bi70cell1hidd-"+str(var), useMemmap=False)
    mmi.extractData()
    mmi.loadSpecificModels({'sede1':7})
    mmi.calculateMistakenSamples()
    mmi.calculateMistakenSamples(top=5)

