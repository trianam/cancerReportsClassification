import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, conf):
        super(MyModel, self).__init__()

        #TODO: add conf.afterFeaturesLayers on phrase and doc levels?

        self.conf = conf
        
        nextDim = conf.vecLen * conf.seqLen
        
        self.hiddenLayers = nn.ModuleList([])
        self.dropouts = nn.ModuleList([])
        for _ in range(conf.hiddenLayers):
            self.hiddenLayers.append(nn.Linear(nextDim, conf.hiddenDim))
            nextDim = conf.hiddenDim

            if conf.dropout > 0.:
                self.dropouts.append(nn.Dropout(conf.dropout))
        
        self.outLayer = nn.Linear(nextDim, conf.outDim)
        
    def forward(self, x, batchIndex=None, getAttention=False):
        #phrase level
        
        x = x.view(-1, self.conf.vecLen * self.conf.seqLen)

        for i in range(len(self.hiddenLayers)):
            x = self.hiddenLayers[i](x)
            x = F.relu(x)
            if self.conf.dropout > 0.:
                x = self.dropouts[i](x)

        
        x = self.outLayer(x)
        if self.conf.outDim == 1:
            x = torch.sigmoid(x)
        #else:
        #    #x = torch.softmax(x,1)
        #    x = torch.log_softmax(x,1)

        return x


