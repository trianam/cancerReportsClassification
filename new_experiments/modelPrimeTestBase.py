import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, conf):
        super(MyModel, self).__init__()
        self.conf = conf
        
        #self.xOneHot = torch.FloatTensor(conf.batchSize, conf.seqLen, conf.numDigits)

        if conf.rnnType == 'LSTM':
            self.seqLayer = nn.LSTM(conf.numDigits, conf.seqFeaturesDim, conf.seqFeaturesLayers, bidirectional=True, batch_first=True)
        elif conf.rnnType == 'GRU':
            self.seqLayer = nn.GRU(conf.numDigits, conf.seqFeaturesDim, conf.seqFeaturesLayers, bidirectional=True, batch_first=True)

        if conf.seqDropout > 0.:
            self.seqDropout = nn.Dropout(conf.seqDropout)
        
        if conf.classLayers > 1:
            if conf.hiddenDim is None:
                hiddenDim = conf.outDim
            else:
                hiddenDim = conf.hiddenDim
        else:
            hiddenDim = 2*conf.seqFeaturesDim

        self.hiddenLayers = nn.ModuleList([])
        for _ in range(conf.classLayers-1):
            self.hiddenLayers.append(nn.Linear(2*conf.seqFeaturesDim, hiddenDim))

        self.outLayer = nn.Linear(hiddenDim, conf.outDim)
    
    #def to(self, *args, **kwargs):
    #    self = super().to(*args, **kwargs) 
    #    self.xOneHot = self.xOneHot.to(*args, **kwargs) 
    #    return self

    def forward(self, x, batchIndex=None, getAttention=False):
        #seq level
        x = x.view(-1, self.conf.seqLen, 1)
        x = x.type(torch.long)

        #self.xOneHot.zero_()
        #self.xOneHot.scatter_(2, x, 1)
        #x = self.xOneHot
        
        #xOneHot = torch.FloatTensor(x.shape[0], x.shape[1], self.conf.numDigits)
        #xOneHot = xOneHot.to(x.device)
        xOneHot = torch.zeros((x.shape[0], x.shape[1], self.conf.numDigits), dtype=torch.float, device=x.device)
        xOneHot.scatter_(2, x, 1)
        x = xOneHot
        
        x,_ = self.seqLayer(x)

        if getAttention:
            status = x.clone().detach()
            status = status.view(-1, self.conf.seqLen, 2*self.conf.seqFeaturesDim)

        x = x[:,-1,:]
        #x = x.contiguous().view(-1, self.conf.seqLen*2*self.conf.seqFeaturesDim)

        if self.conf.seqDropout > 0.:
            x = self.seqDropout(x)

        #classification
        for i in range(len(self.hiddenLayers)):
            x = self.hiddenLayers[i](x)
            x = F.relu(x)

        x = self.outLayer(x)
        if self.conf.outDim == 1:
            x = torch.sigmoid(x)
        #else:
        #    #x = torch.softmax(x,1)
        #    x = torch.log_softmax(x,1)


        if getAttention:
            return x, status
        else:
            return x


