import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MyModel(nn.Module):
    def __init__(self, conf):
        super(MyModel, self).__init__()
        self.conf = conf
        
        #self.xOneHot = torch.FloatTensor(conf.batchSize, conf.seqLen, conf.numDigits)

        if conf.rnnType == 'LSTM':
            self.seqLayer = nn.LSTM(conf.numDigits, conf.seqFeaturesDim, conf.seqFeaturesLayers, bidirectional=True, batch_first=True)
            self.sortSeqLayer = nn.LSTM(conf.numDigits, conf.sortSeqFeaturesDim, conf.sortSeqFeaturesLayers, bidirectional=False, batch_first=True)
        elif conf.rnnType == 'GRU':
            self.seqLayer = nn.GRU(conf.numDigits, conf.seqFeaturesDim, conf.seqFeaturesLayers, bidirectional=True, batch_first=True)
            self.sortSeqLayer = nn.GRU(conf.numDigits, conf.sortSeqFeaturesDim, conf.sortSeqFeaturesLayers, bidirectional=False, batch_first=True)

        if conf.seqDropout > 0.:
            self.seqDropout = nn.Dropout(conf.seqDropout)
        if conf.sortSeqDropout > 0.:
            self.sortSeqDropout = nn.Dropout(conf.sortSeqDropout)
        if conf.featuresDropout > 0.:
            self.featuresDropout = nn.Dropout(conf.featuresDropout)
        
        self.afterFeaturesLayers = nn.ModuleList([])
        for _ in range(conf.afterFeaturesLayers):
            self.afterFeaturesLayers.append(nn.Linear(conf.seqFeaturesDim, conf.seqFeaturesDim))

        if conf.hiddenLayers > 0:
            if conf.hiddenDim is None:
                hiddenDim = conf.outDim
            else:
                hiddenDim = conf.hiddenDim
        else:
            hiddenDim = conf.sortSeqFeaturesDim

        self.hiddenLayers = nn.ModuleList([])
        for _ in range(conf.hiddenLayers):
            self.hiddenLayers.append(nn.Linear(conf.sortSeqFeaturesDim, hiddenDim))

        self.outLayer = nn.Linear(hiddenDim, conf.outDim)
    
    #def to(self, *args, **kwargs):
    #    self = super().to(*args, **kwargs) 
    #    self.xOneHot = self.xOneHot.to(*args, **kwargs) 
    #    return self

    def forward(self, x, batchIndex=None, getAttention=False):
        if False and not (self.conf.bistep is None) and not (batchIndex is None):
            if self.conf.bistep is 'random':
                if np.random.randint(2) > 0:
                    batchEven = True
                else:
                    batchEven = False
            elif self.conf.bistep is 'split':
                if batchIndex % 2 == 0:
                    batchEven = True
                else:
                    batchEven = False
            for p in self.seqLayer.parameters():
                p.requires_grad = not batchEven

            for i in range(len(self.afterFeaturesLayers)):
                for p in self.afterFeaturesLayers[i].parameters():
                    p.requires_grad = not batchEven
                    
            for p in self.sortSeqLayer.parameters():
                p.requires_grad = batchEven

            for i in range(len(self.hiddenLayers)):
                for p in self.hiddenLayers[i].parameters():
                    p.requires_grad = batchEven
                    
            for p in self.outLayer.parameters():
                p.requires_grad = batchEven


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
        
        s,_ = self.seqLayer(x)
        s = s.view(-1, self.conf.seqLen, 2, self.conf.seqFeaturesDim)
        
        if self.conf.bidirectionalMerge == 'avg':
            s = (s[:,:,0] + s[:,:,1]) / 2.
        elif self.conf.bidirectionalMerge == 'max':
            s = torch.max(s, 2).values  

        if self.conf.afterFeaturesLayers > 0:
            s = s.contiguous().view(-1, self.conf.seqFeaturesDim)
            for i in range(len(self.afterFeaturesLayers)):
                s = self.afterFeaturesLayers[i](s)
                s = F.relu(s)
            s = s.view(-1, self.conf.seqLen, self.conf.seqFeaturesDim)

        if getAttention:
            status = s.clone().detach()
            status = status.view(-1, self.conf.seqLen, self.conf.seqFeaturesDim)

        if self.conf.seqDropout > 0.:
            s = self.seqDropout(s)

        #TODO: probably gradient is not propagated through s
        _,s = torch.sort(s, 1)
        #TODO: work only with seqFeaturesDim == 1
        #s = s.view(-1, self.conf.seqLen)
        s = s.expand(-1, -1, self.conf.numDigits)
        x = torch.gather(x,1,s)

        x,_ = self.sortSeqLayer(x)
        x = x[:,-1,:]

        if self.conf.sortSeqDropout > 0.:
            x = self.sortSeqDropout(x)

        #classification
        if self.conf.featuresDropout > 0.:
            x = self.featuresDropout(x)

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


