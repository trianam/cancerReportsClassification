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
        if conf.featuresDropout > 0.:
            self.featuresDropout = nn.Dropout(conf.featuresDropout)
        
        
        if self.conf.bidirectionalMerge is None:
            nextFeaturesDim = 2 * conf.seqFeaturesDim
        else:
            nextFeaturesDim = conf.seqFeaturesDim
        
        self.afterFeaturesLayers = nn.ModuleList([])
        if conf.afterFeaturesLayers > 0:
            for _ in range(conf.afterFeaturesLayers -1):
                self.afterFeaturesLayers.append(nn.Linear(nextFeaturesDim, conf.afterFeaturesDim))
                nextFeaturesDim = conf.afterFeaturesDim

            self.afterFeaturesLayers.append(nn.Linear(nextFeaturesDim, conf.afterFeaturesOutDim))
            nextFeaturesDim = conf.afterFeaturesOutDim

        if conf.classLayers > 0:
            if conf.classLayers > 1:
                if conf.hiddenDim is None:
                    hiddenDim = conf.outDim
                else:
                    hiddenDim = conf.hiddenDim
            else:
                hiddenDim = nextFeaturesDim

            self.hiddenLayers = nn.ModuleList([])
            for _ in range(conf.classLayers - 1):
                self.hiddenLayers.append(nn.Linear(nextFeaturesDim, hiddenDim))

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

        if self.conf.bidirectionalMerge is None:
            nextFeaturesDim = 2*self.conf.seqFeaturesDim
        else:
            x = x.view(-1, self.conf.seqLen, 2, self.conf.seqFeaturesDim)
            
            if self.conf.bidirectionalMerge == 'avg':
                x = (x[:,:,0] + x[:,:,1]) / 2.
            elif self.conf.bidirectionalMerge == 'max':
                x = torch.max(x, 2).values  

            nextFeaturesDim = self.conf.seqFeaturesDim

        if self.conf.afterFeaturesLayers > 0:
            x = x.contiguous().view(-1, nextFeaturesDim)
            for i in range(len(self.afterFeaturesLayers)-1):
                x = self.afterFeaturesLayers[i](x)
                x = F.relu(x)
            x = self.afterFeaturesLayers[-1](x)
            x = torch.sigmoid(x)
            x = x.view(-1, self.conf.seqLen, self.conf.afterFeaturesOutDim)
            nextFeaturesDim = self.conf.afterFeaturesOutDim

        if getAttention:
            status = x.clone().detach()
            #status = status.view(-1, self.conf.seqLen, self.conf.seqFeaturesDim)

        if self.conf.seqDropout > 0.:
            x = self.seqDropout(x)

        x = torch.max(x, 1).values

        #classification
        if self.conf.featuresDropout > 0.:
            x = self.featuresDropout(x)

        if self.conf.classLayers > 0:
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


