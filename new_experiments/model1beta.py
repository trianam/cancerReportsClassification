import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, conf):
        super(MyModel, self).__init__()

        #TODO: add conf.afterFeaturesLayers on phrase and doc levels?

        self.conf = conf

        if conf.rnnType == 'LSTM':
            self.phraseLayer = nn.LSTM(conf.vecLen, conf.phraseFeaturesDim, conf.phraseFeaturesLayers, bidirectional=True, batch_first=True)
            self.fieldLayer = nn.LSTM(conf.phraseFeaturesDim*2, conf.fieldFeaturesDim, conf.fieldFeaturesLayers, bidirectional=True, batch_first=True)
            self.docLayer = nn.LSTM(conf.fieldFeaturesDim*2, conf.docFeaturesDim, conf.docFeaturesLayers, bidirectional=True, batch_first=True)
        elif conf.rnnType == 'GRU':
            self.phraseLayer = nn.GRU(conf.vecLen, conf.phraseFeaturesDim, conf.phraseFeaturesLayers, bidirectional=True, batch_first=True)
            self.fieldLayer = nn.GRU(conf.phraseFeaturesDim*2, conf.fieldFeaturesDim, conf.fieldFeaturesLayers, bidirectional=True, batch_first=True)
            self.docLayer = nn.GRU(conf.fieldFeaturesDim*2, conf.docFeaturesDim, conf.docFeaturesLayers, bidirectional=True, batch_first=True)

        if conf.phraseDropout > 0.:
            self.phraseDropout = nn.Dropout(conf.phraseDropout)
        if conf.fieldDropout > 0.:
            self.fieldDropout = nn.Dropout(conf.fieldDropout)
        if conf.docDropout > 0.:
            self.docDropout = nn.Dropout(conf.docDropout)
        if conf.featuresDropout > 0.:
            self.featuresDropout = nn.Dropout(conf.featuresDropout)
        
        self.afterFeaturesLayers = nn.ModuleList([])
        for _ in range(conf.afterFeaturesLayers):
            self.afterFeaturesLayers.append(nn.Linear(conf.phraseFeaturesDim*2, conf.phraseFeaturesDim*2))

        if conf.hiddenLayers > 0:
            if conf.hiddenDim is None:
                hiddenDim = conf.outDim
            else:
                hiddenDim = conf.hiddenDim
        else:
            hiddenDim = conf.docFeaturesDim*2

        self.hiddenLayers = nn.ModuleList([])
        for _ in range(conf.hiddenLayers):
            self.hiddenLayers.append(nn.Linear(conf.docFeaturesDim*2, hiddenDim))

        self.outLayer = nn.Linear(hiddenDim, conf.outDim)
        
    def forward(self, x, batchIndex=None, getAttention=False):
        #phrase level
        x = x.view(-1, self.conf.phraseLen, self.conf.vecLen)
        
        if self.conf.masking:
            mask = (x != 0).all(2).float()

        x,_ = self.phraseLayer(x)

        if self.conf.afterFeaturesLayers > 0:
            x = x.contiguous().view(-1, self.conf.phraseFeaturesDim*2)
            for i in range(len(self.afterFeaturesLayers)):
                x = self.afterFeaturesLayers[i](x)
                x = F.relu(x)
            x = x.view(-1, self.conf.phraseLen, self.conf.phraseFeaturesDim*2)

        if self.conf.masking:
            mask = mask.view(-1, self.conf.phraseLen, 1).expand(-1, -1, self.conf.phraseFeaturesDim*2)
            x = x * mask
            x = x - (1-mask)

        if getAttention:
            status = x.clone().detach()
            status = status.view(-1, self.conf.numFields, self.conf.numPhrases, self.conf.phraseLen, self.conf.phraseFeaturesDim*2)

        if self.conf.phraseDropout > 0.:
            x = self.phraseDropout(x)

        x = torch.max(x, 1).values

        #field level
        x = x.view(-1, self.conf.numPhrases, self.conf.phraseFeaturesDim*2)

        if self.conf.masking:
            mask = (x != -1).all(2).float()

        x,_ = self.fieldLayer(x)

        if self.conf.masking:
            mask = mask.view(-1, self.conf.numPhrases, 1).expand(-1, -1, self.conf.fieldFeaturesDim*2)
            x = x * mask
            x = x - (1-mask)

        if self.conf.fieldDropout > 0.:
            x = self.fieldDropout(x)

        x = torch.max(x, 1).values

        #doc level
        x = x.view(-1, self.conf.numFields, self.conf.fieldFeaturesDim*2)
        
        if self.conf.masking:
            mask = (x != -1).all(2).float()
        
        x,_ = self.docLayer(x)
        
        if self.conf.masking:
            mask = mask.view(-1, self.conf.numFields, 1).expand(-1, -1, self.conf.docFeaturesDim*2)
            x = x * mask
            x = x - (1-mask)
        
        if self.conf.docDropout > 0.:
            x = self.docDropout(x)
        x = torch.max(x, 1).values

        #classification
        if self.conf.featuresDropout > 0.:
            x = self.featuresDropout(x)

        for i in range(len(self.hiddenLayers)):
            x = self.hiddenLayers[i](x)
            x = F.relu(x)

        x = self.outLayer(x)
        if self.conf.outDim == 1:
            x = torch.sigmoid(x)
        else:
            x = torch.softmax(x,1)
            #TODO: log_softmax?

        if getAttention:
            return x,status
        else:
            return x


