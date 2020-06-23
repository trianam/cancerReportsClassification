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
        elif conf.rnnType == 'GRU':
            self.phraseLayer = nn.GRU(conf.vecLen, conf.phraseFeaturesDim, conf.phraseFeaturesLayers, bidirectional=True, batch_first=True)

        if conf.phraseDropout > 0.:
            self.phraseDropout = nn.Dropout(conf.phraseDropout)
        
        if conf.hiddenLayers > 0:
            if conf.hiddenDim is None:
                hiddenDim = conf.outDim
            else:
                hiddenDim = conf.hiddenDim
        else:
            hiddenDim = 2*conf.phraseFeaturesDim

        self.hiddenLayers = nn.ModuleList([])
        for _ in range(conf.hiddenLayers):
            self.hiddenLayers.append(nn.Linear(2*conf.phraseFeaturesDim, hiddenDim))

        self.outLayer = nn.Linear(hiddenDim, conf.outDim)
        
    def forward(self, x, batchIndex=None, getAttention=False):
        #phrase level
        
        x,_ = self.phraseLayer(x)

        if getAttention:
            statusPhrase = x.clone().detach()
            statusPhrase = statusPhrase.view(-1, self.conf.seqLen, 2*self.conf.phraseFeaturesDim)

        x = x[:,-1,:]
        
        if self.conf.phraseDropout > 0.:
            x = self.phraseDropout(x)

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
            return x, statusPhrase
        else:
            return x


