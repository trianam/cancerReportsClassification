import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, conf):
        super(MyModel, self).__init__()
        self.conf = conf

        if conf.rnnType == 'LSTM':
            self.phraseLayer = nn.LSTM(conf.vecLen, conf.phraseFeaturesDim, conf.phraseFeaturesLayers, bidirectional=True, batch_first=True)
        elif conf.rnnType == 'GRU':
            self.phraseLayer = nn.GRU(conf.vecLen, conf.phraseFeaturesDim, conf.phraseFeaturesLayers, bidirectional=True, batch_first=True)

        if conf.phraseDropout > 0.:
            self.phraseDropout = nn.Dropout(conf.phraseDropout)
        if conf.featuresDropout > 0.:
            self.featuresDropout = nn.Dropout(conf.featuresDropout)
       
        if self.conf.bidirectionalMerge is None:
            nextFeaturesDim = 2 * conf.phraseFeaturesDim
        else:
            nextFeaturesDim = conf.phraseFeaturesDim

        self.afterFeaturesLayers = nn.ModuleList([])
        if conf.afterFeaturesLayers > 0:
            for _ in range(conf.afterFeaturesLayers -1):
                self.afterFeaturesLayers.append(nn.Linear(nextFeaturesDim, conf.afterFeaturesDim))
                nextFeaturesDim = conf.afterFeaturesDim
                
            self.afterFeaturesLayers.append(nn.Linear(nextFeaturesDim, conf.afterFeaturesOutDim))
            nextFeaturesDim = conf.afterFeaturesOutDim

        if conf.classLayers > 0:
            if conf.hiddenDim is None:
                hiddenDim = conf.outDim
            else:
                hiddenDim = conf.hiddenDim

            self.hiddenLayers = nn.ModuleList([])
            for _ in range(conf.classLayers - 1):
                self.hiddenLayers.append(nn.Linear(nextFeaturesDim, hiddenDim))
                nextFeaturesDim = hiddenDim

            self.outLayer = nn.Linear(nextFeaturesDim, conf.outDim)
        
    def forward(self, x, batchIndex=None, getAttention=False):
        seqLen = x.shape[1]

        #phrase level
        if self.conf.masking:
            mask = (x != 0).all(2).float()

        x,_ = self.phraseLayer(x)

        if self.conf.bidirectionalMerge is None:
            nextFeaturesDim = 2*self.conf.phraseFeaturesDim
        else:
            x = x.view(-1, seqLen, 2, self.conf.phraseFeaturesDim)
            if self.conf.bidirectionalMerge == 'avg':
                x = (x[:,:,0] + x[:,:,1]) / 2.
            elif self.conf.bidirectionalMerge == 'max':
                x = torch.max(x, 2).values  
            
            nextFeaturesDim = self.conf.phraseFeaturesDim

        if self.conf.afterFeaturesLayers > 0:
            x = x.contiguous().view(-1, nextFeaturesDim)
            for i in range(len(self.afterFeaturesLayers) -1):
                x = self.afterFeaturesLayers[i](x)
                x = F.relu(x)
            x = self.afterFeaturesLayers[-1](x)
            x = torch.sigmoid(x)
            x = x.view(-1, seqLen, self.conf.afterFeaturesOutDim)
            nextFeaturesDim = self.conf.afterFeaturesOutDim

        if self.conf.masking:
            mask = mask.view(-1, seqLen, 1).expand(-1, -1, nextFeaturesDim)
            x = x * mask
            if self.conf.afterFeaturesLayers == 0:
                x = x - (1-mask)

        if getAttention:
            statusPhrase = x.clone().detach()
            #statusPhrase = statusPhrase.view(-1, self.conf.numFields, self.conf.numPhrases, self.conf.phraseLen, nextFeaturesDim)

        if self.conf.phraseDropout > 0.:
            x = self.phraseDropout(x)

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
            return x, statusPhrase
        else:
            return x


