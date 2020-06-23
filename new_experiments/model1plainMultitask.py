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

        #Site part
        nextFeaturesDimSite = nextFeaturesDim
        self.afterFeaturesLayersSite = nn.ModuleList([])
        if conf.afterFeaturesLayersSite > 0:
            for _ in range(conf.afterFeaturesLayersSite -1):
                self.afterFeaturesLayersSite.append(nn.Linear(nextFeaturesDimSite, conf.afterFeaturesDimSite))
                nextFeaturesDimSite = conf.afterFeaturesDimSite
                
            self.afterFeaturesLayersSite.append(nn.Linear(nextFeaturesDimSite, conf.afterFeaturesOutDimSite))
            nextFeaturesDimSite = conf.afterFeaturesOutDimSite

        if conf.classLayersSite > 0:
            if conf.hiddenDimSite is None:
                hiddenDimSite = conf.outDimSite
            else:
                hiddenDimSite = conf.hiddenDimSite

            self.hiddenLayersSite = nn.ModuleList([])
            for _ in range(conf.classLayersSite - 1):
                self.hiddenLayersSite.append(nn.Linear(nextFeaturesDimSite, hiddenDimSite))
                nextFeaturesDimSite = hiddenDimSite

            self.outLayerSite = nn.Linear(nextFeaturesDimSite, conf.outDimSite)
        

        #Morpho part
        nextFeaturesDimMorpho = nextFeaturesDim
        self.afterFeaturesLayersMorpho = nn.ModuleList([])
        if conf.afterFeaturesLayersMorpho > 0:
            for _ in range(conf.afterFeaturesLayersMorpho -1):
                self.afterFeaturesLayersMorpho.append(nn.Linear(nextFeaturesDimMorpho, conf.afterFeaturesDimMorpho))
                nextFeaturesDimMorpho = conf.afterFeaturesDimMorpho
                
            self.afterFeaturesLayersMorpho.append(nn.Linear(nextFeaturesDimMorpho, conf.afterFeaturesOutDimMorpho))
            nextFeaturesDimMorpho = conf.afterFeaturesOutDimMorpho

        if conf.classLayersMorpho > 0:
            if conf.hiddenDimMorpho is None:
                hiddenDimMorpho = conf.outDimMorpho
            else:
                hiddenDimMorpho = conf.hiddenDimMorpho

            self.hiddenLayersMorpho = nn.ModuleList([])
            for _ in range(conf.classLayersMorpho - 1):
                self.hiddenLayersMorpho.append(nn.Linear(nextFeaturesDimMorpho, hiddenDimMorpho))
                nextFeaturesDimMorpho = hiddenDimMorpho

            self.outLayerMorpho = nn.Linear(nextFeaturesDimMorpho, conf.outDimMorpho)
        
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

        #Site part
        nextFeaturesDimSite = nextFeaturesDim
        if self.conf.afterFeaturesLayersSite > 0:
            xSite = x.contiguous().view(-1, nextFeaturesDimSite)
            for i in range(len(self.afterFeaturesLayersSite) -1):
                xSite = self.afterFeaturesLayersSite[i](xSite)
                xSite = F.relu(xSite)
            xSite = self.afterFeaturesLayersSite[-1](xSite)
            xSite = torch.sigmoid(xSite)
            xSite = xSite.view(-1, seqLen, self.conf.afterFeaturesOutDimSite)
            nextFeaturesDimSite = self.conf.afterFeaturesOutDimSite

        if self.conf.masking:
            maskSite = mask.view(-1, seqLen, 1).expand(-1, -1, nextFeaturesDimSite)
            xSite = xSite * maskSite
            if self.conf.afterFeaturesLayersSite == 0:
                xSite = xSite - (1-maskSite)

        if getAttention:
            statusPhraseSite = xSite.clone().detach()
            #statusPhrase = statusPhrase.view(-1, self.conf.numFields, self.conf.numPhrases, self.conf.phraseLen, nextFeaturesDim)

        if self.conf.phraseDropout > 0.:
            xSite = self.phraseDropout(xSite)

        xSite = torch.max(xSite, 1).values

        #classification
        if self.conf.featuresDropout > 0.:
            xSite = self.featuresDropout(xSite)

        if self.conf.classLayersSite > 0:
            for i in range(len(self.hiddenLayersSite)):
                xSite = self.hiddenLayers[i](xSite)
                xSite = F.relu(xSite)

            xSite = self.outLayerSite(xSite)
            if self.conf.outDimSite == 1:
                xSite = torch.sigmoid(xSite)
            #else:
            #    #xSite = torch.softmax(xSite,1)
            #    xSite = torch.log_softmax(xSite,1)

        #Morpho part
        nextFeaturesDimMorpho = nextFeaturesDim
        if self.conf.afterFeaturesLayersMorpho > 0:
            xMorpho = x.contiguous().view(-1, nextFeaturesDimMorpho)
            for i in range(len(self.afterFeaturesLayersMorpho) -1):
                xMorpho = self.afterFeaturesLayersMorpho[i](xMorpho)
                xMorpho = F.relu(xMorpho)
            xMorpho = self.afterFeaturesLayersMorpho[-1](xMorpho)
            xMorpho = torch.sigmoid(xMorpho)
            xMorpho = xMorpho.view(-1, seqLen, self.conf.afterFeaturesOutDimMorpho)
            nextFeaturesDimMorpho = self.conf.afterFeaturesOutDimMorpho

        if self.conf.masking:
            maskMorpho = mask.view(-1, seqLen, 1).expand(-1, -1, nextFeaturesDimMorpho)
            xMorpho = xMorpho * maskMorpho
            if self.conf.afterFeaturesLayersMorpho == 0:
                xMorpho = xMorpho - (1-maskMorpho)

        if getAttention:
            statusPhraseMorpho = xMorpho.clone().detach()
            #statusPhrase = statusPhrase.view(-1, self.conf.numFields, self.conf.numPhrases, self.conf.phraseLen, nextFeaturesDim)

        if self.conf.phraseDropout > 0.:
            xMorpho = self.phraseDropout(xMorpho)

        xMorpho = torch.max(xMorpho, 1).values

        #classification
        if self.conf.featuresDropout > 0.:
            xMorpho = self.featuresDropout(xMorpho)

        if self.conf.classLayersMorpho > 0:
            for i in range(len(self.hiddenLayersMorpho)):
                xMorpho = self.hiddenLayers[i](xMorpho)
                xMorpho = F.relu(xMorpho)

            xMorpho = self.outLayerMorpho(xMorpho)
            if self.conf.outDimMorpho == 1:
                xMorpho = torch.sigmoid(xMorpho)
            #else:
            #    #xMorpho = torch.softmax(xMorpho,1)
            #    xMorpho = torch.log_softmax(xMorpho,1)

       
        if getAttention:
            return [xSite, xMorpho], [statusPhraseSite, statusPhraseMorpho]
        else:
            return [xSite, xMorpho]


