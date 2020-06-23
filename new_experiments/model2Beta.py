import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, conf):
        super(MyModel, self).__init__()

        #TODO: add conf.afterFeaturesLayers on phrase and doc levels?
        #TODO: only featuresDim1 (what to do with more?)

        self.conf = conf

        if conf.rnnType == 'LSTM':
            self.phraseLayer = nn.LSTM(conf.vecLen, conf.phraseFeaturesDim, conf.phraseFeaturesLayers, bidirectional=True, batch_first=True)
            self.fieldLayer = nn.LSTM(conf.vecLen, conf.fieldFeaturesDim, conf.fieldFeaturesLayers, bidirectional=True, batch_first=True)
            self.docLayer = nn.LSTM(conf.vecLen, conf.docFeaturesDim, conf.docFeaturesLayers, bidirectional=True, batch_first=True)
        elif conf.rnnType == 'GRU':
            self.phraseLayer = nn.GRU(conf.vecLen, conf.phraseFeaturesDim, conf.phraseFeaturesLayers, bidirectional=True, batch_first=True)
            self.fieldLayer = nn.GRU(conf.vecLen, conf.fieldFeaturesDim, conf.fieldFeaturesLayers, bidirectional=True, batch_first=True)
            self.docLayer = nn.GRU(conf.vecLen, conf.docFeaturesDim, conf.docFeaturesLayers, bidirectional=True, batch_first=True)

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

        self.afterPhraseLayers = nn.ModuleList([])
        for _ in range(conf.afterPhraseLayers -1):
            self.afterPhraseLayers.append(nn.Linear(conf.phraseFeaturesDim*2, conf.phraseFeaturesDim*2))
        self.afterPhraseLayers.append(nn.Linear(conf.phraseFeaturesDim*2, 1))

        self.afterFieldLayers = nn.ModuleList([])
        for _ in range(conf.afterFieldLayers -1):
            self.afterFieldLayers.append(nn.Linear(conf.fieldFeaturesDim*2, conf.fieldFeaturesDim*2))
        self.afterFieldLayers.append(nn.Linear(conf.fieldFeaturesDim*2, 1))

        self.afterDocLayers = nn.ModuleList([])
        for _ in range(conf.afterDocLayers -1):
            self.afterDocLayers.append(nn.Linear(conf.docFeaturesDim*2, conf.docFeaturesDim*2))
        self.afterDocLayers.append(nn.Linear(conf.docFeaturesDim*2, 1))

        if conf.hiddenLayers > 0:
            if conf.hiddenDim is None:
                hiddenDim = conf.outDim
            else:
                hiddenDim = conf.hiddenDim
        else:
            hiddenDim = conf.vecLen

        self.hiddenLayers = nn.ModuleList([])
        for _ in range(conf.hiddenLayers):
            self.hiddenLayers.append(nn.Linear(conf.vecLen, hiddenDim))

        self.outLayer = nn.Linear(hiddenDim, conf.outDim)
        
        self.testLstmStatus = False

    def activateTestLstmStatus(self):
        self.testLstmStatus = True

    def deactivateTestLstmStatus(self):
        self.testLstmStatus = False

    def forward(self, x, batchIndex=None, getAttention=False):
        #phrase level
        x = x.view(-1, self.conf.phraseLen, self.conf.vecLen)
        
        if self.conf.masking:
            mask = (x != 0).all(2).float()

        att,_ = self.phraseLayer(x)

        if self.conf.afterFeaturesLayers > 0:
            att = att.contiguous().view(-1, self.conf.phraseFeaturesDim*2)
            for l in self.afterFeaturesLayers:
                att = l(att)
                att = F.relu(att)
            att = att.view(-1, self.conf.phraseLen, self.conf.phraseFeaturesDim*2)

        if self.conf.masking:
            mask = mask.view(-1, self.conf.phraseLen, 1).expand(-1, -1, self.conf.phraseFeaturesDim*2)
            att = att * mask
            att = att - (1-mask)

        if self.testLstmStatus:
            status = att.clone().detach()
            status = status.view(-1, self.conf.numFields, self.conf.numPhrases, self.conf.phraseLen, self.conf.phraseFeaturesDim*2)

        if self.conf.phraseDropout > 0.:
            att = self.phraseDropout(att)

        for l in self.afterPhraseLayers:
            att = l(att)
            #att = F.relu(att)
            #TODO: add function or softmax
        
        x = torch.sum(x*att, 1)#.values

        #field level
        x = x.view(-1, self.conf.numPhrases, self.conf.vecLen)

        if self.conf.masking:
            mask = (x != -1).all(2).float()

        att,_ = self.fieldLayer(x)

        if self.conf.masking:
            mask = mask.view(-1, self.conf.numPhrases, 1).expand(-1, -1, self.conf.fieldFeaturesDim*2)
            att = att * mask
            att = att - (1-mask)

        if self.conf.fieldDropout > 0.:
            att = self.fieldDropout(att)

        for l in self.afterFieldLayers:
            att = l(att)
            #att = F.relu(att)
       
        x = torch.sum(x*att, 1)#.values

        #doc level
        x = x.view(-1, self.conf.numFields, self.conf.vecLen)
        
        if self.conf.masking:
            mask = (x != -1).all(2).float()
        
        att,_ = self.docLayer(x)
        
        if self.conf.masking:
            mask = mask.view(-1, self.conf.numFields, 1).expand(-1, -1, self.conf.docFeaturesDim*2)
            att = att * mask
            att = att - (1-mask)
        
        if self.conf.docDropout > 0.:
            att = self.docDropout(att)
        
        for l in self.afterDocLayers:
            att = l(att)
            #att = F.relu(att)
        
        x = torch.sum(x*att, 1)#.values

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

        if self.testLstmStatus:
            return x,status
        else:
            return x








    def forwardaccio(self, x):
        #phrase level
        x = x.view(-1, self.conf.phraseLen, self.conf.vecLen)
        
        if self.conf.masking:
            mask = (x != 0).all(2).float()

        att,_ = self.phraseLayer(x)
        if self.conf.afterFeaturesLayers > 0:
            att = att.contiguous().view(-1, self.conf.phraseFeaturesDim*2)
            for i in range(len(self.afterFeaturesLayers)):
                att = self.afterFeaturesLayers[i](att)
                att = F.relu(att)
            att = att.view(-1, self.conf.phraseLen, self.conf.phraseFeaturesDim*2)
        if self.conf.masking:
            mask = mask.view(-1, self.conf.phraseLen, 1).expand(-1, -1, self.conf.phraseFeaturesDim*2)
            att = att * mask
            att = att - (1-mask)
        if self.testLstmStatus:
            status = att.clone().detach()
            status = status.view(-1, self.conf.numFields, self.conf.numPhrases, self.conf.phraseLen, self.conf.phraseFeaturesDim*2)
        if self.conf.phraseDropout > 0.:
            att = self.phraseDropout(att)

        x = torch.sum(x*att, 1).values

        #field level
        x = x.view(-1, self.conf.numPhrases, self.conf.phraseFeaturesDim*2)
        att,_ = self.fieldLayer(x)
        if self.conf.fieldDropout > 0.:
            att = self.fieldDropout(att)
        x = torch.sum(x*att, 1).values

        #doc level
        x = x.view(-1, self.conf.numFields, self.conf.fieldFeaturesDim*2)
        att,_ = self.docLayer(x)
        if self.conf.docDropout > 0.:
            att = self.docDropout(att)
        x = torch.sum(x*att, 1).values

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

        if self.testLstmStatus:
            return x,status
        else:
            return x


