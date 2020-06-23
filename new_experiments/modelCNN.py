import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):

    def __init__(self, conf):
        super(MyModel, self).__init__()

        self.conf = conf

        if conf.preLayer:
            self.fc0 = nn.Linear(conf.vecLen, conf.preLayerDim)
            convDim = conf.preLayerDim
        else:
            convDim = conf.vecLen

        Ks = [3, 4, 5]
        self.convs1 = nn.ModuleList([nn.Conv2d(1, conf.numKernels, (K, convDim)) for K in Ks])
        self.dropout = nn.Dropout(conf.dropout)
        self.fc1 = nn.Linear(len(Ks) * conf.numKernels, conf.outDim)

    def forward(self, x, batchIndex=None, getAttention=False):
        # (N, W, D)
        if self.conf.preLayer:
            x = self.fc0(x)
        x = x.unsqueeze(1)  # (N, 1, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, nk, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, nk), ...]*len(Ks)
        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*nk)
        logit = self.fc1(x)  # (N, C)
        return logit

