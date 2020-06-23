import numpy as np

class MAPScorer (object):
    def __init__(self, k=0):
        self.k = k
    
    def _getLength(self, sorted_labels):
        length = self.k
        if length>len(sorted_labels) or length<=0:
            length = len(sorted_labels)
        return length

    def _ap(self, y_true, y_pred):
        sorted_labels = y_true[np.argsort(y_pred)[::-1]]
        nr_relevant = len([x for x in sorted_labels if x > 0])
        if nr_relevant == 0:
            return 0.0

        length = self._getLength(sorted_labels)
        ap = 0.0
        rel = 0

        for i in range(length):
            lab = sorted_labels[i]
            if lab >= 1:
                rel += 1
                ap += float(rel) / (i+1.0)
        ap /= nr_relevant
        return ap

    def classesScore(self, y_true, y_pred):
        numClasses = y_true.shape[1]
        mAp = 0.0
        
        for c in range(numClasses):
            mAp += self._ap(y_true[:,c], y_pred[:,c])

        mAp /= numClasses

        return mAp

    def samplesScore(self, y_true, y_pred):
        numSamples = len(y_true)
        mAp = 0.0
        
        for s in range(numSamples):
            mAp += self._ap(y_true[s], y_pred[s])

        mAp /= numSamples

        return mAp
