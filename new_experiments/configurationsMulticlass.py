import itertools
from conf import Conf

config1 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    16,
    "fieldFeaturesDim":     16,
    "docFeaturesDim":       16,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf1',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf1/epoch{}.pt',      #None or dir
})

config2 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 8,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    2,
    "fieldFeaturesDim":     2,
    "docFeaturesDim":       2,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf2',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf2/epoch{}.pt',      #None or dir
})

config2continue = config2.copy({
    "startEpoch":           100,
    "epochs":               300,
    "modelFileLoad":        'modelsPytorchMulticlass/conf2/epoch99.pt',      #None or dir
})

config3 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    32,
    "fieldFeaturesDim":     32,
    "docFeaturesDim":       32,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf3',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf3/epoch{}.pt',      #None or dir
})

config4 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    64,
    "fieldFeaturesDim":     64,
    "docFeaturesDim":       64,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            64,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf4',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf4/epoch{}.pt',      #None or dir
})

config5 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    128,
    "fieldFeaturesDim":     128,
    "docFeaturesDim":       128,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            128,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf5',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf5/epoch{}.pt',      #None or dir
})

config6 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    256,
    "fieldFeaturesDim":     256,
    "docFeaturesDim":       256,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            256,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf6',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf6/epoch{}.pt',      #None or dir
})

#valid acc 90
config7 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    512,
    "fieldFeaturesDim":     256,
    "docFeaturesDim":       256,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            32,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf7',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf7/epoch{}.pt',      #None or dir
})

config7folds = {}
config7folds[0] = config7
for f in range(1,10):
    config7folds[f] = config7.copy({
        "fold":                 f,
        "tensorboard":          'tensorBoardPytorchMulticlass/conf7fold{}'.format(f),
        "modelFile":            'modelsPytorchMulticlass/conf7/fold{}/epoch{{}}.pt'.format(f),
    })

config7plain = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1plain',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "phraseFeaturesDim":    512,
    "phraseDropout":        0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            32,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf7plain',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf7plain/epoch{}.pt',      #None or dir
})

config7plainFolds = {}
config7plainFolds[0] = config7plain
for f in range(1,10):
    config7plainFolds[f] = config7plain.copy({
        "fold":                 f,
        "tensorboard":          'tensorBoardPytorchMulticlass/conf7plainFold{}'.format(f),
        "modelFile":            'modelsPytorchMulticlass/conf7plain/fold{}/epoch{{}}.pt'.format(f),
    })

config7plainDims = {}
config7plainDims[512] = config7plain
for d in [384,256,192,128,64,32]:
    config7plainDims[d] = config7plain.copy({
        "phraseFeaturesDim":    d,
        "tensorboard":          'tensorBoardPytorchMulticlass/conf7plainDim{}'.format(d),
        "modelFile":            'modelsPytorchMulticlass/conf7plainDim{}/epoch{{}}.pt'.format(d),
    })

config7plain2Folds = {}
config7plain2Folds[0] = config7plainDims[256] 
for f in range(1,10):
    config7plain2Folds[f] = config7plainDims[256].copy({
        "fold":                 f,
        "tensorboard":          'tensorBoardPytorchMulticlass/conf7plain2Fold{}'.format(f),
        "modelFile":            'modelsPytorchMulticlass/conf7plain2/fold{}/epoch{{}}.pt'.format(f),
    })

config7plain2maxFolds = {}
for f in range(0,10):
    config7plain2maxFolds[f] = config7plain2Folds[f].copy({
        "bidirectionalMerge":   'max',      #avg or max
        "tensorboard":          'tensorBoardPytorchMulticlass/conf7plain2maxFold{}'.format(f),
        "modelFile":            'modelsPytorchMulticlass/conf7plain2max/fold{}/epoch{{}}.pt'.format(f),
    })

config7dropout = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    512,
    "fieldFeaturesDim":     256,
    "docFeaturesDim":       256,
    "phraseDropout":        0.5,
    "fieldDropout":         0.5,
    "docDropout":           0.5,
    "featuresDropout":      0.5,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            32,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf7dropout',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf7dropout/epoch{}.pt',      #None or dir
})


config7dropout2 = config7dropout.copy({
    "phraseDropout":        0.5,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "tensorboard":          'tensorBoardPytorchMulticlass/conf7dropout2',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf7dropout2/epoch{}.pt',      #None or dir
})

config7dropout3 = config7dropout.copy({
    "phraseDropout":        0.2,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "tensorboard":          'tensorBoardPytorchMulticlass/conf7dropout3',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf7dropout3/epoch{}.pt',      #None or dir
})


config7base = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'modelBase',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "phraseFeaturesDim":    512,
    "phraseDropout":        0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "hiddenLayers":         0,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf7base',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf7base/epoch{}.pt',      #None or dir
})

config7baseFolds = {}
config7baseFolds[0] = config7base
for f in range(1,10):
    config7baseFolds[f] = config7base.copy({
        "fold":                 f,
        "tensorboard":          'tensorBoardPytorchMulticlass/conf7baseFold{}'.format(f),
        "modelFile":            'modelsPytorchMulticlass/conf7base/fold{}/epoch{{}}.pt'.format(f),
    })

config7baseNoAltre = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'modelBase',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "phraseFeaturesDim":    512,
    "phraseDropout":        0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "hiddenLayers":         0,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               54,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare-noAltre.p',
    "fileValues":           'corpusMMI/sede1-values-noAltre.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass-noAltre.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf7baseNoAltre',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf7baseNoAltre/epoch{}.pt',      #None or dir
})


config8base = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'modelBase',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "phraseFeaturesDim":    1024,
    "phraseDropout":        0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "hiddenLayers":         0,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf8base',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf8base/epoch{}.pt',      #None or dir
})

config8baseFolds = {}
config8baseFolds[0] = config8base
for f in range(1,10):
    config8baseFolds[f] = config8base.copy({
        "fold":                 f,
        "tensorboard":          'tensorBoardPytorchMulticlass/conf8baseFold{}'.format(f),
        "modelFile":            'modelsPytorchMulticlass/conf8base/fold{}/epoch{{}}.pt'.format(f),
    })


#val acc 90
config8 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    512,
    "fieldFeaturesDim":     512,
    "docFeaturesDim":       512,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            32,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf8',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf8/epoch{}.pt',      #None or dir
})

config8folds = {}
config8folds[0] = config8
for f in range(1,10):
    config8folds[f] = config8.copy({
        "fold":                 f,
        "tensorboard":          'tensorBoardPytorchMulticlass/conf8fold{}'.format(f),
        "modelFile":            'modelsPytorchMulticlass/conf8/fold{}/epoch{{}}.pt'.format(f),
    })

config9 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    512,
    "fieldFeaturesDim":     512,
    "docFeaturesDim":       512,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "hiddenLayers":         1,
    "hiddenDim":            256,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            32,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf9',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf9/epoch{}.pt',      #None or dir
})

config10 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    1024,
    "fieldFeaturesDim":     1024,
    "docFeaturesDim":       1024,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            16,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf10',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf10/epoch{}.pt',      #None or dir
})

config11 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 2,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    512,
    "fieldFeaturesDim":     512,
    "docFeaturesDim":       512,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            32,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf11',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf11/epoch{}.pt',      #None or dir
})

config12 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 4,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    512,
    "fieldFeaturesDim":     512,
    "docFeaturesDim":       512,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            16,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf12',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf12/epoch{}.pt',      #None or dir
})

config13 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    512,
    "fieldFeaturesDim":     4,
    "docFeaturesDim":       4,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            32,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf13',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf13/epoch{}.pt',      #None or dir
})

config14 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    512,
    "fieldFeaturesDim":     16,
    "docFeaturesDim":       16,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            32,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf14',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf14/epoch{}.pt',      #None or dir
})

config15 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    512,
    "fieldFeaturesDim":     8,
    "docFeaturesDim":       8,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            32,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf15',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf15/epoch{}.pt',      #None or dir
})

config16 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    512,
    "fieldFeaturesDim":     4,
    "docFeaturesDim":       4,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            32,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf16',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf16/epoch{}.pt',      #None or dir
})

config17 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    512,
    "fieldFeaturesDim":     2,
    "docFeaturesDim":       2,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            32,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf17',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf17/epoch{}.pt',      #None or dir
})

config18 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    256,
    "fieldFeaturesDim":     2,
    "docFeaturesDim":       2,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            32,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf18',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf18/epoch{}.pt',      #None or dir
})

config19 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    512,
    "fieldFeaturesDim":     256,
    "docFeaturesDim":       128,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            32,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf19',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf19/epoch{}.pt',      #None or dir
})

config20 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'modelMLP',   #model1 or model2
    "dropout":              0.,
    "hiddenLayers":         2,
    "hiddenDim":            4096,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            32,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf20',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf20/epoch{}.pt',      #None or dir
})

config21 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'modelMLP',   #model1 or model2
    "dropout":              0.3,
    "hiddenLayers":         2,
    "hiddenDim":            4096,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            32,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf21',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf21/epoch{}.pt',      #None or dir
})

config21continue = config21.copy({
    "startEpoch":           17,
    "epochs":               83,
    "modelFileLoad":        'modelsPytorchMulticlass/conf21/epoch16.pt',      #None or dir
})

config22 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'modelMLP',   #model1 or model2
    "dropout":              0.5,
    "hiddenLayers":         2,
    "hiddenDim":            4096,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            32,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf22',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf22/epoch{}.pt',      #None or dir
})

config22continue = config22.copy({
    "startEpoch":           3,
    "epochs":               97,
    "modelFileLoad":        'modelsPytorchMulticlass/conf22/epoch2.pt',      #None or dir
})

#interpretable (to run)
config23 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1plain',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "phraseFeaturesDim":    512,
    "phraseDropout":        0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   None,      #avg or max
    "afterFeaturesLayers":  2,
    "afterFeaturesDim":     64,
    "afterFeaturesOutDim":  67,
    "classLayers":          0,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            32,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf23',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf23/epoch{}.pt',      #None or dir
})

#73.7 valid
config24 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1plain',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "phraseFeaturesDim":    512,
    "phraseDropout":        0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   None,      #avg or max
    "afterFeaturesLayers":  2,
    "afterFeaturesDim":     128,
    "afterFeaturesOutDim":  67,
    "classLayers":          0,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            32,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf24',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf24/epoch{}.pt',      #None or dir
})

config25 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1plain',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "phraseFeaturesDim":    512,
    "phraseDropout":        0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   None,      #avg or max
    "afterFeaturesLayers":  2,
    "afterFeaturesDim":     256,
    "afterFeaturesOutDim":  67,
    "classLayers":          0,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            32,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        10,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf25',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf25/epoch{}.pt',      #None or dir
})

#valid 84.4
config26 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1plain',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "phraseFeaturesDim":    512,
    "phraseDropout":        0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   None,      #avg or max
    "afterFeaturesLayers":  1,
    "afterFeaturesDim":     0,
    "afterFeaturesOutDim":  67,
    "classLayers":          0,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            32,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf26',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf26/epoch{}.pt',      #None or dir
    "modelFileLoad":        'modelsPytorchMulticlass/conf26/epoch8.pt',      #None or dir
})

config26continue = config26.copy({
    "startEpoch":           15,
    "epochs":               85,
    "earlyStopping":        10,                   #None or patience
    "modelFileLoad":        'modelsPytorchMulticlass/conf26/epoch14.pt',      #None or dir
})

config27 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1plain',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "phraseFeaturesDim":    512,
    "phraseDropout":        0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   None,      #avg or max
    "afterFeaturesLayers":  3,
    "afterFeaturesDim":     128,
    "afterFeaturesOutDim":  67,
    "classLayers":          0,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            32,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf27',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf27/epoch{}.pt',      #None or dir
})

#val acc 86.7
config28 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1plain',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "phraseFeaturesDim":    256,
    "phraseDropout":        0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   None,      #avg or max
    "afterFeaturesLayers":  1,
    "afterFeaturesDim":     0,
    "afterFeaturesOutDim":  67,
    "classLayers":          0,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            32,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        10,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf28',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf28/epoch{}.pt',      #None or dir
    "modelFileLoad":        'modelsPytorchMulticlass/conf28/epoch9.pt',      #None or dir
})

#val loss 87.7
config29 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1plain',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "phraseFeaturesDim":    128,
    "phraseDropout":        0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   None,      #avg or max
    "afterFeaturesLayers":  1,
    "afterFeaturesDim":     0,
    "afterFeaturesOutDim":  67,
    "classLayers":          0,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            32,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        10,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf29',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf29/epoch{}.pt',      #None or dir
})

config29Folds = {}
config29Folds[0] = config29
for f in range(1,10):
    config29Folds[f] = config29.copy({
        "fold":                 f,
        "tensorboard":          'tensorBoardPytorchMulticlass/conf29Fold{}'.format(f),
        "modelFile":            'modelsPytorchMulticlass/conf29/fold{}/epoch{{}}.pt'.format(f),
    })

config29Folds2Continue = config29.copy({
    "fold":                 2,
    "startEpoch":           45,
    "epochs":               55,
    "tensorboard":          'tensorBoardPytorchMulticlass/conf29Fold2',
    "modelFile":            'modelsPytorchMulticlass/conf29/fold2/epoch{}.pt',
    "modelFileLoad":        'modelsPytorchMulticlass/conf29/fold2/epoch9.pt',      #None or dir
})

#val acc 87.7
config30 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1plain',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "phraseFeaturesDim":    64,
    "phraseDropout":        0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   None,      #avg or max
    "afterFeaturesLayers":  1,
    "afterFeaturesDim":     0,
    "afterFeaturesOutDim":  67,
    "classLayers":          0,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            32,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        10,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf30',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf30/epoch{}.pt',      #None or dir
})

config30Folds = {}
config30Folds[0] = config30
for f in range(1,10):
    config30Folds[f] = config30.copy({
        "fold":                 f,
        "tensorboard":          'tensorBoardPytorchMulticlass/conf30Fold{}'.format(f),
        "modelFile":            'modelsPytorchMulticlass/conf30/fold{}/epoch{{}}.pt'.format(f),
    })

config30v = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1plain',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "phraseFeaturesDim":    64,
    "phraseDropout":        0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   None,      #avg or max
    "afterFeaturesLayers":  1,
    "afterFeaturesDim":     0,
    "afterFeaturesOutDim":  128,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            32,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        10,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf30v',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf30v/epoch{}.pt',      #None or dir
    "modelFileLoad":        'modelsPytorchMulticlass/conf30v/epoch4.pt',      #None or dir
})

config30vFolds = {}
config30vFolds[0] = config30v
for f in range(1,10):
    config30vFolds[f] = config30v.copy({
        "fold":                 f,
        "tensorboard":          'tensorBoardPytorchMulticlass/conf30vFold{}'.format(f),
        "modelFile":            'modelsPytorchMulticlass/conf30v/fold{}/epoch{{}}.pt'.format(f),
        "modelFileLoad":        None,      #None or dir
    })


#val acc 87.6
config31 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1plain',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "phraseFeaturesDim":    67,
    "phraseDropout":        0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   None,      #avg or max
    "afterFeaturesLayers":  1,
    "afterFeaturesDim":     0,
    "afterFeaturesOutDim":  67,
    "classLayers":          0,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare.p',
    "fileValues":           'corpusMMI/sede1-values.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass.p',
    "batchSize":            32,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        10,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf31',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf31/epoch{}.pt',      #None or dir
})


config30vNoAltre = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1plain',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "phraseFeaturesDim":    64,
    "phraseDropout":        0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   None,      #avg or max
    "afterFeaturesLayers":  1,
    "afterFeaturesDim":     0,
    "afterFeaturesOutDim":  128,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               54,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFiltered-multiclass-noRare-noAltre.p',
    "fileValues":           'corpusMMI/sede1-values-noAltre.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-multiclass-noAltre.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        10,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/conf30vNoAltre',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/conf30vNoAltre/epoch{}.pt',      #None or dir
})

configT1 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1plain',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "phraseFeaturesDim":    64,
    "phraseDropout":        0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   None,      #avg or max
    "afterFeaturesLayers":  1,
    "afterFeaturesDim":     0,
    "afterFeaturesOutDim":  128,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               68,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusTemporal/corpusTemporal.p',
    "fileValues":           'corpusTemporal/valuesTemporal.p',
    "fileVectors":          'corpusTemporal/vectors.txt',
    "corpusKey":            'sede1',
    "batchSize":            32,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        10,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/confT1',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/confT1/epoch{}.pt',      #None or dir
    #"modelFileLoad":        'modelsPytorchMulticlass/confT1/epoch4.pt',      #None or dir
})

configTbase = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'modelBase',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "phraseFeaturesDim":    512,
    "phraseDropout":        0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "hiddenLayers":         0,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               68,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusTemporal/corpusTemporal.p',
    "fileValues":           'corpusTemporal/valuesTemporal.p',
    "fileVectors":          'corpusTemporal/vectors.txt',
    "corpusKey":            'sede1',
    "batchSize":            32,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchMulticlass/confTbase',    #None or logdir
    "modelFile":            'modelsPytorchMulticlass/confTbase/epoch{}.pt',      #None or dir
})

configTbaseMulti = {}
for pfl in [1,2,4]:
    configTbaseMulti[pfl] = {}
    for pfd in [128,256,512,1024]:
        configTbaseMulti[pfl][pfd] = {}
        for rt in ['GRU','LSTM']:
            configTbaseMulti[pfl][pfd][rt] = Conf({
                "numFields":            3,
                "numPhrases":           10,
                "phraseLen":            100,
                "vecLen":               60,
                "modelType":            'modelBase',   #model1 or model2
                "masking":              True,
                "phraseFeaturesLayers": pfl,
                "phraseFeaturesDim":    pfd,
                "phraseDropout":        0.,
                "rnnType":              rt,                 #LSTM or GRU
                "hiddenLayers":         0,
                "hiddenDim":            0,                     #None equals outDim
                "outDim":               68,
                "learningRate":         0.001,
                "learningRateDecay":    0.,
                "loss":                 'categorical_crossentropy',
                "dataMethod":           'process',
                "fileCorpus":           'corpusTemporal/corpusTemporal.p',
                "fileValues":           'corpusTemporal/valuesTemporal.p',
                "fileVectors":          'corpusTemporal/vectors.txt',
                "corpusKey":            'sede1',
                "batchSize":            32,
                "startEpoch":           0,
                "epochs":               100,
                "earlyStopping":        10,                   #None or patience
                "tensorboard":          'tensorBoardPytorchMulticlass/confTbaseMulti/{}/{}/{}'.format(pfl,pfd,rt),    #None or logdir
                "modelSave":            "all",
                "modelFile":            'modelsPytorchMulticlass/confTbaseMulti/{}/{}/{}/epoch{{}}.pt'.format(pfl,pfd,rt),      #None or dir
            })

configTbaseMultiBest = configTbaseMulti[2][256]['GRU'].copy({
            "modelFileLoad":        'modelsPytorchMulticlass/confTbaseMulti/2/256/GRU/epoch4.pt',      #None or dir
        })

configTmulti1 = {}
for pfd in [2,4,8,16,32,64,128,256,512]:
    configTmulti1[pfd] = {}
    for afl in [1,2,4]:
        configTmulti1[pfd][afl] = {}
        for afd in [2,4,8,16,32,64,128,256,512,1024,2048]:
            configTmulti1[pfd][afl][afd] = Conf({
                "numFields":            3,
                "numPhrases":           10,
                "phraseLen":            100,
                "vecLen":               60,
                "modelType":            'model1plain',   #model1 or model2
                "masking":              True,
                "phraseFeaturesLayers": 1,
                "phraseFeaturesDim":    pfd,
                "phraseDropout":        0.,
                "featuresDropout":      0.,
                "rnnType":              'GRU',                 #LSTM or GRU
                "bidirectionalMerge":   None,      #avg or max
                "afterFeaturesLayers":  afl,
                "afterFeaturesDim":     afd,
                "afterFeaturesOutDim":  afd,
                "classLayers":          1,
                "hiddenDim":            0,                     #None equals outDim
                "outDim":               68,
                "learningRate":         0.001,
                "learningRateDecay":    0.,
                "loss":                 'categorical_crossentropy',
                "dataMethod":           'process',
                "fileCorpus":           'corpusTemporal/corpusTemporal.p',
                "fileValues":           'corpusTemporal/valuesTemporal.p',
                "fileVectors":          'corpusTemporal/vectors.txt',
                "corpusKey":            'sede1',
                "batchSize":            32,
                "startEpoch":           0,
                "epochs":               100,
                "earlyStopping":        10,                   #None or patience
                "tensorboard":          'tensorBoardPytorchMulticlass/confTmulti1/{}/{}/{}'.format(pfd, afl, afd),    #None or logdir
                "modelSave":            "all",
                "modelFile":            'modelsPytorchMulticlass/confTmulti1/{}/{}/{}/epoch{{}}.pt'.format(pfd, afl, afd),      #None or dir
                #"modelFileLoad":        'modelsPytorchMulticlass/confT1/epoch4.pt',      #None or dir
            })

configTmulti1best = configTmulti1[128][1][512].copy({
            "modelFileLoad":        'modelsPytorchMulticlass/confTmulti1/128/1/512/epoch5.pt',      #None or dir
        })
 
configTmulti1Flair = configTmulti1best.copy({
        "fileVectors":          'flair/vectors.txt',
        "vecLen":               2048,
        "tensorboard":          'tensorBoardPytorchMulticlass/confTmulti1Flair',    #None or logdir
        "modelSave":            "best",
        "modelFile":            'modelsPytorchMulticlass/confTmulti1Flair/best.pt',      #None or dir
        "modelFileLoad":        'modelsPytorchMulticlass/confTmulti1Flair/best.pt',      #None or dir
    })

configTmulti1GloveVecSize = {}
for vs in [40,50,70,80,90,100,200,300]:
    configTmulti1GloveVecSize[vs] = configTmulti1best.copy({
            "vecLen":               vs,
            "fileVectors":          'corpusTemporal/vectors-vs{}.txt'.format(vs),
            "tensorboard":          'tensorBoardPytorchMulticlass/confTmulti1GloveVecSize/{}'.format(vs),    #None or logdir
            "modelSave":            "best",
            "modelFile":            'modelsPytorchMulticlass/confTmulti1GloveVecSize/{}/best.pt'.format(vs),      #None or dir
            "modelFileLoad":        'modelsPytorchMulticlass/confTmulti1GloveVecSize/{}/best.pt'.format(vs),      #None or dir
        })

configTmulti1GloveWinSize = {}
for ws in [2,5,10,20]:
    configTmulti1GloveWinSize[ws] = configTmulti1best.copy({
            "fileVectors":          'corpusTemporal/vectors-ws{}.txt'.format(ws),
            "tensorboard":          'tensorBoardPytorchMulticlass/confTmulti1GloveWinSize/{}'.format(ws),    #None or logdir
            "modelSave":            "best",
            "modelFile":            'modelsPytorchMulticlass/confTmulti1GloveWinSize/{}/best.pt'.format(ws),      #None or dir
            "modelFileLoad":        'modelsPytorchMulticlass/confTmulti1GloveWinSize/{}/best.pt'.format(ws),      #None or dir
        })

configTmulti1GloveVecSizeWinSize = {}
for vs in [40,50,70,80,90,100,200,300]:
    configTmulti1GloveVecSizeWinSize[vs] = {}
    for ws in [2,5,10,20]:
        configTmulti1GloveVecSizeWinSize[vs][ws] = configTmulti1best.copy({
                "vecLen":               vs,
                "fileVectors":          'corpusTemporal/vectors-vs{}-ws{}.txt'.format(vs,ws),
                "tensorboard":          'tensorBoardPytorchMulticlass/confTmulti1GloveVecSizeWinSize/{}/{}'.format(vs,ws),    #None or logdir
                "modelSave":            "best",
                "modelFile":            'modelsPytorchMulticlass/confTmulti1GloveVecSizeWinSize/{}/{}/best.pt'.format(vs,ws),      #None or dir
                "modelFileLoad":        'modelsPytorchMulticlass/confTmulti1GloveVecSizeWinSize/{}/{}/best.pt'.format(vs,ws),      #None or dir
            })


configTmulti2 = {}
for pfd in [64,128,256,512,1024]:
    configTmulti2[pfd] = {}
    for afl in [1,2]:
        configTmulti2[pfd][afl] = {}
        for afd in [64,128,256,512,1024]:
            configTmulti2[pfd][afl][afd] = Conf({
                "numFields":            3,
                "numPhrases":           10,
                "phraseLen":            100,
                "vecLen":               60,
                "modelType":            'model1plain',   #model1 or model2
                "masking":              True,
                "phraseFeaturesLayers": 2,
                "phraseFeaturesDim":    pfd,
                "phraseDropout":        0.,
                "featuresDropout":      0.,
                "rnnType":              'GRU',                 #LSTM or GRU
                "bidirectionalMerge":   None,      #avg or max
                "afterFeaturesLayers":  afl,
                "afterFeaturesDim":     afd,
                "afterFeaturesOutDim":  afd,
                "classLayers":          1,
                "hiddenDim":            0,                     #None equals outDim
                "outDim":               68,
                "learningRate":         0.001,
                "learningRateDecay":    0.,
                "loss":                 'categorical_crossentropy',
                "dataMethod":           'process',
                "fileCorpus":           'corpusTemporal/corpusTemporal.p',
                "fileValues":           'corpusTemporal/valuesTemporal.p',
                "fileVectors":          'corpusTemporal/vectors.txt',
                "corpusKey":            'sede1',
                "batchSize":            32,
                "startEpoch":           0,
                "epochs":               100,
                "earlyStopping":        10,                   #None or patience
                "tensorboard":          'tensorBoardPytorchMulticlass/confTmulti2/{}/{}/{}'.format(pfd, afl, afd),    #None or logdir
                "modelFile":            'modelsPytorchMulticlass/confTmulti2/{}/{}/{}/epoch{{}}.pt'.format(pfd, afl, afd),      #None or dir
                #"modelFileLoad":        'modelsPytorchMulticlass/confT1/epoch4.pt',      #None or dir
            })

configTmultiCNN = {}
for nk in [16,32,64,128,256,512]:
    configTmultiCNN[nk] = Conf({
                "numFields":            3,
                "numPhrases":           10,
                "phraseLen":            100,
                "vecLen":               60,
                "modelType":            'modelCNN',   #model1 or model2
                "preLayer":             False,
                "numKernels":           nk,
                "dropout":              0.5,
                "outDim":               68,
                "learningRate":         0.001,
                "learningRateDecay":    0.,
                "loss":                 'categorical_crossentropy',
                "dataMethod":           'process',
                "fileCorpus":           'corpusTemporal/corpusTemporal.p',
                "fileValues":           'corpusTemporal/valuesTemporal.p',
                "fileVectors":          'corpusTemporal/vectors.txt',
                "corpusKey":            'sede1',
                "batchSize":            32,
                "startEpoch":           0,
                "epochs":               100,
                "earlyStopping":        10,                   #None or patience
                "tensorboard":          'tensorBoardPytorchMulticlass/confTmultiCNN/{}'.format(nk),    #None or logdir
                "modelSave":            "best",
                "modelFile":            'modelsPytorchMulticlass/confTmultiCNN/{}/epoch{{}}.pt'.format(nk),      #None or dir
            })

configTmultiCNNnd = {}
for nk in [16,32,64,128,256,512]:
    configTmultiCNNnd[nk] = Conf({
                "numFields":            3,
                "numPhrases":           10,
                "phraseLen":            100,
                "vecLen":               60,
                "modelType":            'modelCNN',   #model1 or model2
                "preLayer": False,
                "numKernels":           nk,
                "dropout":              0.,
                "outDim":               68,
                "learningRate":         0.001,
                "learningRateDecay":    0.,
                "loss":                 'categorical_crossentropy',
                "dataMethod":           'process',
                "fileCorpus":           'corpusTemporal/corpusTemporal.p',
                "fileValues":           'corpusTemporal/valuesTemporal.p',
                "fileVectors":          'corpusTemporal/vectors.txt',
                "corpusKey":            'sede1',
                "batchSize":            32,
                "startEpoch":           0,
                "epochs":               100,
                "earlyStopping":        10,                   #None or patience
                "tensorboard":          'tensorBoardPytorchMulticlass/confTmultiCNNnd/{}'.format(nk),    #None or logdir
                "modelSave":            "best",
                "modelFile":            'modelsPytorchMulticlass/confTmultiCNNnd/{}/epoch{{}}.pt'.format(nk),      #None or dir
            })

configTmultiCNNp = {}
for nk in [64,128,256,512]:
    configTmultiCNNp[nk] = {}
    for pd in [64,128,256,512]:
        configTmultiCNNp[nk][pd] = Conf({
                "numFields":            3,
                "numPhrases":           10,
                "phraseLen":            100,
                "vecLen":               60,
                "modelType":            'modelCNN',   #model1 or model2
                "preLayer":             True,
                "preLayerDim":          pd,
                "numKernels":           nk,
                "dropout":              0.5,
                "outDim":               68,
                "learningRate":         0.001,
                "learningRateDecay":    0.,
                "loss":                 'categorical_crossentropy',
                "dataMethod":           'process',
                "fileCorpus":           'corpusTemporal/corpusTemporal.p',
                "fileValues":           'corpusTemporal/valuesTemporal.p',
                "fileVectors":          'corpusTemporal/vectors.txt',
                "corpusKey":            'sede1',
                "batchSize":            32,
                "startEpoch":           0,
                "epochs":               100,
                "earlyStopping":        10,                   #None or patience
                "tensorboard":          'tensorBoardPytorchMulticlass/confTmultiCNNp/{}/{}'.format(nk,pd),    #None or logdir
                "modelSave":            "best",
                "modelFile":            'modelsPytorchMulticlass/confTmultiCNNp/{}/{}/best.pt'.format(nk,pd),      #None or dir
            })

configTmultiCNNpBest = configTmultiCNNp[128][64]



configTImulti1 = {}
for pfd in [2,4,8,16,32,64,128,256,512]:
    configTImulti1[pfd] = {}
    for afl in [1,2,4]:
        configTImulti1[pfd][afl] = {}
        for afd in [2,4,8,16,32,64,128,256,512,1024,2048]:
            #ALERT: with afl==1, afd is meaningless
            if afl==1 and afd > 2:
                continue
            configTImulti1[pfd][afl][afd] = Conf({
                "numFields":            3,
                "numPhrases":           10,
                "phraseLen":            100,
                "vecLen":               60,
                "modelType":            'model1plain',   #model1 or model2
                "masking":              True,
                "phraseFeaturesLayers": 1,
                "phraseFeaturesDim":    pfd,
                "phraseDropout":        0.,
                "featuresDropout":      0.,
                "rnnType":              'GRU',                 #LSTM or GRU
                "bidirectionalMerge":   None,      #avg or max
                "afterFeaturesLayers":  afl,
                "afterFeaturesDim":     afd,
                "afterFeaturesOutDim":  68,
                "classLayers":          0,
                "hiddenDim":            0,                     #None equals outDim
                "outDim":               68,
                "learningRate":         0.001,
                "learningRateDecay":    0.,
                "loss":                 'categorical_crossentropy',
                "dataMethod":           'process',
                "fileCorpus":           'corpusTemporal/corpusTemporal.p',
                "fileValues":           'corpusTemporal/valuesTemporal.p',
                "fileVectors":          'corpusTemporal/vectors.txt',
                "corpusKey":            'sede1',
                "batchSize":            32,
                "startEpoch":           0,
                "epochs":               100,
                "earlyStopping":        10,                   #None or patience
                "tensorboard":          'tensorBoardPytorchMulticlass/confTImulti1/{}/{}/{}'.format(pfd, afl, afd),    #None or logdir
                "modelFile":            'modelsPytorchMulticlass/confTImulti1/{}/{}/{}/epoch{{}}.pt'.format(pfd, afl, afd),      #None or dir
                #"modelFileLoad":        'modelsPytorchMulticlass/confT1/epoch4.pt',      #None or dir
            })


configTImulti1best = configTImulti1[256][1][2].copy({
        "modelFileLoad":        'modelsPytorchMulticlass/confTImulti1/256/1/2/epoch52.pt',      #None or dir
    })

configTImulti2 = {}
for pfl in [2,4]:
    configTImulti2[pfl] = {}
    for pfd in [64,128,256,512]:
        configTImulti2[pfl][pfd] = Conf({
            "numFields":            3,
            "numPhrases":           10,
            "phraseLen":            100,
            "vecLen":               60,
            "modelType":            'model1plain',   #model1 or model2
            "masking":              True,
            "phraseFeaturesLayers": pfl,
            "phraseFeaturesDim":    pfd,
            "phraseDropout":        0.,
            "featuresDropout":      0.,
            "rnnType":              'GRU',                 #LSTM or GRU
            "bidirectionalMerge":   None,      #avg or max
            "afterFeaturesLayers":  1,
            "afterFeaturesDim":     0,
            "afterFeaturesOutDim":  68,
            "classLayers":          0,
            "hiddenDim":            0,                     #None equals outDim
            "outDim":               68,
            "learningRate":         0.001,
            "learningRateDecay":    0.,
            "loss":                 'categorical_crossentropy',
            "dataMethod":           'process',
            "fileCorpus":           'corpusTemporal/corpusTemporal.p',
            "fileValues":           'corpusTemporal/valuesTemporal.p',
            "fileVectors":          'corpusTemporal/vectors.txt',
            "corpusKey":            'sede1',
            "batchSize":            32,
            "startEpoch":           0,
            "epochs":               100,
            "earlyStopping":        10,                   #None or patience
            "tensorboard":          'tensorBoardPytorchMulticlass/confTImulti2/{}/{}'.format(pfl, pfd),    #None or logdir
            "modelSave":            "all",
            "modelFile":            'modelsPytorchMulticlass/confTImulti2/{}/{}/epoch{{}}.pt'.format(pfl,pfd),      #None or dir
            #"modelFileLoad":        'modelsPytorchMulticlass/confT1/epoch4.pt',      #None or dir
        })

configTImulti2best = configTImulti2[2][128].copy({
        "modelFileLoad":        'modelsPytorchMulticlass/confTImulti2/2/128/epoch35.pt',      #None or dir
    })












configTMbaseMulti = {}
for pfl in [1,2,4,8]:
    configTMbaseMulti[pfl] = {}
    for pfd in [64,128,256,512,1024]:
        configTMbaseMulti[pfl][pfd] = {}
        for rt in ['GRU','LSTM']:
            configTMbaseMulti[pfl][pfd][rt] = Conf({
                "numFields":            3,
                "numPhrases":           10,
                "phraseLen":            100,
                "vecLen":               60,
                "modelType":            'modelBase',   #model1 or model2
                "masking":              True,
                "phraseFeaturesLayers": pfl,
                "phraseFeaturesDim":    pfd,
                "phraseDropout":        0.,
                "rnnType":              rt,                 #LSTM or GRU
                "hiddenLayers":         0,
                "hiddenDim":            0,                     #None equals outDim
                "outDim":               203,
                "learningRate":         0.001,
                "learningRateDecay":    0.,
                "loss":                 'categorical_crossentropy',
                "dataMethod":           'process',
                "fileCorpus":           'corpusTemporalV2b/corpusTemporal.p',
                "fileValues":           'corpusTemporalV2b/valuesTemporalMorfo1.p',
                "fileVectors":          'corpusTemporalV2b/vectors.txt',
                "corpusKey":            'morfo1',
                "batchSize":            32,
                "startEpoch":           0,
                "epochs":               100,
                "earlyStopping":        10,                   #None or patience
                "tensorboard":          'tensorBoardPytorchMulticlass/confTMbaseMulti/{}/{}/{}'.format(pfl,pfd,rt),    #None or logdir
                "modelSave":            "all",
                "modelFile":            'modelsPytorchMulticlass/confTMbaseMulti/{}/{}/{}/epoch{{}}.pt'.format(pfl,pfd,rt),      #None or dir
            })

configTMbaseMultiBest = configTMbaseMulti[1][256]['GRU'].copy({
            "modelFileLoad":        'modelsPytorchMulticlass/confTMbaseMulti/1/256/GRU/epoch4.pt',      #None or dir
        })

configTMmulti1 = {}
for pfd in [2,4,8,16,32,64,128,256,512]:
    configTMmulti1[pfd] = {}
    for afl in [1,2,4]:
        configTMmulti1[pfd][afl] = {}
        for afd in [2,4,8,16,32,64,128,256,512,1024,2048]:
            configTMmulti1[pfd][afl][afd] = Conf({
                "numFields":            3,
                "numPhrases":           10,
                "phraseLen":            100,
                "vecLen":               60,
                "modelType":            'model1plain',   #model1 or model2
                "masking":              True,
                "phraseFeaturesLayers": 1,
                "phraseFeaturesDim":    pfd,
                "phraseDropout":        0.,
                "featuresDropout":      0.,
                "rnnType":              'GRU',                 #LSTM or GRU
                "bidirectionalMerge":   None,      #avg or max
                "afterFeaturesLayers":  afl,
                "afterFeaturesDim":     afd,
                "afterFeaturesOutDim":  afd,
                "classLayers":          1,
                "hiddenDim":            0,                     #None equals outDim
                "outDim":               203,
                "learningRate":         0.001,
                "learningRateDecay":    0.,
                "loss":                 'categorical_crossentropy',
                "dataMethod":           'process',
                "fileCorpus":           'corpusTemporalV2b/corpusTemporal.p',
                "fileValues":           'corpusTemporalV2b/valuesTemporalMorfo1.p',
                "fileVectors":          'corpusTemporalV2b/vectors.txt',
                "corpusKey":            'morfo1',
                "batchSize":            32,
                "startEpoch":           0,
                "epochs":               100,
                "earlyStopping":        10,                   #None or patience
                "tensorboard":          'tensorBoardPytorchMulticlass/confTMmulti1/{}/{}/{}'.format(pfd, afl, afd),    #None or logdir
                "modelSave":            "all",
                "modelFile":            'modelsPytorchMulticlass/confTMmulti1/{}/{}/{}/epoch{{}}.pt'.format(pfd, afl, afd),      #None or dir
            })

configTMmulti1best = configTMmulti1[128][1][128].copy({
            "modelFileLoad":        'modelsPytorchMulticlass/confTMmulti1/128/1/128/epoch8.pt',      #None or dir
        })

configTMmultiCNN = {}
for nk in [16,32,64,128,256,512,1024,2048]:
    configTMmultiCNN[nk] = Conf({
                "numFields":            3,
                "numPhrases":           10,
                "phraseLen":            100,
                "vecLen":               60,
                "modelType":            'modelCNN',   #model1 or model2
                "preLayer": False,
                "numKernels":           nk,
                "dropout":              0.5,
                "outDim":               203,
                "learningRate":         0.001,
                "learningRateDecay":    0.,
                "loss":                 'categorical_crossentropy',
                "dataMethod":           'process',
                "fileCorpus": 'corpusTemporalV2b/corpusTemporal.p',
                "fileValues": 'corpusTemporalV2b/valuesTemporalMorfo1.p',
                "fileVectors": 'corpusTemporalV2b/vectors.txt',
                "corpusKey": 'morfo1',
                "batchSize":            32,
                "startEpoch":           0,
                "epochs":               100,
                "earlyStopping":        10,                   #None or patience
                "tensorboard":          'tensorBoardPytorchMulticlass/confTMmultiCNN/{}'.format(nk),    #None or logdir
                "modelSave":            "best",
                "modelFile":            'modelsPytorchMulticlass/confTMmultiCNN/{}/best.pt'.format(nk),      #None or dir
            })

configTMmultiCNNp = {}
for nk in [64,128,256,512]:
    configTMmultiCNNp[nk] = {}
    for pd in [64,128,256,512]:
        configTMmultiCNNp[nk][pd] = Conf({
                "numFields":            3,
                "numPhrases":           10,
                "phraseLen":            100,
                "vecLen":               60,
                "modelType":            'modelCNN',   #model1 or model2
                "preLayer":             True,
                "preLayerDim":          pd,
                "numKernels":           nk,
                "dropout":              0.5,
                "outDim":               203,
                "learningRate":         0.001,
                "learningRateDecay":    0.,
                "loss":                 'categorical_crossentropy',
                "dataMethod":           'process',
                "fileCorpus": 'corpusTemporalV2b/corpusTemporal.p',
                "fileValues": 'corpusTemporalV2b/valuesTemporalMorfo1.p',
                "fileVectors": 'corpusTemporalV2b/vectors.txt',
                "corpusKey": 'morfo1',
                "batchSize":            32,
                "startEpoch":           0,
                "epochs":               100,
                "earlyStopping":        10,                   #None or patience
                "tensorboard":          'tensorBoardPytorchMulticlass/confTMmultiCNNp/{}/{}'.format(nk,pd),    #None or logdir
                "modelSave":            "best",
                "modelFile":            'modelsPytorchMulticlass/confTMmultiCNNp/{}/{}/best.pt'.format(nk,pd),      #None or dir
            })

configTMmultiCNNpBest = configTMmultiCNNp[128][64]





configTMImulti = {}
for pfl in [1,2,4]:
    configTMImulti[pfl] = {}
    for pfd in [64,128,256,512]:
        configTMImulti[pfl][pfd] = Conf({
            "numFields":            3,
            "numPhrases":           10,
            "phraseLen":            100,
            "vecLen":               60,
            "modelType":            'model1plain',   #model1 or model2
            "masking":              True,
            "phraseFeaturesLayers": pfl,
            "phraseFeaturesDim":    pfd,
            "phraseDropout":        0.,
            "featuresDropout":      0.,
            "rnnType":              'GRU',                 #LSTM or GRU
            "bidirectionalMerge":   None,      #avg or max
            "afterFeaturesLayers":  1,
            "afterFeaturesDim":     0,
            "afterFeaturesOutDim":  203,
            "classLayers":          0,
            "hiddenDim":            0,                     #None equals outDim
            "outDim":               203,
            "learningRate":         0.001,
            "learningRateDecay":    0.,
            "loss":                 'categorical_crossentropy',
            "dataMethod":           'process',
            "fileCorpus":           'corpusTemporalV2b/corpusTemporal.p',
            "fileValues":           'corpusTemporalV2b/valuesTemporalMorfo1.p',
            "fileVectors":          'corpusTemporalV2b/vectors.txt',
            "corpusKey":            'morfo1',
            "batchSize":            32,
            "startEpoch":           0,
            "epochs":               100,
            "earlyStopping":        10,                   #None or patience
            "tensorboard":          'tensorBoardPytorchMulticlass/confTMImulti/{}/{}'.format(pfl, pfd),    #None or logdir
            "modelSave":            "all",
            "modelFile":            'modelsPytorchMulticlass/confTMImulti/{}/{}/epoch{{}}.pt'.format(pfl,pfd),      #None or dir
        })

configTMImultiBest = configTMImulti[2][256].copy({
        "modelFileLoad":        'modelsPytorchMulticlass/confTMImulti/2/256/epoch10.pt',      #None or dir
   })



configTMTmulti1 = {}
for pfl in [1,2,4]:
    configTMTmulti1[pfl] = {}
    for pfd in [64,128,256,512,1024]:
        configTMTmulti1[pfl][pfd] = {}
        for afl in [1,2,4]:
            configTMTmulti1[pfl][pfd][afl] = {}
            for afd in [64,128,256,512,1024]:
                configTMTmulti1[pfl][pfd][afl][afd] = Conf({
                    "numFields":                3,
                    "numPhrases":               10,
                    "phraseLen":                100,
                    "vecLen":                   60,
                    "modelType":                'model1plainMultitask',   #model1 or model2
                    "masking":                  True,
                    "phraseFeaturesLayers":     pfl,
                    "phraseFeaturesDim":        pfd,
                    "phraseDropout":            0.,
                    "featuresDropout":          0.,
                    "rnnType":                  'GRU',                 #LSTM or GRU
                    "bidirectionalMerge":       None,      #avg or max
                    "afterFeaturesLayersSite":      afl,
                    "afterFeaturesDimSite":         afd,
                    "afterFeaturesOutDimSite":      afd,
                    "classLayersSite":              1,
                    "hiddenDimSite":                0,                     #None equals outDim
                    "outDimSite":                   68,
                    "afterFeaturesLayersMorpho":    afl,
                    "afterFeaturesDimMorpho":       afd,
                    "afterFeaturesOutDimMorpho":    afd,
                    "classLayersMorpho":            1,
                    "hiddenDimMorpho":              0,                     #None equals outDim
                    "outDimMorpho":                 203,
                    "outDim":                666, #Fake, only for checks
                    "learningRate":         0.001,
                    "learningRateDecay":    0.,
                    "loss":                 'categorical_crossentropy',
                    "dataMethod":           'process',
                    "fileCorpus":           'corpusTemporalV2b/corpusTemporal.p',
                    "fileValues":           'corpusTemporalV2b/valuesTemporalBoth.p',
                    "fileVectors":          'corpusTemporalV2b/vectors.txt',
                    "corpusKey":            ['sede1','morfo1'],
                    "batchSize":            32,
                    "startEpoch":           0,
                    "epochs":               100,
                    "earlyStopping":        10,                   #None or patience
                    "tensorboard":          'tensorBoardPytorchMulticlass/confTMTmulti1/{}/{}/{}/{}'.format(pfl, pfd, afl, afd),    #None or logdir
                    "modelFile":            'modelsPytorchMulticlass/confTMTmulti1/{}/{}/{}/{}/epoch{{}}.pt'.format(pfl, pfd, afl, afd),      #None or dir
                })

#TODO:TMP
configTMTmulti1best = configTMTmulti1[1][64][2][512].copy({
            "modelFileLoad":        'modelsPytorchMulticlass/confTMTmulti1/1/64/2/512/epoch5.pt',      #None or dir
        })
 




configTSmulti1 = {}
for pfd in [64,128,256]:
    configTSmulti1[pfd] = {}
    for afl in [0,1]: 
        configTSmulti1[pfd][afl] = {}
        for afd in [256,512,1024]:
            if afl==0 and afd>256: #if afl == 0, afd meaningles
                continue
            configTSmulti1[pfd][afl][afd] = {}
            for ad in [128,256,512,1024]:
                configTSmulti1[pfd][afl][afd][ad] = Conf({
                    "numFields":            3,
                    "numPhrases":           10,
                    "phraseLen":            100,
                    "vecLen":               60,
                    "modelType":            'model1plainSoftmax',   #model1 or model2
                    "masking":              True,
                    "phraseFeaturesLayers": 1,
                    "phraseFeaturesDim":    pfd,
                    "phraseDropout":        0.,
                    "featuresDropout":      0.,
                    "rnnType":              'GRU',                 #LSTM or GRU
                    "bidirectionalMerge":   None,      #avg or max
                    "afterFeaturesLayers":  afl,
                    "afterFeaturesDim":     afd,
                    "afterFeaturesOutDim":  afd,
                    "attentionDim":         ad,
                    "classLayers":          1,
                    "hiddenDim":            0,                     #None equals outDim
                    "outDim":               68,
                    "learningRate":         0.001,
                    "learningRateDecay":    0.,
                    "loss":                 'categorical_crossentropy',
                    "dataMethod":           'process',
                    "fileCorpus":           'corpusTemporal/corpusTemporal.p',
                    "fileValues":           'corpusTemporal/valuesTemporal.p',
                    "fileVectors":          'corpusTemporal/vectors.txt',
                    "corpusKey":            'sede1',
                    "batchSize":            32,
                    "startEpoch":           0,
                    "epochs":               100,
                    "earlyStopping":        10,                   #None or patience
                    "tensorboard":          'tensorBoardPytorchMulticlass/confTSmulti1/{}/{}/{}/{}'.format(pfd, afl, afd, ad),    #None or logdir
                    "modelSave":            "all",
                    "modelFile":            'modelsPytorchMulticlass/confTSmulti1/{}/{}/{}/{}/epoch{{}}.pt'.format(pfd, afl, afd, ad),      #None or dir
                    "getAttentionType":     'weights',     #features, weights or weighted
                })

configTSmulti1best = configTSmulti1[128][1][512][256].copy({
            "modelFileLoad":        'modelsPytorchMulticlass/confTSmulti1/128/1/512/256/epoch8.pt',      #None or dir
        })
 
configTSMmulti1 = {}
for pfd in [64,128,256]:
    configTSMmulti1[pfd] = {}
    for afl in [0,1]: 
        configTSMmulti1[pfd][afl] = {}
        for afd in [64,128,256]:
            if afl==0 and afd>64: #if afl == 0, afd meaningles
                continue
            configTSMmulti1[pfd][afl][afd] = {}
            for ad in [128,256,512,1024]:
                configTSMmulti1[pfd][afl][afd][ad] = Conf({
                    "numFields":            3,
                    "numPhrases":           10,
                    "phraseLen":            100,
                    "vecLen":               60,
                    "modelType":            'model1plainSoftmax',   #model1 or model2
                    "masking":              True,
                    "phraseFeaturesLayers": 1,
                    "phraseFeaturesDim":    pfd,
                    "phraseDropout":        0.,
                    "featuresDropout":      0.,
                    "rnnType":              'GRU',                 #LSTM or GRU
                    "bidirectionalMerge":   None,      #avg or max
                    "afterFeaturesLayers":  afl,
                    "afterFeaturesDim":     afd,
                    "afterFeaturesOutDim":  afd,
                    "attentionDim":         ad,
                    "classLayers":          1,
                    "hiddenDim":            0,                     #None equals outDim
                    "outDim":               203,
                    "learningRate":         0.001,
                    "learningRateDecay":    0.,
                    "loss":                 'categorical_crossentropy',
                    "dataMethod":           'process',
                    "fileCorpus":           'corpusTemporalV2b/corpusTemporal.p',
                    "fileValues":           'corpusTemporalV2b/valuesTemporalMorfo1.p',
                    "fileVectors":          'corpusTemporalV2b/vectors.txt',
                    "corpusKey":            'morfo1',
                    "batchSize":            32,
                    "startEpoch":           0,
                    "epochs":               100,
                    "earlyStopping":        10,                   #None or patience
                    "tensorboard":          'tensorBoardPytorchMulticlass/confTSMmulti1/{}/{}/{}/{}'.format(pfd, afl, afd, ad),    #None or logdir
                    "modelSave":            "all",
                    "modelFile":            'modelsPytorchMulticlass/confTSMmulti1/{}/{}/{}/{}/epoch{{}}.pt'.format(pfd, afl, afd, ad),      #None or dir
                    "getAttentionType":     'weighted',     #features, weights or weighted
                })

configTSMmulti1best = configTSMmulti1[256][1][128][256].copy({
            "modelFileLoad":        'modelsPytorchMulticlass/confTSMmulti1/256/1/128/256/epoch12.pt',      #None or dir
        })
 


configSintex = {}
for mod in ['max', 'soft']:
    configSintex[mod] = {}
    for keep in [1,2,3,4,5,6,7,8,9,10,20,30,40,50]:
        configSintex[mod][keep] = Conf({
            "numFields":            3,
            "numPhrases":           10,
            "phraseLen":            100,
            "vecLen":               60,
            "modelType":            'modelBase',   #model1 or model2
            "masking":              True,
            "phraseFeaturesLayers": 2,
            "phraseFeaturesDim":    256,
            "phraseDropout":        0.,
            "rnnType":              'GRU',                 #LSTM or GRU
            "hiddenLayers":         0,
            "hiddenDim":            0,                     #None equals outDim
            "outDim":               68,
            "learningRate":         0.001,
            "learningRateDecay":    0.,
            "loss":                 'categorical_crossentropy',
            "dataMethod":           'process',
            "fileCorpus":           'corpusSintex/mod-{}/keep-{}/corpusSintex.p'.format(mod,keep),
            "fileValues":           'corpusSintex/values.p',
            "fileVectors":          'corpusSintex/vectors.txt',
            "corpusKey":            'sede1',
            "batchSize":            32,
            "startEpoch":           0,
            "epochs":               100,
            "earlyStopping":        10,                   #None or patience
            "tensorboard":          'tensorBoardPytorchMulticlass/confSintex/{}/{}/'.format(mod,keep),    #None or logdir
            "modelFile":            'modelsPytorchMulticlass/confSintex/{}/{}/epoch{{}}.pt'.format(mod,keep),      #None or dir
        })

configSintex2 = {}
for mod in ['max', 'soft']:
    configSintex2[mod] = {}
    for keep in [1,2,3,4,5,6,7,8,9,10,20,30,40,50]:
        configSintex2[mod][keep] = Conf({
            "numFields":            3,
            "numPhrases":           10,
            "phraseLen":            100,
            "vecLen":               60,
            "modelType":            'modelBase',   #model1 or model2
            "masking":              True,
            "phraseFeaturesLayers": 1,
            "phraseFeaturesDim":    128,
            "phraseDropout":        0.,
            "rnnType":              'GRU',                 #LSTM or GRU
            "hiddenLayers":         0,
            "hiddenDim":            0,                     #None equals outDim
            "outDim":               68,
            "learningRate":         0.001,
            "learningRateDecay":    0.,
            "loss":                 'categorical_crossentropy',
            "dataMethod":           'process',
            "fileCorpus":           'corpusSintex/mod-{}/keep-{}/corpusSintex.p'.format(mod,keep),
            "fileValues":           'corpusSintex/values.p',
            "fileVectors":          'corpusSintex/vectors.txt',
            "corpusKey":            'sede1',
            "batchSize":            32,
            "startEpoch":           0,
            "epochs":               100,
            "earlyStopping":        10,                   #None or patience
            "tensorboard":          'tensorBoardPytorchMulticlass/confSintex2/{}/{}/'.format(mod,keep),    #None or logdir
            "modelFile":            'modelsPytorchMulticlass/confSintex2/{}/{}/epoch{{}}.pt'.format(mod,keep),      #None or dir
        })

configSintex3 = {}
for mod in ['max', 'soft']:
    configSintex3[mod] = {}
    for keep in [1,2,3,4,5,6,7,8,9,10,20,30,40,50]:
        configSintex3[mod][keep] = Conf({
            "numFields":            3,
            "numPhrases":           10,
            "phraseLen":            100,
            "vecLen":               60,
            "modelType":            'modelBase',   #model1 or model2
            "masking":              True,
            "phraseFeaturesLayers": 1,
            "phraseFeaturesDim":    64,
            "phraseDropout":        0.,
            "rnnType":              'GRU',                 #LSTM or GRU
            "hiddenLayers":         0,
            "hiddenDim":            0,                     #None equals outDim
            "outDim":               68,
            "learningRate":         0.001,
            "learningRateDecay":    0.,
            "loss":                 'categorical_crossentropy',
            "dataMethod":           'process',
            "fileCorpus":           'corpusSintexV2/mod-{}/keep-{}/corpusSintex.p'.format(mod,keep),
            "fileValues":           'corpusSintexV2/values.p',
            "fileVectors":          'corpusSintexV2/vectors.txt',
            "corpusKey":            'sede1',
            "batchSize":            32,
            "startEpoch":           0,
            "epochs":               100,
            "earlyStopping":        10,                   #None or patience
            "tensorboard":          'tensorBoardPytorchMulticlass/confSintex3/{}/{}/'.format(mod,keep),    #None or logdir
            "modelFile":            'modelsPytorchMulticlass/confSintex3/{}/{}/epoch{{}}.pt'.format(mod,keep),      #None or dir
        })


configSintex4 = {}
for mod in ['max', 'soft']:
    configSintex4[mod] = {}
    for keep in [1,2,3,4,5,6,7,8,9,10,20,30,40,50,100]:
        configSintex4[mod][keep] = Conf({
            "phraseLen":            keep,
            "vecLen":               60,
            "modelType":            'modelBase',   #model1 or model2
            "phraseFeaturesLayers": 1,
            "phraseFeaturesDim":    32,
            "phraseDropout":        0.,
            "rnnType":              'GRU',                 #LSTM or GRU
            "hiddenLayers":         0,
            "hiddenDim":            0,                     #None equals outDim
            "outDim":               68,
            "learningRate":         0.001,
            "learningRateDecay":    0.,
            "loss":                 'categorical_crossentropy',
            "dataMethod":           'process',
            "fileCorpus":           'corpusSintexV2/mod-{}/keep-{}/corpusSintex.p'.format(mod,keep),
            "fileValues":           'corpusSintexV2/values.p',
            "fileVectors":          'corpusSintexV2/vectors.txt',
            "corpusKey":            'sede1',
            "batchSize":            32,
            "startEpoch":           0,
            "epochs":               100,
            "earlyStopping":        10,                   #None or patience
            "tensorboard":          'tensorBoardPytorchMulticlass/confSintex4/{}/{}/'.format(mod,keep),    #None or logdir
            "modelFile":            'modelsPytorchMulticlass/confSintex4/{}/{}/epoch{{}}.pt'.format(mod,keep),      #None or dir
        })

configSintex5 = {}
for mod in ['max', 'soft']:
    configSintex5[mod] = {}
    for keep in [1,2,3,4,5,6,7,8,9,10,20,30,40,50]:
        configSintex5[mod][keep] = Conf({
            "phraseLen":            keep,
            "vecLen":               60,
            "modelType":            'modelBase',   #model1 or model2
            "phraseFeaturesLayers": 1,
            "phraseFeaturesDim":    16,
            "phraseDropout":        0.1,
            "rnnType":              'GRU',                 #LSTM or GRU
            "hiddenLayers":         0,
            "hiddenDim":            0,                     #None equals outDim
            "outDim":               68,
            "learningRate":         0.001,
            "learningRateDecay":    0.,
            "loss":                 'categorical_crossentropy',
            "dataMethod":           'process',
            "fileCorpus":           'corpusSintexV2/mod-{}/keep-{}/corpusSintex.p'.format(mod,keep),
            "fileValues":           'corpusSintexV2/values.p',
            "fileVectors":          'corpusSintexV2/vectors.txt',
            "corpusKey":            'sede1',
            "batchSize":            32,
            "startEpoch":           0,
            "epochs":               100,
            "earlyStopping":        10,                   #None or patience
            "tensorboard":          'tensorBoardPytorchMulticlass/confSintex5/{}/{}/'.format(mod,keep),    #None or logdir
            "modelFile":            'modelsPytorchMulticlass/confSintex5/{}/{}/epoch{{}}.pt'.format(mod,keep),      #None or dir
        })


configSintex6 = {}
for mod in ['max', 'soft']:
    configSintex6[mod] = {}
    for keep in [1,2,3,4,5,6,7,8,9,10,20,30,40,50,100]:
        configSintex6[mod][keep] = Conf({
            "phraseLen":            keep,
            "vecLen":               60,
            "modelType":            'modelBase',   #model1 or model2
            "phraseFeaturesLayers": 1,
            "phraseFeaturesDim":    32,
            "phraseDropout":        0.,
            "rnnType":              'GRU',                 #LSTM or GRU
            "hiddenLayers":         0,
            "hiddenDim":            0,                     #None equals outDim
            "outDim":               68,
            "learningRate":         0.001,
            "learningRateDecay":    0.,
            "loss":                 'categorical_crossentropy',
            "dataMethod":           'process',
            "fileCorpus":           'corpusSintex/mod-{}/keep-{}/corpusSintex.p'.format(mod,keep),
            "fileValues":           'corpusSintex/values.p',
            "fileVectors":          'corpusSintex/vectors.txt',
            "corpusKey":            'sede1',
            "batchSize":            32,
            "startEpoch":           0,
            "epochs":               100,
            "earlyStopping":        10,                   #None or patience
            "tensorboard":          'tensorBoardPytorchMulticlass/confSintex6/{}/{}/'.format(mod,keep),    #None or logdir
            "modelFile":            'modelsPytorchMulticlass/confSintex6/{}/{}/epoch{{}}.pt'.format(mod,keep),      #None or dir
        })


configSintexB = {}
for mod in ['max', 'soft']:
    configSintexB[mod] = {}
    for keep in [1,2,3,4,5,6,7,8,9,10,20,30,40,50,100]:
        configSintexB[mod][keep] = Conf({
            "phraseLen":            keep,
            "vecLen":               60,
            "modelType":            'modelBase',   #model1 or model2
            "phraseFeaturesLayers": 2,
            "phraseFeaturesDim":    256,
            "phraseDropout":        0.,
            "rnnType":              'GRU',                 #LSTM or GRU
            "hiddenLayers":         0,
            "hiddenDim":            0,                     #None equals outDim
            "outDim":               68,
            "learningRate":         0.001,
            "learningRateDecay":    0.,
            "loss":                 'categorical_crossentropy',
            "dataMethod":           'process',
            "fileCorpus":           'corpusSintex/mod-{}/keep-{}/corpusSintex.p'.format(mod,keep),
            "fileValues":           'corpusSintex/values.p',
            "fileVectors":          'corpusSintex/vectors.txt',
            "corpusKey":            'sede1',
            "batchSize":            32,
            "startEpoch":           0,
            "epochs":               100,
            "earlyStopping":        10,                   #None or patience
            "tensorboard":          'tensorBoardPytorchMulticlass/confSintexB/{}/{}/'.format(mod,keep),    #None or logdir
            "modelFile":            'modelsPytorchMulticlass/confSintexB/{}/{}/epoch{{}}.pt'.format(mod,keep),      #None or dir
        })


configSintexBO = {}
for mod in ['max', 'soft']:
    configSintexBO[mod] = {}
    for keep in [1,2,3,4,5,6,7,8,9,10,20,30,40,50,100]:
        configSintexBO[mod][keep] = Conf({
            "phraseLen":            keep,
            "vecLen":               60,
            "modelType":            'modelBase',   #model1 or model2
            "phraseFeaturesLayers": 2,
            "phraseFeaturesDim":    256,
            "phraseDropout":        0.,
            "rnnType":              'GRU',                 #LSTM or GRU
            "hiddenLayers":         0,
            "hiddenDim":            0,                     #None equals outDim
            "outDim":               68,
            "learningRate":         0.001,
            "learningRateDecay":    0.,
            "loss":                 'categorical_crossentropy',
            "dataMethod":           'process',
            "fileCorpus":           'corpusSintexV2/mod-{}/keep-{}/corpusSintex.p'.format(mod,keep),
            "fileValues":           'corpusSintexV2/values.p',
            "fileVectors":          'corpusSintexV2/vectors.txt',
            "corpusKey":            'sede1',
            "batchSize":            32,
            "startEpoch":           0,
            "epochs":               100,
            "earlyStopping":        10,                   #None or patience
            "tensorboard":          'tensorBoardPytorchMulticlass/confSintexBO/{}/{}/'.format(mod,keep),    #None or logdir
            "modelFile":            'modelsPytorchMulticlass/confSintexBO/{}/{}/epoch{{}}.pt'.format(mod,keep),      #None or dir
        })


configSintexBdrop = {}
for mod in ['max', 'soft']:
    configSintexBdrop[mod] = {}
    for keep in [1,2,3,4,5,6,7,8,9,10,20,30,40,50,100]:
        configSintexBdrop[mod][keep] = {}
        for drop in [0.1,0.2,0.4]:
            configSintexBdrop[mod][keep][drop] = Conf({
                "phraseLen":            keep,
                "vecLen":               60,
                "modelType":            'modelBase',   #model1 or model2
                "phraseFeaturesLayers": 2,
                "phraseFeaturesDim":    256,
                "phraseDropout":        0.,
                "rnnType":              'GRU',                 #LSTM or GRU
                "hiddenLayers":         0,
                "hiddenDim":            0,                     #None equals outDim
                "outDim":               68,
                "learningRate":         0.001,
                "learningRateDecay":    0.,
                "loss":                 'categorical_crossentropy',
                "dataMethod":           'process',
                "fileCorpus":           'corpusSintex/mod-{}/keep-{}/corpusSintex.p'.format(mod,keep),
                "fileValues":           'corpusSintex/values.p',
                "fileVectors":          'corpusSintex/vectors.txt',
                "corpusKey":            'sede1',
                "batchSize":            32,
                "startEpoch":           0,
                "epochs":               100,
                "earlyStopping":        10,                   #None or patience
                "tensorboard":          'tensorBoardPytorchMulticlass/confSintexBdrop/{}/{}/{}/'.format(mod,keep,drop),    #None or logdir
                "modelFile":            'modelsPytorchMulticlass/confSintexBdrop/{}/{}/{}/epoch{{}}.pt'.format(mod,keep,drop),      #None or dir
            })


configSintexBOdrop = {}
for mod in ['max', 'soft']:
    configSintexBOdrop[mod] = {}
    for keep in [1,2,3,4,5,6,7,8,9,10,20,30,40,50,100]:
        configSintexBOdrop[mod][keep] = {}
        for drop in [0.1,0.2,0.4]:
            configSintexBOdrop[mod][keep][drop] = Conf({
                "phraseLen":            keep,
                "vecLen":               60,
                "modelType":            'modelBase',   #model1 or model2
                "phraseFeaturesLayers": 2,
                "phraseFeaturesDim":    256,
                "phraseDropout":        drop,
                "rnnType":              'GRU',                 #LSTM or GRU
                "hiddenLayers":         0,
                "hiddenDim":            0,                     #None equals outDim
                "outDim":               68,
                "learningRate":         0.001,
                "learningRateDecay":    0.,
                "loss":                 'categorical_crossentropy',
                "dataMethod":           'process',
                "fileCorpus":           'corpusSintexV2/mod-{}/keep-{}/corpusSintex.p'.format(mod,keep),
                "fileValues":           'corpusSintexV2/values.p',
                "fileVectors":          'corpusSintexV2/vectors.txt',
                "corpusKey":            'sede1',
                "batchSize":            32,
                "startEpoch":           0,
                "epochs":               100,
                "earlyStopping":        10,                   #None or patience
                "tensorboard":          'tensorBoardPytorchMulticlass/confSintexBOdrop/{}/{}/{}/'.format(mod,keep,drop),    #None or logdir
                "modelFile":            'modelsPytorchMulticlass/confSintexBOdrop/{}/{}/{}/epoch{{}}.pt'.format(mod,keep,drop),      #None or dir
            })

configH = {}
for pfd in [32,64,128,256]:
    configH[pfd] = {}
    for afl in [0,1,2,4]:
        configH[pfd][afl] = {}
        for afd in [256,512,1024,2048]:
            if afl == 0 and afd != 256: #afd useless if afl==0
                continue
            first = False
            configH[pfd][afl][afd] = Conf({
                "numFields":            3,
                "numPhrases":           10,
                "phraseLen":            100,
                "vecLen":               60,
                "modelType":            'model1sent',   #model1 or model2
                "aggregationType":      'max',  #max or softmax
                "masking":              True,
                "phraseFeaturesLayers": 1,
                "phraseFeaturesDim":    pfd,
                "phraseDropout":        0.,
                "docFeaturesLayers":    1,
                "docFeaturesDim":       pfd,
                "docDropout":           0.,
                "featuresDropout":      0.,
                "rnnType":              'GRU',                 #LSTM or GRU
                "bidirectionalMerge":   None,      #avg or max
                "afterFeaturesLayers":  afl,
                "afterFeaturesDim":     afd,
                "afterFeaturesOutDim":  afd,
                "classLayers":          1,
                "hiddenDim":            0,                     #None equals outDim
                "outDim":               68,
                "learningRate":         0.001,
                "learningRateDecay":    0.,
                "loss":                 'categorical_crossentropy',
                "dataMethod":           'process',
                "fileCorpus":           'corpusTemporal/corpusTemporal.p',
                "fileValues":           'corpusTemporal/valuesTemporal.p',
                "fileVectors":          'corpusTemporal/vectors.txt',
                "preprocess":           'load', #load or save
                "preprocessFile":       'corpusTemporal/corpusTemporal-sentPreprocess.p',
                "corpusKey":            'sede1',
                "batchSize":            32,
                "startEpoch":           0,
                "epochs":               100,
                "earlyStopping":        10,                   #None or patience
                "tensorboard":          'tensorBoardPytorchMulticlass/confH/{}/{}/{}'.format(pfd, afl, afd),    #None or logdir
                "modelSave":            "best",
                "modelFile":            'modelsPytorchMulticlass/confH/{}/{}/{}/best.pt'.format(pfd, afl, afd),      #None or dir
            })

configHbest = configH[64][2][1024].copy({
            "modelFileLoad":        'modelsPytorchMulticlass/confH/64/2/1024/best.pt',      #None or dir
        })
 


configHS = {}
for pfd in [32,64,128,256]:
    configHS[pfd] = {}
    for afl in [0,1,2,4]:
        configHS[pfd][afl] = {}
        for afd in [256,512,1024,2048]:
            if afl == 0 and afd != 256: #afd useless if afl==0
                continue
            first = False
            configHS[pfd][afl][afd] = {}
            for ad in [64,128,256,512]:
                configHS[pfd][afl][afd][ad] = Conf({
                    "numFields":            3,
                    "numPhrases":           10,
                    "phraseLen":            100,
                    "vecLen":               60,
                    "modelType":            'model1sent',   #model1 or model2
                    "aggregationType":      'softmax',  #max or softmax
                    "masking":              True,
                    "phraseFeaturesLayers": 1,
                    "phraseFeaturesDim":    pfd,
                    "phraseDropout":        0.,
                    "docFeaturesLayers":    1,
                    "docFeaturesDim":       pfd,
                    "docDropout":           0.,
                    "featuresDropout":      0.,
                    "rnnType":              'GRU',                 #LSTM or GRU
                    "bidirectionalMerge":   None,      #avg or max
                    "afterFeaturesLayers":  afl,
                    "afterFeaturesDim":     afd,
                    "afterFeaturesOutDim":  afd,
                    "attentionDim":         ad,
                    "classLayers":          1,
                    "hiddenDim":            0,                     #None equals outDim
                    "outDim":               68,
                    "learningRate":         0.001,
                    "learningRateDecay":    0.,
                    "loss":                 'categorical_crossentropy',
                    "dataMethod":           'process',
                    "fileCorpus":           'corpusTemporal/corpusTemporal.p',
                    "fileValues":           'corpusTemporal/valuesTemporal.p',
                    "fileVectors":          'corpusTemporal/vectors.txt',
                    "preprocess":           'load', #load or save
                    "preprocessFile":       'corpusTemporal/corpusTemporal-sentPreprocess.p',
                    "corpusKey":            'sede1',
                    "batchSize":            32,
                    "startEpoch":           0,
                    "epochs":               100,
                    "earlyStopping":        10,                   #None or patience
                    "tensorboard":          'tensorBoardPytorchMulticlass/confHS/{}/{}/{}/{}'.format(pfd, afl, afd, ad),    #None or logdir
                    "modelSave":            "best",
                    "modelFile":            'modelsPytorchMulticlass/confHS/{}/{}/{}/{}/best.pt'.format(pfd, afl, afd, ad),      #None or dir
                    "getAttentionType":     'weighted',     #features, weights or weighted
                })

configHSbest = configHS[128][1][1024][128].copy({
            "modelFileLoad":        'modelsPytorchMulticlass/confHS/128/1/1024/128/best.pt',      #None or dir
        })
 
configHM = {}
for pfd in [32,64,128,256]:
    configHM[pfd] = {}
    for afl in [0,1,2,4]:
        configHM[pfd][afl] = {}
        for afd in [256,512,1024,2048]:
            if afl == 0 and afd != 256: #afd useless if afl==0
                continue
            first = False
            configHM[pfd][afl][afd] = Conf({
                "numFields":            3,
                "numPhrases":           10,
                "phraseLen":            100,
                "vecLen":               60,
                "modelType":            'model1sent',   #model1 or model2
                "aggregationType":      'max',  #max or softmax
                "masking":              True,
                "phraseFeaturesLayers": 1,
                "phraseFeaturesDim":    pfd,
                "phraseDropout":        0.,
                "docFeaturesLayers":    1,
                "docFeaturesDim":       pfd,
                "docDropout":           0.,
                "featuresDropout":      0.,
                "rnnType":              'GRU',                 #LSTM or GRU
                "bidirectionalMerge":   None,      #avg or max
                "afterFeaturesLayers":  afl,
                "afterFeaturesDim":     afd,
                "afterFeaturesOutDim":  afd,
                "classLayers":          1,
                "hiddenDim":            0,                     #None equals outDim
                "outDim":               203,
                "learningRate":         0.001,
                "learningRateDecay":    0.,
                "loss":                 'categorical_crossentropy',
                "dataMethod":           'process',
                "fileCorpus":           'corpusTemporalV2b/corpusTemporal.p',
                "fileValues":           'corpusTemporalV2b/valuesTemporalMorfo1.p',
                "fileVectors":          'corpusTemporalV2b/vectors.txt',
                "preprocess":           'load', #load or save
                "preprocessFile":       'corpusTemporal/corpusTemporal-sentPreprocessMorfo1.p',
                "corpusKey":            'morfo1',
                "batchSize":            32,
                "startEpoch":           0,
                "epochs":               100,
                "earlyStopping":        10,                   #None or patience
                "tensorboard":          'tensorBoardPytorchMulticlass/confHM/{}/{}/{}'.format(pfd, afl, afd),    #None or logdir
                "modelSave":            "best",
                "modelFile":            'modelsPytorchMulticlass/confHM/{}/{}/{}/best.pt'.format(pfd, afl, afd),      #None or dir
            })

configHMbest = configHM[64][1][1024].copy({
            "modelFileLoad":        'modelsPytorchMulticlass/confHM/64/1/1024/best.pt',      #None or dir
        })
 


configHSM = {}
for pfd in [32,64,128,256]:
    configHSM[pfd] = {}
    for afl in [0,1,2,4]:
        configHSM[pfd][afl] = {}
        for afd in [256,512,1024,2048]:
            if afl == 0 and afd != 256: #afd useless if afl==0
                continue
            first = False
            configHSM[pfd][afl][afd] = {}
            for ad in [64,128,256,512]:
                configHSM[pfd][afl][afd][ad] = Conf({
                    "numFields":            3,
                    "numPhrases":           10,
                    "phraseLen":            100,
                    "vecLen":               60,
                    "modelType":            'model1sent',   #model1 or model2
                    "aggregationType":      'softmax',  #max or softmax
                    "masking":              True,
                    "phraseFeaturesLayers": 1,
                    "phraseFeaturesDim":    pfd,
                    "phraseDropout":        0.,
                    "docFeaturesLayers":    1,
                    "docFeaturesDim":       pfd,
                    "docDropout":           0.,
                    "featuresDropout":      0.,
                    "rnnType":              'GRU',                 #LSTM or GRU
                    "bidirectionalMerge":   None,      #avg or max
                    "afterFeaturesLayers":  afl,
                    "afterFeaturesDim":     afd,
                    "afterFeaturesOutDim":  afd,
                    "attentionDim":         ad,
                    "classLayers":          1,
                    "hiddenDim":            0,                     #None equals outDim
                    "outDim":               203,
                    "learningRate":         0.001,
                    "learningRateDecay":    0.,
                    "loss":                 'categorical_crossentropy',
                    "dataMethod":           'process',
                    "fileCorpus":           'corpusTemporalV2b/corpusTemporal.p',
                    "fileValues":           'corpusTemporalV2b/valuesTemporalMorfo1.p',
                    "fileVectors":          'corpusTemporalV2b/vectors.txt',
                    "preprocess":           'load', #load or save
                    "preprocessFile":       'corpusTemporal/corpusTemporal-sentPreprocessMorfo1.p',
                    "corpusKey":            'morfo1',
                    "batchSize":            32,
                    "startEpoch":           0,
                    "epochs":               100,
                    "earlyStopping":        10,                   #None or patience
                    "tensorboard":          'tensorBoardPytorchMulticlass/confHSM/{}/{}/{}/{}'.format(pfd, afl, afd, ad),    #None or logdir
                    "modelSave":            "best",
                    "modelFile":            'modelsPytorchMulticlass/confHSM/{}/{}/{}/{}/best.pt'.format(pfd, afl, afd, ad),      #None or dir
                    "getAttentionType":     'weighted',     #features, weights or weighted
                })

configHSMbest = configHSM[64][1][1024][128].copy({
            "modelFileLoad":        'modelsPytorchMulticlass/confHSM/64/1/1024/128/best.pt',      #None or dir
        })
 

configDummy = Conf({
                    "numFields":            3,
                    "numPhrases":           10,
                    "phraseLen":            100,
                    "vecLen":               60,
                    "modelType":            'model1sent',   #model1 or model2
                    "aggregationType":      'softmax',  #max or softmax
                    "masking":              True,
                    "phraseFeaturesLayers": 1,
                    "phraseFeaturesDim":    16,
                    "phraseDropout":        0.,
                    "docFeaturesLayers":    1,
                    "docFeaturesDim":       16,
                    "docDropout":           0.,
                    "featuresDropout":      0.,
                    "rnnType":              'GRU',                 #LSTM or GRU
                    "bidirectionalMerge":   None,      #avg or max
                    "afterFeaturesLayers":  1,
                    "afterFeaturesDim":     16,
                    "afterFeaturesOutDim":  16,
                    "attentionDim":         16,
                    "classLayers":          1,
                    "hiddenDim":            0,                     #None equals outDim
                    "outDim":               203,
                    "learningRate":         0.001,
                    "learningRateDecay":    0.,
                    "loss":                 'categorical_crossentropy',
                    "dataMethod":           'process',
                    "fileCorpus":           'corpusTemporalV2b/corpusTemporal-DBG.p',
                    "fileValues":           'corpusTemporalV2b/valuesTemporalMorfo1.p',
                    "fileVectors":          'corpusTemporalV2b/vectors.txt',
                    "preprocess":           'load', #load or save
                    "preprocessFile":       'corpusTemporal/corpusTemporal-sentPreprocessMorfo1-DBG.p',
                    "corpusKey":            'morfo1',
                    "batchSize":            32,
                    "startEpoch":           0,
                    "epochs":               10,
                    "earlyStopping":        1,                   #None or patience
                    "tensorboard":          'tensorBoardPytorchMulticlass/confDummy',    #None or logdir
                    "modelSave":            "best",
                    "modelFile":            'modelsPytorchMulticlass/confDummy/best.pt',      #None or dir
                    "getAttentionType":     'weighted',     #features, weights or weighted
                })
configDummy2 = configDummy.copy({
                    "tensorboard":          'tensorBoardPytorchMulticlass/confDummy2',    #None or logdir
                    "modelFile":            'modelsPytorchMulticlass/confDummy2/best.pt',      #None or dir
        })


configRunSeqPar = {
        1:[(configHM[pfd][1][afd], "configHM-{}-1-{}".format(pfd,afd)) for pfd,afd in itertools.product([32,64,128,256], [256,512,1024,2048])] + [(configHM[pfd][0][256], "configHM-{}-0-256".format(pfd)) for pfd in [32,64]],
        2:[(configHM[pfd][2][afd], "configHM-{}-2-{}".format(pfd,afd)) for pfd,afd in itertools.product([32,64,128,256], [256,512,1024,2048])] + [(configHM[128][0][256], "configHM-128-0-256")],
        3:[(configHM[pfd][4][afd], "configHM-{}-4-{}".format(pfd,afd)) for pfd,afd in itertools.product([32,64,128,256], [256,512,1024,2048])] + [(configHM[256][0][256], "configHM-256-0-256")],
        
        4:[(configHSM[pfd][1][256][ad], "configHSM-{}-1-256-{}".format(pfd,ad)) for pfd,ad in itertools.product([64,128,256], [128,256,512])],
        5:[(configHSM[pfd][1][512][ad], "configHSM-{}-1-512-{}".format(pfd,ad)) for pfd,ad in itertools.product([64,128,256], [128,256,512])],
        6:[(configHSM[pfd][1][1024][ad], "configHSM-{}-1-1024-{}".format(pfd,ad)) for pfd,ad in itertools.product([64,128,256], [128,256,512])],
 
        7:[(configHSM[pfd][2][256][ad], "configHSM-{}-2-256-{}".format(pfd,ad)) for pfd,ad in itertools.product([64,128,256], [128,256,512])],
        8:[(configHSM[pfd][2][512][ad], "configHSM-{}-2-512-{}".format(pfd,ad)) for pfd,ad in itertools.product([64,128,256], [128,256,512])],
        9:[(configHSM[pfd][2][1024][ad], "configHSM-{}-2-1024-{}".format(pfd,ad)) for pfd,ad in itertools.product([64,128,256], [128,256,512])],
        
        10:[(configHS[pfd][afl][afd][ad], "configHS-{}-{}-{}-{}".format(pfd,afl,afd,ad)) for pfd,afl,afd,ad in [
                (128,1,2048,128),
                (128,1,1024,64),
            ]],

        666:[(configDummy, "configDummy")]
    }






configRun1 = configTmultiCNN[16]
configRun2 = configTmultiCNN[32]
configRun3 = configTmultiCNN[64]
configRun4 = configTmultiCNN[128]
configRun5 = configTmultiCNN[256]
configRun6 = configTmultiCNN[512]

configRunSeq1 = [(configTmultiCNNp[nk][pd], "{}-{}".format(nk,pd)) for nk,pd in itertools.product([64],[64,128,256,512])]
configRunSeq2 = [(configTmultiCNNp[nk][pd], "{}-{}".format(nk,pd)) for nk,pd in itertools.product([128],[64,128,256,512])]
configRunSeq3 = [(configTmultiCNNp[nk][pd], "{}-{}".format(nk,pd)) for nk,pd in itertools.product([256],[64,128,256,512])]
configRunSeq4 = [(configTmultiCNNp[nk][pd], "{}-{}".format(nk,pd)) for nk,pd in itertools.product([512],[64,128,256,512])]
configRunSeq5 = [(configTMmultiCNNp[nk][pd], "{}-{}".format(nk,pd)) for nk,pd in itertools.product([64],[64,128,256,512])]
configRunSeq6 = [(configTMmultiCNNp[nk][pd], "{}-{}".format(nk,pd)) for nk,pd in itertools.product([128],[64,128,256,512])]
configRunSeq7 = [(configTMmultiCNNp[nk][pd], "{}-{}".format(nk,pd)) for nk,pd in itertools.product([256],[64,128,256,512])]
configRunSeq8 = [(configTMmultiCNNp[nk][pd], "{}-{}".format(nk,pd)) for nk,pd in itertools.product([512],[64,128,256,512])]

#configRunSeq2 = [(configTMmultiCNN[nk], "{}".format(nk)) for nk in [64,128]]
#configRunSeq3 = [(configTMmultiCNN[nk], "{}".format(nk)) for nk in [256,512]]
#configRunSeq4 = [(configTMmultiCNN[nk], "{}".format(nk)) for nk in [1024,2048]]

#configRunSeq2 = [(configTmulti1[pfd][1][afd], "{}-1-{}".format(pfd, afd)) for pfd,afd in itertools.product([8,16],[2,4,8,16,32,64,128,256])]
#configRunSeq3 = [(configTmulti1[pfd][1][afd], "{}-1-{}".format(pfd, afd)) for pfd,afd in itertools.product([32],[2,4,8,16,32,64,128,256])]
#configRunSeq3b = [(configTmulti1[pfd][1][afd], "{}-1-{}".format(pfd, afd)) for pfd,afd in itertools.product([64],[2,4,8,16,32,64,128,256])]
#configRunSeq4 = [(configTmulti1[pfd][1][afd], "{}-1-{}".format(pfd, afd)) for pfd,afd in itertools.product([128,256],[2,4,8,16,32,64,128,256])]
#configRunSeq5 = [(configTmulti1[pfd][2][afd], "{}-2-{}".format(pfd, afd)) for pfd,afd in itertools.product([2,4],[2,4,8,16,32,64,128,256])]
#configRunSeq6 = [(configTmulti1[pfd][2][afd], "{}-2-{}".format(pfd, afd)) for pfd,afd in itertools.product([8,16],[2,4,8,16,32,64,128,256])]
#configRunSeq7 = [(configTmulti1[pfd][2][afd], "{}-2-{}".format(pfd, afd)) for pfd,afd in itertools.product([32,64],[2,4,8,16,32,64,128,256])]
#configRunSeq8 = [(configTmulti1[pfd][2][afd], "{}-2-{}".format(pfd, afd)) for pfd,afd in itertools.product([128,256],[2,4,8,16,32,64,128,256])]
#configRunSeq9 = [(configTmulti1[pfd][4][afd], "{}-4-{}".format(pfd, afd)) for pfd,afd in itertools.product([2,4],[2,4,8,16,32,64,128,256])]
#configRunSeq10 = [(configTmulti1[pfd][4][afd], "{}-4-{}".format(pfd, afd)) for pfd,afd in itertools.product([8,16],[2,4,8,16,32,64,128,256])]
#configRunSeq11 = [(configTmulti1[pfd][4][afd], "{}-4-{}".format(pfd, afd)) for pfd,afd in itertools.product([32,64],[2,4,8,16,32,64,128,256])]
#configRunSeq12 = [(configTmulti1[pfd][4][afd], "{}-4-{}".format(pfd, afd)) for pfd,afd in itertools.product([128,256],[2,4,8,16,32,64,128,256])]


configRun1continue = config29Folds2Continue
configRun2continue = config22continue

configFolds1 = config30Folds
configFolds2 = config29Folds

