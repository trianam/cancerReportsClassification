import itertools
from conf import Conf

config1 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "featuresDim":          70,
    "dropout":              0.2,
    "featuresLayers":       1,
    "afterFeaturesLayers":  2,
    "hiddenLayers":         2,
    "hiddenDim":            64,                       #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               100,
    "earlyStopping":        2,                          #None or patience
    "tensorboard":          'tensorBoardStefanoModel',  #None or logdir
})

config2 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "featuresDim":          128,
    "dropout":              0.2,
    "featuresLayers":       1,
    "afterFeaturesLayers":  0,
    "hiddenLayers":         2,
    "hiddenDim":            128,                       #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               100,
    "earlyStopping":        None,                          #None or patience
    "tensorboard":          'tensorBoardStefanoModel2',  #None or logdir
})


config3 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "featuresDim":          64,
    "dropout":              0.5,
    "featuresLayers":       1,
    "afterFeaturesLayers":  0,
    "hiddenLayers":         2,
    "hiddenDim":            64,                       #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               100,
    "earlyStopping":        None,                          #None or patience
    "tensorboard":          'tensorBoardStefanoModel3',  #None or logdir
})

config4 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "featuresDim":          128,
    "dropout":              0.5,
    "featuresLayers":       1,
    "afterFeaturesLayers":  0,
    "hiddenLayers":         2,
    "hiddenDim":            64,                       #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               100,
    "earlyStopping":        None,                          #None or patience
    "tensorboard":          'tensorBoardStefanoModel4',  #None or logdir
})

config5 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "featuresDim":          32,
    "dropout":              0.5,
    "featuresLayers":       1,
    "afterFeaturesLayers":  0,
    "hiddenLayers":         2,
    "hiddenDim":            32,                       #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               100,
    "earlyStopping":        None,                          #None or patience
    "tensorboard":          'tensorBoardStefanoModel5',  #None or logdir
})

config6 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "featuresDim":          32,
    "dropout":              0.5,
    "featuresLayers":       1,
    "afterFeaturesLayers":  0,
    "hiddenLayers":         1,
    "hiddenDim":            32,                       #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               100,
    "earlyStopping":        None,                          #None or patience
    "tensorboard":          'tensorBoardStefanoModel6',  #None or logdir
})

config7 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "featuresDim":          32,
    "dropout":              0.5,
    "featuresLayers":       1,
    "afterFeaturesLayers":  0,
    "hiddenLayers":         2,
    "hiddenDim":            64,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoard/conf7',    #None or logdir
    "modelFile":            'models/conf7.h5',      #None or dir
})

config8 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "featuresDim":          32,
    "dropout":              0.5,
    "featuresLayers":       2,
    "afterFeaturesLayers":  0,
    "hiddenLayers":         2,
    "hiddenDim":            64,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoard/conf8',    #None or logdir
    "modelFile":            'models/conf8.h5',      #None or dir
})

config8t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "masking":              False,
    "featuresDim":          32,
    "dropout":              0.5,
    "rnnType":              'LSTM',                 #LSTM or GRU
    "featuresLayers":       2,
    "afterFeaturesLayers":  0,
    "hiddenLayers":         2,
    "hiddenDim":            64,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf8',    #None or logdir
    "modelFile":            'modelsPytorch/conf8.pt',      #None or dir
})

config9t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "masking":              True,
    "featuresDim":          32,
    "dropout":              0.5,
    "rnnType":              'LSTM',                 #LSTM or GRU
    "featuresLayers":       2,
    "afterFeaturesLayers":  0,
    "hiddenLayers":         2,
    "hiddenDim":            64,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf9',    #None or logdir
    "modelFile":            'modelsPytorch/conf9.pt',      #None or dir
})

config10t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "masking":              True,
    "featuresDim":          32,
    "dropout":              0.8,
    "rnnType":              'LSTM',                 #LSTM or GRU
    "featuresLayers":       2,
    "afterFeaturesLayers":  0,
    "hiddenLayers":         2,
    "hiddenDim":            64,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf10',    #None or logdir
    "modelFile":            'modelsPytorch/conf10.pt',      #None or dir
})

config11t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "masking":              True,
    "featuresDim":          8,
    "dropout":              0.5,
    "rnnType":              'LSTM',                 #LSTM or GRU
    "featuresLayers":       2,
    "afterFeaturesLayers":  0,
    "hiddenLayers":         2,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf11',    #None or logdir
    "modelFile":            'modelsPytorch/conf11.pt',      #None or dir
})

config12t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "masking":              True,
    "featuresDim":          8,
    "dropout":              0.5,
    "rnnType":              'GRU',                 #LSTM or GRU
    "featuresLayers":       2,
    "afterFeaturesLayers":  0,
    "hiddenLayers":         2,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf12',    #None or logdir
    "modelFile":            'modelsPytorch/conf12.pt',      #None or dir
})

config13t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "masking":              True,
    "featuresDim":          8,
    "dropout":              0.5,
    "rnnType":              'GRU',                 #LSTM or GRU
    "featuresLayers":       1,
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf13',    #None or logdir
    "modelFile":            'modelsPytorch/conf13.pt',      #None or dir
})

#good train, overfitting
config14t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "masking":              True,
    "featuresDim":          16,
    "dropout":              0.5,
    "rnnType":              'GRU',                 #LSTM or GRU
    "featuresLayers":       1,
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf14',    #None or logdir
    "modelFile":            'modelsPytorch/conf14.pt',      #None or dir
})

config15t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "masking":              True,
    "featuresDim":          16,
    "phraseDropout":        0.5,
    "fieldDropout":         0.5,
    "docDropout":           0.5,
    "featuresDropout":      0.5,
    "rnnType":              'GRU',                 #LSTM or GRU
    "featuresLayers":       1,
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf15',    #None or logdir
    "modelFile":            'modelsPytorch/conf15.pt',      #None or dir
})

config16t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "masking":              True,
    "featuresDim":          16,
    "phraseDropout":        0.1,
    "fieldDropout":         0.1,
    "docDropout":           0.1,
    "featuresDropout":      0.5,
    "rnnType":              'GRU',                 #LSTM or GRU
    "featuresLayers":       1,
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf16',    #None or logdir
    "modelFile":            'modelsPytorch/conf16.pt',      #None or dir
})

config17t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "masking":              True,
    "featuresDim":          16,
    "phraseDropout":        0.1,
    "fieldDropout":         0.1,
    "docDropout":           0.1,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "featuresLayers":       1,
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf17',    #None or logdir
    "modelFile":            'modelsPytorch/conf17.pt',      #None or dir
})

config18t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "masking":              True,
    "featuresDim":          16,
    "phraseDropout":        0.2,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "featuresLayers":       1,
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf18',    #None or logdir
    "modelFile":            'modelsPytorch/conf18.pt',      #None or dir
})

config19t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "masking":              True,
    "featuresDim":          8,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "featuresLayers":       1,
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf19',    #None or logdir
    "modelFile":            'modelsPytorch/conf19.pt',      #None or dir
})

config20t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "masking":              True,
    "featuresDim":          4,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "featuresLayers":       1,
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf20',    #None or logdir
    "modelFile":            'modelsPytorch/conf20.pt',      #None or dir
})

# best val acc 87.7
config21t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "masking":              True,
    "featuresDim":          2,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "featuresLayers":       1,
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf21',    #None or logdir
    "modelFile":            'modelsPytorch/conf21.pt',      #None or dir
})

config21tContinue = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "masking":              True,
    "featuresDim":          2,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "featuresLayers":       1,
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf21Continue',    #None or logdir
    "modelFile":            'modelsPytorch/conf21.pt',      #None or dir
})

config22t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "masking":              True,
    "featuresDim":          1,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "featuresLayers":       1,
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf22',    #None or logdir
    "modelFile":            'modelsPytorch/conf22.pt',      #None or dir
})

config23t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "masking":              True,
    "featuresDim":          2,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "featuresLayers":       2,
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf23',    #None or logdir
    "modelFile":            'modelsPytorch/conf23.pt',      #None or dir
})

config24t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "masking":              True,
    "featuresDim":          2,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "featuresLayers":       1,
    "afterFeaturesLayers":  0,
    "hiddenLayers":         2,
    "hiddenDim":            4,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf24',    #None or logdir
    "modelFile":            'modelsPytorch/conf24.pt',      #None or dir
})

config25t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "masking":              True,
    "featuresDim":          2,
    "phraseDropout":        0.1,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "featuresLayers":       1,
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               200,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf25',    #None or logdir
    "modelFile":            'modelsPytorch/conf25.pt',      #None or dir
})

config26t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "masking":              True,
    "featuresDim":          2,
    "phraseDropout":        0.01,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "featuresLayers":       1,
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               200,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf26',    #None or logdir
    "modelFile":            'modelsPytorch/conf26.pt',      #None or dir
})


config26tContinue = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "masking":              True,
    "featuresDim":          2,
    "phraseDropout":        0.01,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "featuresLayers":       1,
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               200,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf26Continue',    #None or logdir
    "modelFile":            'modelsPytorch/conf26.pt',      #None or dir
})

config26tContinue2 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "masking":              True,
    "featuresDim":          2,
    "phraseDropout":        0.01,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "featuresLayers":       1,
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               200,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf26Continue2',    #None or logdir
    "modelFile":            'modelsPytorch/conf26.pt',      #None or dir
})

config27t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "masking":              False,
    "featuresDim":          2,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "featuresLayers":       1,
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               200,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf27',    #None or logdir
    "modelFile":            'modelsPytorch/conf27.pt',      #None or dir
})

config28t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "masking":              True,
    "featuresDim":          2,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "featuresLayers":       1,
    "afterFeaturesLayers":  1,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf28',    #None or logdir
    "modelFile":            'modelsPytorch/conf28.pt',      #None or dir
})

config29t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "masking":              False,
    "phraseFeaturesLayers": 1,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    3,
    "fieldFeaturesDim":     2,
    "docFeaturesDim":       1,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               200,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf29',    #None or logdir
    "modelFile":            'modelsPytorch/conf29.pt',      #None or dir
})

config29tBis = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "masking":              False,
    "phraseFeaturesLayers": 1,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    3,
    "fieldFeaturesDim":     2,
    "docFeaturesDim":       1,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               200,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf29Bis',    #None or logdir
    "modelFile":            'modelsPytorch/conf29Bis.pt',      #None or dir
})

config29tTer = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',
    "masking":              False,
    "phraseFeaturesLayers": 1,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    3,
    "fieldFeaturesDim":     2,
    "docFeaturesDim":       1,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               200,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf29ter',    #None or logdir
    "modelFile":            'modelsPytorch/conf29ter.pt',      #None or dir
})

config30t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "masking":              False,
    "phraseFeaturesLayers": 1,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    2,
    "fieldFeaturesDim":     1,
    "docFeaturesDim":       1,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               200,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf30',    #None or logdir
    "modelFile":            'modelsPytorch/conf30.pt',      #None or dir
})

config31t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "masking":              False,
    "phraseFeaturesLayers": 1,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    1,
    "fieldFeaturesDim":     2,
    "docFeaturesDim":       3,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               200,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf31',    #None or logdir
    "modelFile":            'modelsPytorch/conf31.pt',      #None or dir
})

config31tContinue = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "masking":              False,
    "phraseFeaturesLayers": 1,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    1,
    "fieldFeaturesDim":     2,
    "docFeaturesDim":       3,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               200,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf31Continue',    #None or logdir
    "modelFile":            'modelsPytorch/conf31.pt',      #None or dir
})

config31tSave = Conf({
    "modelFile":            'modelsPytorch/conf31Continue.pt',      #None or dir
})

config31tContinue2 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "masking":              False,
    "phraseFeaturesLayers": 1,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    1,
    "fieldFeaturesDim":     2,
    "docFeaturesDim":       3,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               200,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf31Continue2',    #None or logdir
    "modelFile":            'modelsPytorch/conf31Continue.pt',      #None or dir
})

config31tSave2 = Conf({
    "modelFile":            'modelsPytorch/conf31Continue2.pt',      #None or dir
})

config32t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    3,
    "fieldFeaturesDim":     2,
    "docFeaturesDim":       1,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusMMI/corpusFilteredBalanced.p',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               200,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf32',    #None or logdir
    "modelFile":            'modelsPytorch/conf32.pt',      #None or dir
})

config33t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model2',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    3,
    "fieldFeaturesDim":     2,
    "docFeaturesDim":       1,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "afterFeaturesLayers":  0,
    "afterPhraseLayers":    0,
    "afterFieldLayers":     0,
    "afterDocLayers":       0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               200,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf33',    #None or logdir
    "modelFile":            'modelsPytorch/conf33.pt',      #None or dir
})

config33tBis = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model2',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    3,
    "fieldFeaturesDim":     2,
    "docFeaturesDim":       1,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "afterFeaturesLayers":  0,
    "afterPhraseLayers":    0,
    "afterFieldLayers":     0,
    "afterDocLayers":       0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               200,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf33Bis',    #None or logdir
    "modelFile":            'modelsPytorch/conf33Bis.pt',      #None or dir
})

config34t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model2',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    4,
    "fieldFeaturesDim":     4,
    "docFeaturesDim":       4,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "afterFeaturesLayers":  0,
    "afterPhraseLayers":    0,
    "afterFieldLayers":     0,
    "afterDocLayers":       0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusMMI/corpusFilteredBalanced.p',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               200,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf34',    #None or logdir
    "modelFile":            'modelsPytorch/conf34.pt',      #None or dir
})

config35t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model2',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    3,
    "fieldFeaturesDim":     2,
    "docFeaturesDim":       1,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "afterFeaturesLayers":  0,
    "afterPhraseLayers":    0,
    "afterFieldLayers":     0,
    "afterDocLayers":       0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusMMI/corpusFilteredBalanced.p',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf35',    #None or logdir
    "modelFile":            'modelsPytorch/conf35.pt',      #None or dir
})

config36t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    1,
    "fieldFeaturesDim":     1,
    "docFeaturesDim":       1,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusMMI/corpusFilteredBalanced.p',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf36',    #None or logdir
    "modelFile":            'modelsPytorch/conf36.pt',      #None or dir
})

#val accuracy 86, interpretable?
config37t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
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
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusMMI/corpusFilteredBalanced.p',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf37',    #None or logdir
    "modelFile":            'modelsPytorch/conf37.pt',      #None or dir
})


config37tContinue = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
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
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusMMI/corpusFilteredBalanced.p',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf37continue',    #None or logdir
    "modelFile":            'modelsPytorch/conf37.pt',      #None or dir
})

config37tSave = Conf({
    "modelFile":            'modelsPytorch/conf37continue.pt',      #None or dir
})
config38t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    4,
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
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusMMI/corpusFilteredBalanced.p',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf38',    #None or logdir
    "modelFile":            'modelsPytorch/conf38.pt',      #None or dir
})

config39t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
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
    "bidirectionalMerge":   'max',      #avg or max
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusMMI/corpusFilteredBalanced.p',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               200,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf39',    #None or logdir
    "modelFile":            'modelsPytorch/conf39.pt',      #None or dir
})

config40t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 2,
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
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusMMI/corpusFilteredBalanced.p',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               1000,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf40',    #None or logdir
    "modelFile":            'modelsPytorch/conf40.pt',      #None or dir
})

config40tContinue = config40t.copy({
    "tensorboard":          'tensorBoardPytorch/conf40continue',
})
config40tSave = config40t.copy({
    "modelFile":            'modelsPytorch/conf40continue.pt',
})

config40tContinue2 = config40t.copy({
    "tensorboard":          'tensorBoardPytorch/conf40continue2',
    "modelFile":            'modelsPytorch/conf40continue.pt',      #None or dir
})
config40tSave2 = config40t.copy({
    "modelFile":            'modelsPytorch/conf40continue2.pt',
})

config41t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 4,
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
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusMMI/corpusFilteredBalanced.p',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               1000,
    "earlyStopping":        3,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf41',    #None or logdir
    "modelFile":            'modelsPytorch/conf41.pt',      #None or dir
})

#valid 87 interpretable
config42t = Conf({
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
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusMMI/corpusFilteredBalanced.p',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               1000,
    "earlyStopping":        3,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf42',    #None or logdir
    "modelFile":            'modelsPytorch/conf42.pt',      #None or dir
    "modelFileLoad":            'modelsPytorch/conf42.pt',      #None or dir
})


config43t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 16,
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
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusMMI/corpusFilteredBalanced.p',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "epochs":               1000,
    "earlyStopping":        3,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf43',    #None or logdir
    "modelFile":            'modelsPytorch/conf43.pt',      #None or dir
})

config44t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 8,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    4,
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
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusMMI/corpusFilteredBalanced.p',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               1000,
    "earlyStopping":        2,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf44',    #None or logdir
    "modelFile":            'modelsPytorch/conf44/epoch{}.pt',      #None or dir
})

config44tLoad = config44t.copy({
    "modelFileLoad":            'modelsPytorch/conf44/epoch90.pt',
})

config45t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 8,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    4,
    "fieldFeaturesDim":     2,
    "docFeaturesDim":       2,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'LSTM',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusMMI/corpusFilteredBalanced.p',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               200,
    "earlyStopping":        2,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf45',    #None or logdir
    "modelFile":            'modelsPytorch/conf45/epoch{}.pt',      #None or dir
})

config45tContinue = config45t.copy({
    "startEpoch":           200,
    "modelFileLoad":        'modelsPytorch/conf45/epoch199.pt',
})

config45tLoad = config45t.copy({
    "modelFileLoad":        'modelsPytorch/conf45/epoch180.pt',
})

#strange down peaks
config46t = Conf({
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
    "rnnType":              'LSTM',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusMMI/corpusFilteredBalanced.p',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               200,
    "earlyStopping":        1,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf46',    #None or logdir
    "modelFile":            'modelsPytorch/conf46/epoch{}.pt',      #None or dir
})

config46tContinue = config46t.copy({
    "startEpoch":           200,
    "modelFileLoad":        'modelsPytorch/conf46/epoch199.pt',
})

config47t = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 8,
    "fieldFeaturesLayers":  1,
    "docFeaturesLayers":    1,
    "phraseFeaturesDim":    3,
    "fieldFeaturesDim":     2,
    "docFeaturesDim":       2,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'LSTM',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusMMI/corpusFilteredBalanced.p',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               200,
    "earlyStopping":        1,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf47',    #None or logdir
    "modelFile":            'modelsPytorch/conf47/epoch{}.pt',      #None or dir
})

config48t = Conf({
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
    "fieldFeaturesDim":     1,
    "docFeaturesDim":       1,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusMMI/corpusFilteredBalanced.p',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               400,
    "earlyStopping":        1,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf48',    #None or logdir
    "modelFile":            'modelsPytorch/conf48/epoch{}.pt',      #None or dir
})

config48tContinue = config48t.copy({
    "startEpoch":           400,
    "modelFileLoad":        'modelsPytorch/conf48/epoch399.pt',
})

config48tLoad = config48t.copy({
    "modelFileLoad":        'modelsPytorch/conf48/epoch799.pt',
})

config49t = Conf({
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
    "fieldFeaturesDim":     1,
    "docFeaturesDim":       1,
    "phraseDropout":        0.,
    "fieldDropout":         0.,
    "docDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'LSTM',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "hiddenLayers":         0,
    "hiddenDim":            16,                     #None equals outDim
    "outDim":               1, #67,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusMMI/corpusFilteredBalanced.p',
    "fileX":                'corpusMMI/X.npy',
    "fileY":                'corpusMMI/y.npy',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               400,
    "earlyStopping":        1,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf49',    #None or logdir
    "modelFile":            'modelsPytorch/conf49/epoch{}.pt',      #None or dir
})

config49tContinue = config49t.copy({
    "startEpoch":           400,
    "modelFileLoad":        'modelsPytorch/conf49/epoch399.pt',
})

config49tLoad = config49t.copy({
    "modelFileLoad":        'modelsPytorch/conf49/epoch399.pt',
})

config50 = Conf({
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
    "afterFeaturesDim":     8,
    "afterFeaturesOutDim":  1,
    "classLayers":          0,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFilteredBalanced.p',
    "fileValues":           'corpusMMI/sede1-values-binary.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-array.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf50',    #None or logdir
    "modelFile":            'modelsPytorch/conf50/epoch{}.pt',      #None or dir
    "modelFileLoad":        'modelsPytorch/conf50/epoch11.pt',
})

config50c = Conf({
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
    "afterFeaturesDim":     8,
    "afterFeaturesOutDim":  2,
    "classLayers":          0,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               2,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFilteredBalanced.p',
    "fileValues":           'corpusMMI/sede1-values-binary.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-array.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf50c',    #None or logdir
    "modelFile":            'modelsPytorch/conf50c/epoch{}.pt',      #None or dir
    "modelFileLoad":        'modelsPytorch/conf50c/epoch19.pt',
})

config50cContinue = config50c.copy({
    "startEpoch":           19,
    "earlyStopping":        20,
    "modelFileLoad":        'modelsPytorch/conf50c/epoch18.pt',
})


config51 = Conf({
    "numFields":            3,
    "numPhrases":           10,
    "phraseLen":            100,
    "vecLen":               60,
    "modelType":            'model1plain',   #model1 or model2
    "masking":              True,
    "phraseFeaturesLayers": 1,
    "phraseFeaturesDim":    8,
    "phraseDropout":        0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   None,      #avg or max
    "afterFeaturesLayers":  2,
    "afterFeaturesDim":     8,
    "afterFeaturesOutDim":  2,
    "classLayers":          0,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               2,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'categorical_crossentropy',
    "dataMethod":           'process',
    "fileCorpus":           'corpusMMI/corpusFilteredBalanced.p',
    "fileValues":           'corpusMMI/sede1-values-binary.p',
    "fileVectors":          'corpusMMI/vectors.txt',
    "corpusKey":            'sede1',
    "fold":                 0,
    "validSplit":           10,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusMMI/split-array.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorch/conf51',    #None or logdir
    "modelFile":            'modelsPytorch/conf51/epoch{}.pt',      #None or dir
    "modelFileLoad":        'modelsPytorch/conf51/epoch32.pt',
})










configFmulti1 = {}
for pfd in [2,4,8,16,32,64,128,256,512]:
    configFmulti1[pfd] = {}
    for afl in [1,2,4]:
        configFmulti1[pfd][afl] = {}
        for afd in [2,4,8,16,32,64,128,256,512,1024,2048]:
            configFmulti1[pfd][afl][afd] = Conf({
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
                "outDim":               2,
                "learningRate":         0.001,
                "learningRateDecay":    0.,
                "loss":                 'categorical_crossentropy',
                "dataMethod":           'process',
                "fileCorpus":           'corpusMMI/corpusFilteredBalanced.p',
                "fileValues":           'corpusMMI/sede1-values-binary.p',
                "fileVectors":          'corpusMMI/vectors.txt',
                "corpusKey":            'sede1',
                #"fold":                 0,
                "validSplit":           10,
                "loadSplit":            True,
                "saveSplit":            False,
                "fileSplit":            'corpusMMI/split-array.p',
                "batchSize":            32,
                "startEpoch":           0,
                "epochs":               100,
                "earlyStopping":        10,                   #None or patience
                "tensorboard":          'tensorBoardPytorch/confFmulti1/{}/{}/{}'.format(pfd, afl, afd),    #None or logdir
                "modelSave":            "best",
                "modelFile":            'modelsPytorch/confFmulti1/{}/{}/{}/best.pt'.format(pfd, afl, afd),      #None or dir
            })

configFmulti1best = configFmulti1[128][1][256]


configFHS = {}
for pfd in [32,64,128,256]:
    configFHS[pfd] = {}
    for afl in [0,1,2,4]:
        configFHS[pfd][afl] = {}
        for afd in [256,512,1024,2048]:
            if afl == 0 and afd != 256: #afd useless if afl==0
                continue
            first = False
            configFHS[pfd][afl][afd] = {}
            for ad in [64,128,256,512]:
                configFHS[pfd][afl][afd][ad] = Conf({
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
                    "outDim":               2,
                    "learningRate":         0.001,
                    "learningRateDecay":    0.,
                    "loss":                 'categorical_crossentropy',
                    "dataMethod":           'process',
                    "fileCorpus":           'corpusMMI/corpusFilteredBalanced.p',
                    "fileValues":           'corpusMMI/sede1-values-binary.p',
                    "fileVectors":          'corpusMMI/vectors.txt',
                    "preprocess":           'load', #load or save
                    "preprocessFile":       'corpusMMI/corpusFilteredBalanced-sentPreprocess.p',
                    "corpusKey":            'sede1',
                    "validSplit":           10,
                    "loadSplit":            True,
                    "saveSplit":            False,
                    "fileSplit":            'corpusMMI/split-array.p',
                    "batchSize":            32,
                    "startEpoch":           0,
                    "epochs":               100,
                    "earlyStopping":        10,                   #None or patience
                    "tensorboard":          'tensorBoardPytorch/confFHS/{}/{}/{}/{}'.format(pfd, afl, afd, ad),    #None or logdir
                    "modelSave":            "best",
                    "modelFile":            'modelsPytorch/confFHS/{}/{}/{}/{}/best.pt'.format(pfd, afl, afd, ad),      #None or dir
                    "getAttentionType":     'weighted',     #features, weights or weighted
                })

configFHSbest = configFHS[64][2][1024][256]
 


configB = {}
for pfd in [2,4,8,16,32,64,128,256,512]:
    configB[pfd] = {}
    for afl in [1,2,4]:
        configB[pfd][afl] = {}
        for afd in [2,4,8,16,32,64,128,256,512,1024,2048]:
            configB[pfd][afl][afd] = Conf({
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
                "outDim":               2,
                "learningRate":         0.001,
                "learningRateDecay":    0.,
                "loss":                 'categorical_crossentropy',
                "dataMethod":           'process',
                "fileCorpus":           'corpusTemporalBreastLung/corpusTemporal.p',
                "fileValues":           'corpusTemporalBreastLung/valuesTemporalSede1.p',
                "fileVectors":          'corpusTemporalBreastLung/vectors.txt',
                "corpusKey":            'sede1',
                "batchSize":            32,
                "startEpoch":           0,
                "epochs":               100,
                "earlyStopping":        10,                   #None or patience
                "tensorboard":          'tensorBoardPytorch/confB/{}/{}/{}'.format(pfd, afl, afd),    #None or logdir
                "modelSave":            "best",
                "modelFile":            'modelsPytorch/confB/{}/{}/{}/best.pt'.format(pfd, afl, afd),      #None or dir
            })

configBbest = configB[16][2][256]

configBHS = {}
for pfd in [8,16,32,64,128,256]:
    configBHS[pfd] = {}
    for afl in [0,1,2,4]:
        configBHS[pfd][afl] = {}
        for afd in [256,512,1024,2048]:
            if afl == 0 and afd != 256: #afd useless if afl==0
                continue
            first = False
            configBHS[pfd][afl][afd] = {}
            for ad in [32,64,128,256,512]:
                configBHS[pfd][afl][afd][ad] = Conf({
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
                    "outDim":               2,
                    "learningRate":         0.001,
                    "learningRateDecay":    0.,
                    "loss":                 'categorical_crossentropy',
                    "dataMethod":           'process',
                    "fileCorpus":           'corpusTemporalBreastLung/corpusTemporal.p',
                    "fileValues":           'corpusTemporalBreastLung/valuesTemporalSede1.p',
                    "fileVectors":          'corpusTemporalBreastLung/vectors.txt',
                    "preprocess":           'load', #load or save
                    "preprocessFile":       'corpusTemporalBreastLung/corpusFilteredBalanced-sentPreprocess.p',
                    "corpusKey":            'sede1',
                    "batchSize":            32,
                    "startEpoch":           0,
                    "epochs":               100,
                    "earlyStopping":        10,                   #None or patience
                    "tensorboard":          'tensorBoardPytorch/confBHS/{}/{}/{}/{}'.format(pfd, afl, afd, ad),    #None or logdir
                    "modelSave":            "best",
                    "modelFile":            'modelsPytorch/confBHS/{}/{}/{}/{}/best.pt'.format(pfd, afl, afd, ad),      #None or dir
                    "getAttentionType":     'weighted',     #features, weights or weighted
                })

configBHSbest = configBHS[16][1][512][64]
 

configB1000 = {}
for pfd in [2,4,8,16,32,64,128,256,512]:
    configB1000[pfd] = {}
    for afl in [1,2,4,8]:
        configB1000[pfd][afl] = {}
        for afd in [2,4,8,16,32,64,128,256,512,1024,2048]:
            configB1000[pfd][afl][afd] = Conf({
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
                "outDim":               2,
                "learningRate":         0.001,
                "learningRateDecay":    0.,
                "loss":                 'categorical_crossentropy',
                "dataMethod":           'process',
                "fileCorpus":           'corpusTemporalBreastLung/corpusTemporal-1000.p',
                "fileValues":           'corpusTemporalBreastLung/valuesTemporalSede1.p',
                "fileVectors":          'corpusTemporalBreastLung/vectors.txt',
                "corpusKey":            'sede1',
                "batchSize":            32,
                "startEpoch":           0,
                "epochs":               100,
                "earlyStopping":        10,                   #None or patience
                "tensorboard":          'tensorBoardPytorch/confB1000/{}/{}/{}'.format(pfd, afl, afd),    #None or logdir
                "modelSave":            "best",
                "modelFile":            'modelsPytorch/confB1000/{}/{}/{}/best.pt'.format(pfd, afl, afd),      #None or dir
            })

#configB1000best = configB1000[128][1][256]

configBHS1000 = {}
for pfd in [4,8,16,32,64,128,256]:
    configBHS1000[pfd] = {}
    for afl in [0,1,2,4]:
        configBHS1000[pfd][afl] = {}
        for afd in [128,256,512,1024,2048]:
            if afl == 0 and afd != 128: #afd useless if afl==0
                continue
            first = False
            configBHS1000[pfd][afl][afd] = {}
            for ad in [8,16,32,64,128,256,512]:
                configBHS1000[pfd][afl][afd][ad] = Conf({
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
                    "outDim":               2,
                    "learningRate":         0.001,
                    "learningRateDecay":    0.,
                    "loss":                 'categorical_crossentropy',
                    "dataMethod":           'process',
                    "fileCorpus":           'corpusTemporalBreastLung/corpusTemporal-1000.p',
                    "fileValues":           'corpusTemporalBreastLung/valuesTemporalSede1.p',
                    "fileVectors":          'corpusTemporalBreastLung/vectors.txt',
                    "preprocess":           'load', #load or save
                    "preprocessFile":       'corpusTemporalBreastLung/corpusTemporal-1000-sentPreprocess.p',
                    "corpusKey":            'sede1',
                    "batchSize":            32,
                    "startEpoch":           0,
                    "epochs":               100,
                    "earlyStopping":        10,                   #None or patience
                    "tensorboard":          'tensorBoardPytorch/confBHS1000/{}/{}/{}/{}'.format(pfd, afl, afd, ad),    #None or logdir
                    "modelSave":            "best",
                    "modelFile":            'modelsPytorch/confBHS1000/{}/{}/{}/{}/best.pt'.format(pfd, afl, afd, ad),      #None or dir
                    "getAttentionType":     'weighted',     #features, weights or weighted
                })

#configBHS1000best = configBHS1000[64][2][1024][256]
 
configB1000b = {}
for pfd in [2,4,8,16,32,64,128,256,512]:
    configB1000b[pfd] = {}
    for afl in [1,2,4,8]:
        configB1000b[pfd][afl] = {}
        for afd in [2,4,8,16,32,64,128,256,512,1024,2048]:
            configB1000b[pfd][afl][afd] = Conf({
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
                "outDim":               2,
                "learningRate":         0.001,
                "learningRateDecay":    0.,
                "loss":                 'categorical_crossentropy',
                "dataMethod":           'process',
                "fileCorpus":           'corpusTemporalBreastLung/corpusTemporal-1000b.p',
                "fileValues":           'corpusTemporalBreastLung/valuesTemporalSede1.p',
                "fileVectors":          'corpusTemporalBreastLung/vectors.txt',
                "corpusKey":            'sede1',
                "batchSize":            32,
                "startEpoch":           0,
                "epochs":               100,
                "earlyStopping":        10,                   #None or patience
                "tensorboard":          'tensorBoardPytorch/confB1000b/{}/{}/{}'.format(pfd, afl, afd),    #None or logdir
                "modelSave":            "best",
                "modelFile":            'modelsPytorch/confB1000b/{}/{}/{}/best.pt'.format(pfd, afl, afd),      #None or dir
            })

#configB1000bBest = configB1000b[128][1][256]

configBHS1000b = {}
for pfd in [4,8,16,32,64,128,256]:
    configBHS1000b[pfd] = {}
    for afl in [0,1,2,4]:
        configBHS1000b[pfd][afl] = {}
        for afd in [128,256,512,1024,2048]:
            if afl == 0 and afd != 128: #afd useless if afl==0
                continue
            first = False
            configBHS1000b[pfd][afl][afd] = {}
            for ad in [4,8,16,32,64,128,256,512]:
                configBHS1000b[pfd][afl][afd][ad] = Conf({
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
                    "outDim":               2,
                    "learningRate":         0.001,
                    "learningRateDecay":    0.,
                    "loss":                 'categorical_crossentropy',
                    "dataMethod":           'process',
                    "fileCorpus":           'corpusTemporalBreastLung/corpusTemporal-1000b.p',
                    "fileValues":           'corpusTemporalBreastLung/valuesTemporalSede1.p',
                    "fileVectors":          'corpusTemporalBreastLung/vectors.txt',
                    "preprocess":           'load', #load or save
                    "preprocessFile":       'corpusTemporalBreastLung/corpusTemporal-1000b-sentPreprocess.p',
                    "corpusKey":            'sede1',
                    "batchSize":            32,
                    "startEpoch":           0,
                    "epochs":               100,
                    "earlyStopping":        10,                   #None or patience
                    "tensorboard":          'tensorBoardPytorch/confBHS1000b/{}/{}/{}/{}'.format(pfd, afl, afd, ad),    #None or logdir
                    "modelSave":            "best",
                    "modelFile":            'modelsPytorch/confBHS1000b/{}/{}/{}/{}/best.pt'.format(pfd, afl, afd, ad),      #None or dir
                    "getAttentionType":     'weighted',     #features, weights or weighted
                })

#configBHS1000bBest = configBHS1000b[64][2][1024][256]
 

configP = {}
for f in [0,1,2,3,4,5,6,7,8,9]:
    configP[f] = {}
    for pfd in [2,4,8,16,32,64,128,256,512]:
        configP[f][pfd] = {}
        for afl in [1,2,4,8]:
            configP[f][pfd][afl] = {}
            for afd in [2,4,8,16,32,64,128,256,512,1024,2048]:
                configP[f][pfd][afl][afd] = Conf({
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
                    "outDim":               12,
                    "learningRate":         0.001,
                    "learningRateDecay":    0.,
                    "loss":                 'categorical_crossentropy',
                    "dataMethod":           'process',
                    "fileCorpus":           'corpusTemporalBreastLung-multiclass/corpusTemporal.p',
                    "fileValues":           'corpusTemporalBreastLung-multiclass/valuesTemporalSede1.p',
                    "fileVectors":          'corpusTemporalBreastLung-multiclass/vectors.txt',
                    "corpusKey":            'site',
                    "fold":                 f,
                    "validSplit":           10,
                    "loadSplit":            True,
                    "saveSplit":            False,
                    "fileSplit":            'corpusTemporalBreastLung-multiclass/split.p',
                    "batchSize":            32,
                    "startEpoch":           0,
                    "epochs":               100,
                    "earlyStopping":        10,                   #None or patience
                    "tensorboard":          'tensorBoardPytorch/confP/{}/{}/{}/{}'.format(f,pfd, afl, afd),    #None or logdir
                    "modelSave":            "best",
                    "modelFile":            'modelsPytorch/confP/{}/{}/{}/{}/best.pt'.format(f,pfd, afl, afd),      #None or dir
                })

configPbest = configP[0][8][2][128]

configPHS = {}
for f in [0,1,2,3,4,5,6,7,8,9]:
    configPHS[f] = {}
    for pfd in [4,8,16,32,64,128,256]:
        configPHS[f][pfd] = {}
        for afl in [0,1,2,4]:
            configPHS[f][pfd][afl] = {}
            for afd in [128,256,512,1024,2048]:
                if afl == 0 and afd != 128: #afd useless if afl==0
                    continue
                first = False
                configPHS[f][pfd][afl][afd] = {}
                for ad in [4,8,16,32,64,128,256,512]:
                    configPHS[f][pfd][afl][afd][ad] = Conf({
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
                        "outDim":               12,
                        "learningRate":         0.001,
                        "learningRateDecay":    0.,
                        "loss":                 'categorical_crossentropy',
                        "dataMethod":           'process',
                        "fileCorpus":           'corpusTemporalBreastLung-multiclass/corpusTemporal.p',
                        "fileValues":           'corpusTemporalBreastLung-multiclass/valuesTemporalSede1.p',
                        "fileVectors":          'corpusTemporalBreastLung-multiclass/vectors.txt',
                        "preprocess":           'load', #load or save
                        "preprocessFile":       'corpusTemporalBreastLung-multiclass/corpusTemporal-sentPreprocess.p',
                        "corpusKey":            'site',
                        "fold":                 f,
                        "validSplit":           10,
                        "loadSplit":            True,
                        "saveSplit":            False,
                        "fileSplit":            'corpusTemporalBreastLung-multiclass/split.p',
                        "batchSize":            32,
                        "startEpoch":           0,
                        "epochs":               100,
                        "earlyStopping":        10,                   #None or patience
                        "tensorboard":          'tensorBoardPytorch/confPHS/{}/{}/{}/{}/{}'.format(f,pfd, afl, afd, ad),    #None or logdir
                        "modelSave":            "best",
                        "modelFile":            'modelsPytorch/confPHS/{}/{}/{}/{}/{}/best.pt'.format(f,pfd, afl, afd, ad),      #None or dir
                        "getAttentionType":     'weighted',     #features, weights or weighted
                    })

configPHSbest = configPHS[0][16][2][256][8]
 

configPHS2 = {}
for f in [0,1,2,3,4,5,6,7,8,9]:
    configPHS2[f] = {}
    for pfl in [1,2,4]:
        configPHS2[f][pfl] = {}
        for pfd in [4,8,16,32,64,128,256]:
            configPHS2[f][pfl][pfd] = {}
            for dfl in [1,2,4]:
                configPHS2[f][pfl][pfd][dfl] = {}
                for dfd in [4,8,16,32,64,128,256]:
                    configPHS2[f][pfl][pfd][dfl][dfd] = {}
                    for afl in [0,1,2,4]:
                        configPHS2[f][pfl][pfd][dfl][dfd][afl] = {}
                        for afd in [128,256,512,1024,2048]:
                            if afl == 0 and afd != 128: #afd useless if afl==0
                                continue
                            first = False
                            configPHS2[f][pfl][pfd][dfl][dfd][afl][afd] = {}
                            for ad in [4,8,16,32,64,128,256,512]:
                                configPHS2[f][pfl][pfd][dfl][dfd][afl][afd][ad] = Conf({
                                    "numFields":            3,
                                    "numPhrases":           10,
                                    "phraseLen":            100,
                                    "vecLen":               60,
                                    "modelType":            'model1sent',   #model1 or model2
                                    "aggregationType":      'softmax',  #max or softmax
                                    "masking":              True,
                                    "phraseFeaturesLayers": pfl,
                                    "phraseFeaturesDim":    pfd,
                                    "phraseDropout":        0.,
                                    "docFeaturesLayers":    dfl,
                                    "docFeaturesDim":       dfd,
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
                                    "outDim":               12,
                                    "learningRate":         0.001,
                                    "learningRateDecay":    0.,
                                    "loss":                 'categorical_crossentropy',
                                    "dataMethod":           'process',
                                    "fileCorpus":           'corpusTemporalBreastLung-multiclass/corpusTemporal.p',
                                    "fileValues":           'corpusTemporalBreastLung-multiclass/valuesTemporalSede1.p',
                                    "fileVectors":          'corpusTemporalBreastLung-multiclass/vectors.txt',
                                    "preprocess":           'load', #load or save
                                    "preprocessFile":       'corpusTemporalBreastLung-multiclass/corpusTemporal-sentPreprocess.p',
                                    "corpusKey":            'site',
                                    "fold":                 f,
                                    "validSplit":           10,
                                    "loadSplit":            True,
                                    "saveSplit":            False,
                                    "fileSplit":            'corpusTemporalBreastLung-multiclass/split.p',
                                    "batchSize":            32,
                                    "startEpoch":           0,
                                    "epochs":               100,
                                    "earlyStopping":        10,                   #None or patience
                                    "tensorboard":          'tensorBoardPytorch/confPHS2/{}/{}/{}/{}/{}/{}/{}/{}'.format(f,pfl,pfd,dfl,dfd, afl, afd, ad),    #None or logdir
                                    "modelSave":            "best",
                                    "modelFile":            'modelsPytorch/confPHS2/{}/{}/{}/{}/{}/{}/{}/{}/best.pt'.format(f,pfl,pfd,dfl,dfd, afl, afd, ad),      #None or dir
                                    "getAttentionType":     'weighted',     #features, weights or weighted
                                })

configPHS2best = configPHS2[0][1][32][1][4][0][128][32]


configPHS2_do = {}
for f in [0,1,2,3,4,5,6,7,8,9]:
    configPHS2_do[f] = {}
    for pfl in [1,2,4]:
        configPHS2_do[f][pfl] = {}
        for pfd in [4,8,16,32,64,128,256]:
            configPHS2_do[f][pfl][pfd] = {}
            for dfl in [1,2,4]:
                configPHS2_do[f][pfl][pfd][dfl] = {}
                for dfd in [4,8,16,32,64,128,256]:
                    configPHS2_do[f][pfl][pfd][dfl][dfd] = {}
                    for afl in [0,1,2,4]:
                        configPHS2_do[f][pfl][pfd][dfl][dfd][afl] = {}
                        for afd in [128,256,512,1024,2048]:
                            if afl == 0 and afd != 128: #afd useless if afl==0
                                continue
                            first = False
                            configPHS2_do[f][pfl][pfd][dfl][dfd][afl][afd] = {}
                            for ad in [4,8,16,32,64,128,256,512]:
                                configPHS2_do[f][pfl][pfd][dfl][dfd][afl][afd][ad] = Conf({
                                    "numFields":            3,
                                    "numPhrases":           10,
                                    "phraseLen":            100,
                                    "vecLen":               60,
                                    "modelType":            'model1sent',   #model1 or model2
                                    "aggregationType":      'softmax',  #max or softmax
                                    "masking":              True,
                                    "phraseFeaturesLayers": pfl,
                                    "phraseFeaturesDim":    pfd,
                                    "phraseDropout":        0.,
                                    "docFeaturesLayers":    dfl,
                                    "docFeaturesDim":       dfd,
                                    "docDropout":           0.,
                                    "featuresDropout":      0.5,
                                    "rnnType":              'GRU',                 #LSTM or GRU
                                    "bidirectionalMerge":   None,      #avg or max
                                    "afterFeaturesLayers":  afl,
                                    "afterFeaturesDim":     afd,
                                    "afterFeaturesOutDim":  afd,
                                    "attentionDim":         ad,
                                    "classLayers":          1,
                                    "hiddenDim":            0,                     #None equals outDim
                                    "outDim":               12,
                                    "learningRate":         0.001,
                                    "learningRateDecay":    0.,
                                    "loss":                 'categorical_crossentropy',
                                    "dataMethod":           'process',
                                    "fileCorpus":           'corpusTemporalBreastLung-multiclass/corpusTemporal.p',
                                    "fileValues":           'corpusTemporalBreastLung-multiclass/valuesTemporalSede1.p',
                                    "fileVectors":          'corpusTemporalBreastLung-multiclass/vectors.txt',
                                    "preprocess":           'load', #load or save
                                    "preprocessFile":       'corpusTemporalBreastLung-multiclass/corpusTemporal-sentPreprocess.p',
                                    "corpusKey":            'site',
                                    "fold":                 f,
                                    "validSplit":           10,
                                    "loadSplit":            True,
                                    "saveSplit":            False,
                                    "fileSplit":            'corpusTemporalBreastLung-multiclass/split.p',
                                    "batchSize":            32,
                                    "startEpoch":           0,
                                    "epochs":               100,
                                    "earlyStopping":        10,                   #None or patience
                                    "tensorboard":          'tensorBoardPytorch/confPHS2_do/{}/{}/{}/{}/{}/{}/{}/{}'.format(f,pfl,pfd,dfl,dfd, afl, afd, ad),    #None or logdir
                                    "modelSave":            "best",
                                    "modelFile":            'modelsPytorch/confPHS2_do/{}/{}/{}/{}/{}/{}/{}/{}/best.pt'.format(f,pfl,pfd,dfl,dfd, afl, afd, ad),      #None or dir
                                    "getAttentionType":     'weighted',     #features, weights or weighted
                                })

configPHS2_dobest = configPHS2_do[0][1][16][1][4][0][128][16]
 

configPHS2_350 = {}
for f in [0,1,2,3,4,5,6,7,8,9]:
    configPHS2_350[f] = {}
    for pfl in [1,2,4]:
        configPHS2_350[f][pfl] = {}
        for pfd in [4,8,16,32,64,128,256]:
            configPHS2_350[f][pfl][pfd] = {}
            for dfl in [1,2,4]:
                configPHS2_350[f][pfl][pfd][dfl] = {}
                for dfd in [4,8,16,32,64,128,256]:
                    configPHS2_350[f][pfl][pfd][dfl][dfd] = {}
                    for afl in [0,1,2,4]:
                        configPHS2_350[f][pfl][pfd][dfl][dfd][afl] = {}
                        for afd in [128,256,512,1024,2048]:
                            if afl == 0 and afd != 128: #afd useless if afl==0
                                continue
                            first = False
                            configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd] = {}
                            for ad in [4,8,16,32,64,128,256,512]:
                                configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad] = Conf({
                                    "numFields":            3,
                                    "numPhrases":           10,
                                    "phraseLen":            100,
                                    "vecLen":               350,
                                    "modelType":            'model1sent',   #model1 or model2
                                    "aggregationType":      'softmax',  #max or softmax
                                    "masking":              True,
                                    "phraseFeaturesLayers": pfl,
                                    "phraseFeaturesDim":    pfd,
                                    "phraseDropout":        0.,
                                    "docFeaturesLayers":    dfl,
                                    "docFeaturesDim":       dfd,
                                    "docDropout":           0.,
                                    "featuresDropout":      0.5,
                                    "rnnType":              'GRU',                 #LSTM or GRU
                                    "bidirectionalMerge":   None,      #avg or max
                                    "afterFeaturesLayers":  afl,
                                    "afterFeaturesDim":     afd,
                                    "afterFeaturesOutDim":  afd,
                                    "attentionDim":         ad,
                                    "classLayers":          1,
                                    "hiddenDim":            0,                     #None equals outDim
                                    "outDim":               12,
                                    "learningRate":         0.001,
                                    "learningRateDecay":    0.,
                                    "loss":                 'categorical_crossentropy',
                                    "dataMethod":           'process',
                                    "fileCorpus":           'corpusTemporalBreastLung-multiclass/corpusTemporal.p',
                                    "fileValues":           'corpusTemporalBreastLung-multiclass/valuesTemporalSede1.p',
                                    "fileVectors":          'corpusTemporalBreastLung-multiclass/vectors_350.txt',
                                    "preprocess":           'load', #load or save
                                    "preprocessFile":       'corpusTemporalBreastLung-multiclass/corpusTemporal-sentPreprocess_350.p',
                                    "corpusKey":            'site',
                                    "fold":                 f,
                                    "validSplit":           10,
                                    "loadSplit":            True,
                                    "saveSplit":            False,
                                    "fileSplit":            'corpusTemporalBreastLung-multiclass/split.p',
                                    "batchSize":            32,
        #uriel
                                    "startEpoch":           0,
                                    "epochs":               100,
                                    "earlyStopping":        10,                   #None or patience
                                    "tensorboard":          'tensorBoardPytorch/confPHS2_350/{}/{}/{}/{}/{}/{}/{}/{}'.format(f,pfl,pfd,dfl,dfd, afl, afd, ad),    #None or logdir
                                    "modelSave":            "best",
                                    "modelFile":            'modelsPytorch/confPHS2_350/{}/{}/{}/{}/{}/{}/{}/{}/best.pt'.format(f,pfl,pfd,dfl,dfd, afl, afd, ad),      #None or dir
                                    "getAttentionType":     'weighted',     #features, weights or weighted
                                })

configPHS2_350best = [
        configPHS2_350[0][1][32][1][8][0][128][64],
        configPHS2_350[1][1][128][1][128][0][128][4],
        configPHS2_350[2][1][16][1][64][0][128][4],
        configPHS2_350[3][1][16][1][32][0][128][16],
        configPHS2_350[4][1][32][1][64][0][128][4],
        configPHS2_350[5][1][32][1][128][0][128][8],
        configPHS2_350[6][1][16][1][16][0][128][32],
        configPHS2_350[7][1][32][1][8][0][128][4],
        configPHS2_350[8][1][256][1][128][0][128][512],
        configPHS2_350[9][1][32][1][32][0][128][4],
        ]
 

configPHS2_350bestTest = [
        configPHS2_350[0][1][128][1][16][0][128][256],
        configPHS2_350[1][1][32][1][8][0][128][8],
        configPHS2_350[2][1][32][1][16][0][128][8],
        configPHS2_350[3][1][64][1][32][0][128][32],
        configPHS2_350[4][1][128][1][16][0][128][32],
        configPHS2_350[5][1][128][1][128][0][128][4],
        configPHS2_350[6][1][256][1][32][0][128][512],
        configPHS2_350[7][1][16][1][8][0][128][4],
        configPHS2_350[8][1][32][1][16][0][128][64],
        configPHS2_350[9][1][64][1][64][0][128][4],
        ]

configPHS2_350_noDo = {}
for f in [0,1,2,3,4,5,6,7,8,9]:
    configPHS2_350_noDo[f] = {}
    for pfl in [1,2,4]:
        configPHS2_350_noDo[f][pfl] = {}
        for pfd in [4,8,16,32,64,128,256]:
            configPHS2_350_noDo[f][pfl][pfd] = {}
            for dfl in [1,2,4]:
                configPHS2_350_noDo[f][pfl][pfd][dfl] = {}
                for dfd in [4,8,16,32,64,128,256]:
                    configPHS2_350_noDo[f][pfl][pfd][dfl][dfd] = {}
                    for afl in [0,1,2,4]:
                        configPHS2_350_noDo[f][pfl][pfd][dfl][dfd][afl] = {}
                        for afd in [128,256,512,1024,2048]:
                            if afl == 0 and afd != 128: #afd useless if afl==0
                                continue
                            first = False
                            configPHS2_350_noDo[f][pfl][pfd][dfl][dfd][afl][afd] = {}
                            for ad in [4,8,16,32,64,128,256,512]:
                                configPHS2_350_noDo[f][pfl][pfd][dfl][dfd][afl][afd][ad] = Conf({
                                    "numFields":            3,
                                    "numPhrases":           10,
                                    "phraseLen":            100,
                                    "vecLen":               350,
                                    "modelType":            'model1sent',   #model1 or model2
                                    "aggregationType":      'softmax',  #max or softmax
                                    "masking":              True,
                                    "phraseFeaturesLayers": pfl,
                                    "phraseFeaturesDim":    pfd,
                                    "phraseDropout":        0.,
                                    "docFeaturesLayers":    dfl,
                                    "docFeaturesDim":       dfd,
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
                                    "outDim":               12,
                                    "learningRate":         0.001,
                                    "learningRateDecay":    0.,
                                    "loss":                 'categorical_crossentropy',
                                    "dataMethod":           'process',
                                    "fileCorpus":           'corpusTemporalBreastLung-multiclass/corpusTemporal.p',
                                    "fileValues":           'corpusTemporalBreastLung-multiclass/valuesTemporalSede1.p',
                                    "fileVectors":          'corpusTemporalBreastLung-multiclass/vectors_350.txt',
                                    "preprocess":           'load', #load or save
                                    "preprocessFile":       'corpusTemporalBreastLung-multiclass/corpusTemporal-sentPreprocess_350.p',
                                    "corpusKey":            'site',
                                    "fold":                 f,
                                    "validSplit":           10,
                                    "loadSplit":            True,
                                    "saveSplit":            False,
                                    "fileSplit":            'corpusTemporalBreastLung-multiclass/split.p',
                                    "batchSize":            32,
        #uriel
                                    "startEpoch":           0,
                                    "epochs":               100,
                                    "earlyStopping":        10,                   #None or patience
                                    "tensorboard":          'tensorBoardPytorch/confPHS2_350_noDo/{}/{}/{}/{}/{}/{}/{}/{}'.format(f,pfl,pfd,dfl,dfd, afl, afd, ad),    #None or logdir
                                    "modelSave":            "best",
                                    "modelFile":            'modelsPytorch/confPHS2_350_noDo/{}/{}/{}/{}/{}/{}/{}/{}/best.pt'.format(f,pfl,pfd,dfl,dfd, afl, afd, ad),      #None or dir
                                    "getAttentionType":     'weighted',     #features, weights or weighted
                                })

configPHS2_350_noDobest = configPHS2_350_noDo[0][1][32][1][32][0][128][16]
 

configPHS2_350_paper = Conf({
                                    "numFields":            3,
                                    "numPhrases":           10,
                                    "phraseLen":            100,
                                    "vecLen":               350,
                                    "modelType":            'model1sent',   #model1 or model2
                                    "aggregationType":      'softmax',  #max or softmax
                                    "masking":              True,
                                    "phraseFeaturesLayers": 1,
                                    "phraseFeaturesDim":    200,
                                    "phraseDropout":        0.,
                                    "docFeaturesLayers":    1,
                                    "docFeaturesDim":       200,
                                    "docDropout":           0.,
                                    "featuresDropout":      0.5,
                                    "rnnType":              'GRU',                 #LSTM or GRU
                                    "bidirectionalMerge":   None,      #avg or max
                                    "afterFeaturesLayers":  0,
                                    "afterFeaturesDim":     0,
                                    "afterFeaturesOutDim":  0,
                                    "attentionDim":         300,
                                    "classLayers":          1,
                                    "hiddenDim":            0,                     #None equals outDim
                                    "outDim":               12,
                                    "learningRate":         0.001,
                                    "learningRateDecay":    0.,
                                    "loss":                 'categorical_crossentropy',
                                    "dataMethod":           'process',
                                    "fileCorpus":           'corpusTemporalBreastLung-multiclass/corpusTemporal.p',
                                    "fileValues":           'corpusTemporalBreastLung-multiclass/valuesTemporalSede1.p',
                                    "fileVectors":          'corpusTemporalBreastLung-multiclass/vectors_350.txt',
                                    "preprocess":           'load', #load or save
                                    "preprocessFile":       'corpusTemporalBreastLung-multiclass/corpusTemporal-sentPreprocess_350.p',
                                    "corpusKey":            'site',
                                    "fold":                 f,
                                    "validSplit":           10,
                                    "loadSplit":            True,
                                    "saveSplit":            False,
                                    "fileSplit":            'corpusTemporalBreastLung-multiclass/split.p',
                                    "batchSize":            32,
                                    "startEpoch":           0,
                                    "epochs":               100,
                                    "earlyStopping":        10,                   #None or patience
                                    "tensorboard":          'tensorBoardPytorch/confPHS2_350_paper',    #None or logdir
                                    "modelSave":            "best",
                                    "modelFile":            'modelsPytorch/confPHS2_350_paper/best.pt',      #None or dir
                                    "getAttentionType":     'weighted',     #features, weights or weighted
                                })
 
configRunSeqPar = {
        1:[(configP[f][pfd][afl][afd], "configP-{}-{}-{}-{}".format(f,pfd,afl,afd)) for f,pfd,afl,afd in itertools.product([0],[32,16,8,4,2],[1],[512,256,128,64,32,16,8,4,2])],
        2:[(configP[f][pfd][afl][afd], "configP-{}-{}-{}-{}".format(f,pfd,afl,afd)) for f,pfd,afl,afd in itertools.product([0],[32,16,8,4,2],[2],[512,256,128,64,32,16,8,4,2])],
        3:[(configP[f][pfd][afl][afd], "configP-{}-{}-{}-{}".format(f,pfd,afl,afd)) for f,pfd,afl,afd in itertools.product([0],[32,16,8,4,2],[4],[512,256,128,64,32,16,8,4,2])],
        4:[(configP[f][pfd][afl][afd], "configP-{}-{}-{}-{}".format(f,pfd,afl,afd)) for f,pfd,afl,afd in itertools.product([0],[512,256,128,64],[1],[512,256,128,64,32,16,8,4,2])],
        5:[(configP[f][pfd][afl][afd], "configP-{}-{}-{}-{}".format(f,pfd,afl,afd)) for f,pfd,afl,afd in itertools.product([0],[512,256,128,64],[2],[512,256,128,64,32,16,8,4,2])],
        6:[(configP[f][pfd][afl][afd], "configP-{}-{}-{}-{}".format(f,pfd,afl,afd)) for f,pfd,afl,afd in itertools.product([0],[512,256,128,64],[4],[512,256,128,64,32,16,8,4,2])],

        7:[(configPHS2[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[4],[256,128,64,32,16,8,4],[4],[256,128,64,32,16,8,4],[0],[128],[512,256,128,64,32,16,8,4])],
        8:[(configPHS2[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[4],[256,128,64,32,16,8,4],[2],[256,128,64,32,16,8,4],[0],[128],[512,256,128,64,32,16,8,4])],
        9:[(configPHS2[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[4],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[512,256,128,64,32,16,8,4])],
        10:[(configPHS2[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[2],[256,128,64,32,16,8,4],[4],[256,128,64,32,16,8,4],[0],[128],[512,256,128,64,32,16,8,4])],
        11:[(configPHS2[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[2],[256,128,64,32,16,8,4],[2],[256,128,64,32,16,8,4],[0],[128],[512,256,128,64,32,16,8,4])],
        12:[(configPHS2[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[2],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[512,256,128,64,32,16,8,4])],
        13:[(configPHS2[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[1],[256,128,64,32,16,8,4],[4],[256,128,64,32,16,8,4],[0],[128],[512,256,128,64,32,16,8,4])],
        14:[(configPHS2[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[1],[256,128,64,32,16,8,4],[2],[256,128,64,32,16,8,4],[0],[128],[512,256,128,64,32,16,8,4])],
        15:[(configPHS2[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[512,256,128,64,32,16,8,4])],

        16:[(configPHS2_350_paper, "configPHS2_350_paper")],

        17:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[4],[256,128,64,32,16,8,4],[4],[256,128,64,32,16,8,4],[0],[128],[512,256,128,64,32,16,8,4])],
        18:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[4],[256,128,64,32,16,8,4],[2],[256,128,64,32,16,8,4],[0],[128],[512,256,128,64,32,16,8,4])],
        19:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[4],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[512,256,128,64,32,16,8,4])],
        20:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[2],[256,128,64,32,16,8,4],[4],[256,128,64,32,16,8,4],[0],[128],[512,256,128,64,32,16,8,4])],
        21:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[2],[256,128,64,32,16,8,4],[2],[256,128,64,32,16,8,4],[0],[128],[512,256,128,64,32,16,8,4])],
        22:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[2],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[512,256,128,64,32,16,8,4])],
        23:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[1],[256,128,64,32,16,8,4],[4],[256,128,64,32,16,8,4],[0],[128],[512,256,128,64,32,16,8,4])],
        24:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[1],[256,128,64,32,16,8,4],[2],[256,128,64,32,16,8,4],[0],[128],[512,256,128,64,32,16,8,4])],
        25:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[512,256,128,64,32,16,8,4])],


        26:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[512])],
        27:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[256])],
        28:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[128])],
        29:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[64])],
        30:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[32])],
        31:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[16])],
        32:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[8])],
        33:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[4])],

        34:[(configPHS2[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[512])],
        35:[(configPHS2[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[256])],
        36:[(configPHS2[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[128])],
        37:[(configPHS2[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[64])],
        38:[(configPHS2[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[32])],
        39:[(configPHS2[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[16])],
        40:[(configPHS2[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[8])],
        41:[(configPHS2[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[4])],


        42:[(configPHS2_do[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_do-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[512])],
        43:[(configPHS2_do[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_do-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[256])],
        44:[(configPHS2_do[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_do-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[128])],
        45:[(configPHS2_do[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_do-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[64])],
        46:[(configPHS2_do[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_do-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[32])],
        47:[(configPHS2_do[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_do-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[16])],
        48:[(configPHS2_do[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_do-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[8])],
        49:[(configPHS2_do[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_do-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([0],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[4])],

        50:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([1,2],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[512])],
        51:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([1,2],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[256])],
        52:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([1,2],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[128])],
        53:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([1,2],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[64])],
        54:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([1,2],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[32])],
        55:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([1,2],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[16])],
        56:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([1,2],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[8])],
        57:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([1,2],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[4])],



        58:[(configPHS2[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([1,2,3,4,5,6,7,8,9],[1],[32],[1],[4],[0],[128],[32])],
        59:[(configPHS2_do[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_do-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([1,2,3,4,5,6,7,8,9],[1],[16],[1],[4],[0],[128],[16])],
        60:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([1,2,3,4,5,6,7,8,9],[1],[32],[1],[8],[0],[128],[64])],
        61:[(configPHS2_350_noDo[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350_noDo-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([1,2,3,4,5,6,7,8,9],[1],[32],[1],[32],[0],[128],[16])],

        62:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([5,6,7],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[512])],
        63:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([5,6,7],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[256])],
        64:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([5,6,7],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[128])],
        65:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([5,6,7],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[64])],
        66:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([5,6,7],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[32])],
        67:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([5,6,7],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[16])],
        68:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([5,6,7],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[8])],
        69:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([5,6,7],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[4])],


        70:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([8,9],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[512])],
        71:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([8,9],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[256])],
        72:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([8,9],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[128])],
        73:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([8,9],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[64])],
        74:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([8,9],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[32])],
        75:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([8,9],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[16])],
        76:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([8,9],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[8])],
        77:[(configPHS2_350[f][pfl][pfd][dfl][dfd][afl][afd][ad], "configPHS2_350-{}-{}-{}-{}-{}-{}-{}-{}".format(f,pfl,pfd,dfl,dfd,afl,afd,ad)) for f,pfl,pfd,dfl,dfd,afl,afd,ad in itertools.product([8,9],[1],[256,128,64,32,16,8,4],[1],[256,128,64,32,16,8,4],[0],[128],[4])],


    }





configRun1 = config51
configRun2 = config50c
configRun3 = config48t
configRun4 = config49t

configRun1Continue = config50cContinue
configRun2Continue = config49tContinue


