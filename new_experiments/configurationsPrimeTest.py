from conf import Conf

config1 = Conf({
    "seqLen":               100,
    "numDigits":            100,
    "modelType":            'modelPrimeTest',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       128,
    "seqDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus1.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split1.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf1',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf1/epoch{}.pt',      #None or dir
})

config2 = Conf({
    "seqLen":               100,
    "numDigits":            100,
    "modelType":            'modelPrimeTestBase',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       128,
    "seqDropout":           0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus1.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split1.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf2',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf2/epoch{}.pt',      #None or dir
})

config3 = Conf({
    "seqLen":               100,
    "numDigits":            100,
    "modelType":            'modelPrimeTest',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       8,
    "seqDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus1.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split1.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf3',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf3/epoch{}.pt',      #None or dir
})

config4 = Conf({
    "seqLen":               100,
    "numDigits":            100,
    "modelType":            'modelPrimeTestBase',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       8,
    "seqDropout":           0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus1.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split1.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf4',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf4/epoch{}.pt',      #None or dir
})

config5 = Conf({
    "seqLen":               100,
    "numDigits":            100,
    "modelType":            'modelPrimeTest',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       1,
    "seqDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus1.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split1.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        1,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf5',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf5/epoch{}.pt',      #None or dir
})

config5Continue = config5.copy({
    "earlyStopping":        None,                   #None or patience
    "startEpoch":           20,
    "epochs":               100,
    "modelFileLoad":        'modelsPytorchPrimes/conf5/epoch19.pt',      #None or dir
    })

config6 = Conf({
    "seqLen":               100,
    "numDigits":            100,
    "modelType":            'modelPrimeTestBase',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       1,
    "seqDropout":           0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus1.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split1.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        1,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf6',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf6/epoch{}.pt',      #None or dir
})

config6Continue = config6.copy({
    "earlyStopping":        None,                   #None or patience
    "startEpoch":           15,
    "epochs":               100,
    "modelFileLoad":        'modelsPytorchPrimes/conf6/epoch14.pt',      #None or dir
    })

config7 = Conf({
    "seqLen":               100,
    "numDigits":            100,
    "modelType":            'modelPrimeTest',
    "seqFeaturesLayers":    2,
    "seqFeaturesDim":       1,
    "seqDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus1.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split1.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf7',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf7/epoch{}.pt',      #None or dir
})

config8 = Conf({
    "seqLen":               100,
    "numDigits":            100,
    "modelType":            'modelPrimeTestBase',
    "seqFeaturesLayers":    2,
    "seqFeaturesDim":       1,
    "seqDropout":           0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus1.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split1.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf8',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf8/epoch{}.pt',      #None or dir
})

config9 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTest',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       2,
    "seqDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus2.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split2.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf9',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf9/epoch{}.pt',      #None or dir
})

config10 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTestBase',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       2,
    "seqDropout":           0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus2.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split2.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf10',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf10/epoch{}.pt',      #None or dir
})

config11 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTest',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       2,
    "seqDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus3.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split3.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf11',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf11/epoch{}.pt',      #None or dir
})

config12 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTestBase',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       2,
    "seqDropout":           0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus3.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split3.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf12',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf12/epoch{}.pt',      #None or dir
})

config13 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTest',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       16,
    "seqDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus3.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split3.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf13',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf13/epoch{}.pt',      #None or dir
})

config14 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTestBase',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       16,
    "seqDropout":           0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus3.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split3.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf14',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf14/epoch{}.pt',      #None or dir
})

config15 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTest',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       32,
    "seqDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus3.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split3.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf15',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf15/epoch{}.pt',      #None or dir
})

config16 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTestBase',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       32,
    "seqDropout":           0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus3.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split3.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf16',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf16/epoch{}.pt',      #None or dir
})

config17 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTest',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       64,
    "seqDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus3.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split3.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf17',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf17/epoch{}.pt',      #None or dir
})

config18 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTestBase',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       64,
    "seqDropout":           0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus3.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split3.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf18',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf18/epoch{}.pt',      #None or dir
})

config19 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTest',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       128,
    "seqDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus3.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split3.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf19',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf19/epoch{}.pt',      #None or dir
})

config20 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTestBase',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       128,
    "seqDropout":           0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus3.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split3.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf20',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf20/epoch{}.pt',      #None or dir
})

config21 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTest',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       256,
    "seqDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus3.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split3.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf21',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf21/epoch{}.pt',      #None or dir
})

config22 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTestBase',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       256,
    "seqDropout":           0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus3.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split3.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf22',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf22/epoch{}.pt',      #None or dir
})

config23 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTest',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       512,
    "seqDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus3.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split3.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf23',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf23/epoch{}.pt',      #None or dir
})

config24 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTestBase',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       512,
    "seqDropout":           0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus3.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split3.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf24',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf24/epoch{}.pt',      #None or dir
})

config25 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTest',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       32,
    "seqDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'LSTM',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus3.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split3.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf25',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf25/epoch{}.pt',      #None or dir
})

config26 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTestBase',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       32,
    "seqDropout":           0.,
    "rnnType":              'LSTM',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus3.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split3.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf26',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf26/epoch{}.pt',      #None or dir
})

#config27 and 28 good difference
config27 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTest',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       32,
    "seqDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'LSTM',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus4.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split4.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf27',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf27/epoch{}.pt',      #None or dir
})

config28 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTestBase',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       32,
    "seqDropout":           0.,
    "rnnType":              'LSTM',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus4.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split4.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf28',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf28/epoch{}.pt',      #None or dir
})

#like 27 and 28
config29 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTest',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       32,
    "seqDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus4.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split4.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf29',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf29/epoch{}.pt',      #None or dir
})

config30 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTestBase',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       32,
    "seqDropout":           0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus4.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split4.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf30',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf30/epoch{}.pt',      #None or dir
})

config31 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTestOrderBeta',
    "bistep":               None,
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       1,
    "seqDropout":           0.,
    "sortSeqFeaturesLayers":1,
    "sortSeqFeaturesDim":   32,
    "sortSeqDropout":       0.,
    "featuresDropout":      0.,
    "rnnType":              'LSTM',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus5.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split5.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf31',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf31/epoch{}.pt',      #None or dir
})

config32 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTestBase',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       32,
    "seqDropout":           0.,
    "rnnType":              'LSTM',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus5.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split5.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf32',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf32/epoch{}.pt',      #None or dir
})

config33 = Conf({
    "seqLen":               100,
    "numDigits":            100,
    "modelType":            'modelPrimeTestOrderBeta',
    "bistep":               None,
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       1,
    "seqDropout":           0.,
    "sortSeqFeaturesLayers":1,
    "sortSeqFeaturesDim":   32,
    "sortSeqDropout":       0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus6.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split6.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf33',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf33/epoch{}.pt',      #None or dir
})

config34 = Conf({
    "seqLen":               100,
    "numDigits":            100,
    "modelType":            'modelPrimeTestBase',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       32,
    "seqDropout":           0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus6.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split6.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf34',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf34/epoch{}.pt',      #None or dir
})

#config35 and 36 difference (WRONG, only initially)
config35 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTestOrderBeta',
    "bistep":               None,
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       1,
    "seqDropout":           0.,
    "sortSeqFeaturesLayers":1,
    "sortSeqFeaturesDim":   32,
    "sortSeqDropout":       0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus5.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split5.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf35',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf35/epoch{}.pt',      #None or dir
})

config35bis = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTest',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       32,
    "seqDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus5.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split5.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf35bis',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf35bis/epoch{}.pt',      #None or dir
})

config36 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTestBase',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       32,
    "seqDropout":           0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus5.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split5.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf36',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf36/epoch{}.pt',      #None or dir
})

config37 = Conf({
    "seqLen":               100,
    "numDigits":            100,
    "modelType":            'modelPrimeTestOrderBeta',
    "bistep":               None,
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       1,
    "seqDropout":           0.,
    "sortSeqFeaturesLayers":1,
    "sortSeqFeaturesDim":   32,
    "sortSeqDropout":       0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus7.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split7.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf37',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf37/epoch{}.pt',      #None or dir
})

config38 = Conf({
    "seqLen":               100,
    "numDigits":            100,
    "modelType":            'modelPrimeTestBase',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       32,
    "seqDropout":           0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus7.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split7.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf38',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf38/epoch{}.pt',      #None or dir
})

#config 39, 40, 41 good difference
config39 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTestOrderBeta',
    "bistep":               None,       #None, random or split
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       1,
    "seqDropout":           0.,
    "sortSeqFeaturesLayers":1,
    "sortSeqFeaturesDim":   32,
    "sortSeqDropout":       0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus8.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split8.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf39',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf39/epoch{}.pt',      #None or dir
})

#good bis, no good ter quater
config39bis = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTestOrder',
    "bistep":               None,       #None, random or split
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       1,
    "seqDropout":           0.,
    "sortSeqFeaturesLayers":1,
    "sortSeqFeaturesDim":   32,
    "sortSeqDropout":       0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus8.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split8.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf39bis',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf39bis/epoch{}.pt',      #None or dir
})

config39ter = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTestOrder',
    "bistep":               'random',       #None, random or split
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       1,
    "seqDropout":           0.,
    "sortSeqFeaturesLayers":1,
    "sortSeqFeaturesDim":   32,
    "sortSeqDropout":       0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus8.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split8.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf39ter',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf39ter/epoch{}.pt',      #None or dir
})

config39quater = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTestOrder',
    "bistep":               'split',       #None, random or split
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       1,
    "seqDropout":           0.,
    "sortSeqFeaturesLayers":1,
    "sortSeqFeaturesDim":   32,
    "sortSeqDropout":       0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus8.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split8.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf39quater',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf39quater/epoch{}.pt',      #None or dir
})


config40 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTest',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       32,
    "seqDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus8.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split8.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf40',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf40/epoch{}.pt',      #None or dir
})

config41 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTestBase',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       32,
    "seqDropout":           0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus8.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split8.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf41',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf41/epoch{}.pt',      #None or dir
})

config41Continue = config41.copy({
    "earlyStopping":        10,                   #None or patience
    "startEpoch":           11,
    "epochs":               100,
    "modelFileLoad":        'modelsPytorchPrimes/conf41/epoch10.pt',      #None or dir
    })


#good
config42 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTestOrder',
    "bistep":               None,       #None, random or split
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       32,
    "seqDropout":           0.,
    "sortSeqFeaturesLayers":1,
    "sortSeqFeaturesDim":   32,
    "sortSeqDropout":       0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus8.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split8.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf42',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf42/epoch{}.pt',      #None or dir
})

config43 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTestOrder',
    "bistep":               'random',       #None, random or split
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       32,
    "seqDropout":           0.,
    "sortSeqFeaturesLayers":1,
    "sortSeqFeaturesDim":   32,
    "sortSeqDropout":       0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus8.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split8.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf43',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf43/epoch{}.pt',      #None or dir
})

config44 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTestOrder',
    "bistep":               'split',       #None, random or split
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       32,
    "seqDropout":           0.,
    "sortSeqFeaturesLayers":1,
    "sortSeqFeaturesDim":   32,
    "sortSeqDropout":       0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus8.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split8.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf44',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf44/epoch{}.pt',      #None or dir
})

config45 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTestOrder',
    "bistep":               None,       #None, random or split
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       16,
    "seqDropout":           0.,
    "sortSeqFeaturesLayers":1,
    "sortSeqFeaturesDim":   2,
    "sortSeqDropout":       0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus8.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split8.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf45',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf45/epoch{}.pt',      #None or dir
})

config46 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTestOrder',
    "bistep":               'random',       #None, random or split
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       16,
    "seqDropout":           0.,
    "sortSeqFeaturesLayers":1,
    "sortSeqFeaturesDim":   2,
    "sortSeqDropout":       0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus8.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split8.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf46',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf46/epoch{}.pt',      #None or dir
})

config47 = Conf({
    "seqLen":               1000,
    "numDigits":            100,
    "modelType":            'modelPrimeTestOrder',
    "bistep":               'split',       #None, random or split
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       16,
    "seqDropout":           0.,
    "sortSeqFeaturesLayers":1,
    "sortSeqFeaturesDim":   2,
    "sortSeqDropout":       0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus8.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split8.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf47',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf47/epoch{}.pt',      #None or dir
})


#config40 and 41 for c=100
config48 = {}
config49 = {}
config48[100] = config40
config49[100] = config41
for c in [20,30,40,50,60,70,80,90,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260]:
    config48[c] = Conf({
        "seqLen":               1000,
        "numDigits":            100,
        "modelType":            'modelPrimeTest',
        "seqFeaturesLayers":    1,
        "seqFeaturesDim":       32,
        "seqDropout":           0.,
        "featuresDropout":      0.,
        "rnnType":              'GRU',                 #LSTM or GRU
        "bidirectionalMerge":   'avg',      #avg or max
        "afterFeaturesLayers":  0,
        "classLayers":          1,
        "hiddenDim":            0,                     #None equals outDim
        "outDim":               1,
        "learningRate":         0.001,
        "learningRateDecay":    0.,
        "loss":                 'binary_crossentropy',
        "dataMethod":           'load',
        "fileCorpus":           'corpusPrimes/corpus58_{}.npz'.format(c),
        "trainSplit":           80,
        "validSplit":           50,
        "loadSplit":            True,
        "saveSplit":            False,
        "fileSplit":            'corpusPrimes/split58_{}.p'.format(c),
        "batchSize":            64,
        "startEpoch":           0,
        "epochs":               100,
        "earlyStopping":        10,                   #None or patience
        "tensorboard":          'tensorBoardPytorchPrimes/conf48_{}'.format(c),    #None or logdir
        "modelFile":            'modelsPytorchPrimes/conf40_{}/epoch{{}}.pt'.format(c),      #None or dir
    })

    config49[c] = Conf({
        "seqLen":               1000,
        "numDigits":            100,
        "modelType":            'modelPrimeTestBase',
        "seqFeaturesLayers":    1,
        "seqFeaturesDim":       32,
        "seqDropout":           0.,
        "rnnType":              'GRU',                 #LSTM or GRU
        "bidirectionalMerge":   'avg',      #avg or max
        "classLayers":          1,
        "hiddenDim":            0,                     #None equals outDim
        "outDim":               1,
        "learningRate":         0.001,
        "learningRateDecay":    0.,
        "loss":                 'binary_crossentropy',
        "dataMethod":           'load',
        "fileCorpus":           'corpusPrimes/corpus58_{}.npz'.format(c),
        "trainSplit":           80,
        "validSplit":           50,
        "loadSplit":            True,
        "saveSplit":            False,
        "fileSplit":            'corpusPrimes/split58_{}.p'.format(c),
        "batchSize":            64,
        "startEpoch":           0,
        "epochs":               100,
        "earlyStopping":        10,                   #None or patience
        "tensorboard":          'tensorBoardPytorchPrimes/conf49_{}'.format(c),    #None or logdir
        "modelFile":            'modelsPytorchPrimes/conf49_{}/epoch{{}}.pt'.format(c),      #None or dir
    })

config49_60Continue = config49[60].copy({
    "earlyStopping":        10,                   #None or patience
    "startEpoch":           8,
    "epochs":               100,
    "modelFileLoad":        'modelsPytorchPrimes/conf49_60/epoch7.pt',      #None or dir
    })

config49_80Continue = config49[80].copy({
    "earlyStopping":        10,                   #None or patience
    "startEpoch":           12,
    "epochs":               100,
    "modelFileLoad":        'modelsPytorchPrimes/conf49_80/epoch11.pt',      #None or dir
    })

config49_110Continue = config49[110].copy({
    "earlyStopping":        10,                   #None or patience
    "startEpoch":           20,
    "epochs":               100,
    "modelFileLoad":        'modelsPytorchPrimes/conf49_110/epoch19.pt',      #None or dir
    })

config49_130Continue = config49[130].copy({
    "earlyStopping":        10,                   #None or patience
    "startEpoch":           18,
    "epochs":               100,
    "modelFileLoad":        'modelsPytorchPrimes/conf49_130/epoch17.pt',      #None or dir
    })

config49_140Continue = config49[140].copy({
    "earlyStopping":        10,                   #None or patience
    "startEpoch":           25,
    "epochs":               100,
    "modelFileLoad":        'modelsPytorchPrimes/conf49_140/epoch24.pt',      #None or dir
    })

config49_150Continue = config49[150].copy({
    "earlyStopping":        10,                   #None or patience
    "startEpoch":           40,
    "epochs":               100,
    "modelFileLoad":        'modelsPytorchPrimes/conf49_150/epoch39.pt',      #None or dir
    })

config49_160Continue = config49[160].copy({
    "earlyStopping":        10,                   #None or patience
    "startEpoch":           25,
    "epochs":               100,
    "modelFileLoad":        'modelsPytorchPrimes/conf49_160/epoch24.pt',      #None or dir
    })

config49_180Continue = config49[180].copy({
    "earlyStopping":        10,                   #None or patience
    "startEpoch":           26,
    "epochs":               100,
    "modelFileLoad":        'modelsPytorchPrimes/conf49_180/epoch25.pt',      #None or dir
    })

config49_200Continue = config49[200].copy({
    "earlyStopping":        None,                   #None or patience
    "startEpoch":           21,
    "epochs":               29,
    "modelFileLoad":        'modelsPytorchPrimes/conf49_200/epoch20.pt',      #None or dir
    })

config49_210Continue = config49[210].copy({
    "earlyStopping":        None,                   #None or patience
    "startEpoch":           50,
    "epochs":               10,
    "modelFileLoad":        'modelsPytorchPrimes/conf49_210/epoch49.pt',      #None or dir
    })

config49_230Continue = config49[230].copy({
    "earlyStopping":        None,                   #None or patience
    "startEpoch":           50,
    "epochs":               10,
    "modelFileLoad":        'modelsPytorchPrimes/conf49_230/epoch49.pt',      #None or dir
    })

config49_240Continue = config49[240].copy({
    "earlyStopping":        None,                   #None or patience
    "startEpoch":           50,
    "epochs":               10,
    "modelFileLoad":        'modelsPytorchPrimes/conf49_240/epoch49.pt',      #None or dir
    })

config50 = Conf({
    "seqLen":               100,
    "numDigits":            10,
    "modelType":            'modelPrimeTest',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       32,
    "seqDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus9.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split9.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf50',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf50/epoch{}.pt',      #None or dir
    "modelFileLoad":        'modelsPytorchPrimes/conf50/epoch65.pt',      #None or dir
})

config50d1 = config50.copy({
    "seqFeaturesDim":       4,
    "tensorboard":          'tensorBoardPytorchPrimes/conf50d1',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf50d1/epoch{}.pt',      #None or dir
    "modelFileLoad":        'modelsPytorchPrimes/conf50d1/epoch80.pt',
})


config51 = Conf({
    "seqLen":               100,
    "numDigits":            10,
    "modelType":            'modelPrimeTestBase',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       32,
    "seqDropout":           0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus9.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split9.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf51',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf51/epoch{}.pt',      #None or dir
})

config50l = {}
config51l = {}
config50l[100] = config50
config51l[100] = config51
for l in [200,300,400,500,600,700,800,900,1000]:
    config50l[l] = config50.copy({
        "seqLen":               l,
        "fileCorpus":           'corpusPrimes/corpus9l_{}.npz'.format(l),
        "fileSplit":            'corpusPrimes/split9l_{}.p'.format(l),
        "tensorboard":          'tensorBoardPytorchPrimes/conf50l_{}'.format(l),    #None or logdir
        "modelFile":            'modelsPytorchPrimes/conf50l_{}/epoch{{}}.pt'.format(l),      #None or dir
        })

    config51l[l] = config51.copy({
        "seqLen":               l,
        "fileCorpus":           'corpusPrimes/corpus9l_{}.npz'.format(l),
        "fileSplit":            'corpusPrimes/split9l_{}.p'.format(l),
        "tensorboard":          'tensorBoardPytorchPrimes/conf51l_{}'.format(l),    #None or logdir
        "modelFile":            'modelsPytorchPrimes/conf51l_{}/epoch{{}}.pt'.format(l),      #None or dir
        })

config51lContinue = {}
config51lContinue[300] = config51l[300].copy({
    "earlyStopping":        None,                   #None or patience
    "startEpoch":           10,
    "epochs":               90,
    "modelFileLoad":        'modelsPytorchPrimes/conf51l_300/epoch9.pt',      #None or dir
    })
config51lContinue[200] = config51l[200].copy({
    "earlyStopping":        None,                   #None or patience
    "startEpoch":           72,
    "epochs":               28,
    "modelFileLoad":        'modelsPytorchPrimes/conf51l_200/epoch71.pt',      #None or dir
    })
config51lContinue[400] = config51l[400].copy({
    "earlyStopping":        None,                   #None or patience
    "startEpoch":           27,
    "epochs":               73,
    "modelFileLoad":        'modelsPytorchPrimes/conf51l_400/epoch26.pt',      #None or dir
    })
config51lContinue[500] = config51l[500].copy({
    "earlyStopping":        None,                   #None or patience
    "startEpoch":           17,
    "epochs":               83,
    "modelFileLoad":        'modelsPytorchPrimes/conf51l_500/epoch16.pt',      #None or dir
    })
config51lContinue[600] = config51l[600].copy({
    "earlyStopping":        None,                   #None or patience
    "startEpoch":           12,
    "epochs":               88,
    "modelFileLoad":        'modelsPytorchPrimes/conf51l_600/epoch11.pt',      #None or dir
    })
config51lContinue[700] = config51l[700].copy({
    "earlyStopping":        None,                   #None or patience
    "startEpoch":           9,
    "epochs":               91,
    "modelFileLoad":        'modelsPytorchPrimes/conf51l_700/epoch8.pt',      #None or dir
    })
config51lContinue[800] = config51l[800].copy({
    "earlyStopping":        None,                   #None or patience
    "startEpoch":           24,
    "epochs":               76,
    "modelFileLoad":        'modelsPytorchPrimes/conf51l_800/epoch23.pt',      #None or dir
    })

config51LSTM = Conf({
    "seqLen":               100,
    "numDigits":            10,
    "modelType":            'modelPrimeTestBase',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       32,
    "seqDropout":           0.,
    "rnnType":              'LSTM',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus9.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split9.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf51LSTM',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf51LSTM/epoch{}.pt',      #None or dir
})

config51LSTMl = {}
config51LSTMl[100] = config51LSTM
for l in [200,300,400,500,600,700,800,900,1000]:
    config51LSTMl[l] = config51LSTM.copy({
        "seqLen":               l,
        "fileCorpus":           'corpusPrimes/corpus9l_{}.npz'.format(l),
        "fileSplit":            'corpusPrimes/split9l_{}.p'.format(l),
        "tensorboard":          'tensorBoardPytorchPrimes/conf51LSTMl_{}'.format(l),    #None or logdir
        "modelFile":            'modelsPytorchPrimes/conf51LSTMl_{}/epoch{{}}.pt'.format(l),      #None or dir
        })



config52 = Conf({
    "seqLen":               100,
    "numDigits":            10,
    "modelType":            'modelPrimeTest',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       32,
    "seqDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "afterFeaturesLayers":  0,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus10.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split10.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf52',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf52/epoch{}.pt',      #None or dir
})

config53 = Conf({
    "seqLen":               100,
    "numDigits":            10,
    "modelType":            'modelPrimeTestBase',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       32,
    "seqDropout":           0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   'avg',      #avg or max
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus10.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split10.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf53',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf53/epoch{}.pt',      #None or dir
})

config54 = Conf({
    "seqLen":               100,
    "numDigits":            10,
    "modelType":            'modelPrimeTest',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       32,
    "seqDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   None,      #avg or max
    "afterFeaturesLayers":  2,
    "afterFeaturesDim":     16,
    "afterFeaturesOutDim":  1,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus9.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split9.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf54',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf54/epoch{}.pt',      #None or dir
})

config55 = Conf({
    "seqLen":               100,
    "numDigits":            10,
    "modelType":            'modelPrimeTest',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       32,
    "seqDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   None,      #avg or max
    "afterFeaturesLayers":  4,
    "afterFeaturesDim":     16,
    "afterFeaturesOutDim":  1,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus9.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split9.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf55',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf55/epoch{}.pt',      #None or dir
})

#good interpretability
config56 = Conf({
    "seqLen":               100,
    "numDigits":            10,
    "modelType":            'modelPrimeTest',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       32,
    "seqDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   None,      #avg or max
    "afterFeaturesLayers":  8,
    "afterFeaturesDim":     16,
    "afterFeaturesOutDim":  1,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus9.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split9.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf56',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf56/epoch{}.pt',      #None or dir
    "modelFileLoad":        'modelsPytorchPrimes/conf56/epoch149.pt',
})

config56Continue = config56.copy({
    "startEpoch":           99,
    "epochs":               100,
    "modelFileLoad":        'modelsPytorchPrimes/conf56/epoch99.pt',      #None or dir
    })

#good interpretability, not allways
config57 = Conf({
    "seqLen":               100,
    "numDigits":            10,
    "modelType":            'modelPrimeTest',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       32,
    "seqDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   None,      #avg or max
    "afterFeaturesLayers":  8,
    "afterFeaturesDim":     8,
    "afterFeaturesOutDim":  1,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus9.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split9.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf57',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf57/epoch{}.pt',      #None or dir
    "modelFileLoad":        'modelsPytorchPrimes/conf57/epoch99.pt',
})

config57Continue = config57.copy({
    "startEpoch":           99,
    "epochs":               100,
    "modelFileLoad":        'modelsPytorchPrimes/conf57/epoch99.pt',      #None or dir
    })

config58 = Conf({
    "seqLen":               100,
    "numDigits":            10,
    "modelType":            'modelPrimeTest',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       16,
    "seqDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   None,      #avg or max
    "afterFeaturesLayers":  16,
    "afterFeaturesDim":     8,
    "afterFeaturesOutDim":  1,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus9.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split9.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf58',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf58/epoch{}.pt',      #None or dir
})

config59 = Conf({
    "seqLen":               100,
    "numDigits":            10,
    "modelType":            'modelPrimeTest',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       16,
    "seqDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   None,      #avg or max
    "afterFeaturesLayers":  8,
    "afterFeaturesDim":     16,
    "afterFeaturesOutDim":  1,
    "classLayers":          1,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus9.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split9.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf59',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf59/epoch{}.pt',      #None or dir
})

config60 = Conf({
    "seqLen":               100,
    "numDigits":            10,
    "modelType":            'modelPrimeTest',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       16,
    "seqDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   None,      #avg or max
    "afterFeaturesLayers":  8,
    "afterFeaturesDim":     16,
    "afterFeaturesOutDim":  1,
    "classLayers":          0,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus9.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split9.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        5,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf60',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf60/epoch{}.pt',      #None or dir
    "modelFileLoad":        'modelsPytorchPrimes/conf60/epoch105.pt',      #None or dir
})

config60Continue = config60.copy({
    "startEpoch":           99,
    "epochs":               100,
    "modelFileLoad":        'modelsPytorchPrimes/conf60/epoch99.pt',      #None or dir
    })

config61 = {}
for sfd in [4,8,16,32]:
    config61[sfd] = {}
    for afl in [2,4,8]:
        config61[sfd][afl] = {}
        for afd in [2,4,8,16]:
            config61[sfd][afl][afd] = Conf({
                "seqLen":               100,
                "numDigits":            10,
                "modelType":            'modelPrimeTest',
                "seqFeaturesLayers":    1,
                "seqFeaturesDim":       sfd,
                "seqDropout":           0.,
                "featuresDropout":      0.,
                "rnnType":              'GRU',                 #LSTM or GRU
                "bidirectionalMerge":   None,      #avg or max
                "afterFeaturesLayers":  afl,
                "afterFeaturesDim":     afd,
                "afterFeaturesOutDim":  1,
                "classLayers":          0,
                "hiddenDim":            0,                     #None equals outDim
                "outDim":               1,
                "learningRate":         0.001,
                "learningRateDecay":    0.,
                "loss":                 'binary_crossentropy',
                "dataMethod":           'load',
                "fileCorpus":           'corpusPrimes/corpus9.npz',
                "trainSplit":           80,
                "validSplit":           50,
                "loadSplit":            True,
                "saveSplit":            False,
                "fileSplit":            'corpusPrimes/split9.p',
                "batchSize":            64,
                "startEpoch":           0,
                "epochs":               200,
                "earlyStopping":        10,                   #None or patience
                "tensorboard":          'tensorBoardPytorchPrimes/conf61/{}/{}/{}'.format(sfd,afl,afd),    #None or logdir
                "modelFile":            'modelsPytorchPrimes/conf61/{}/{}/{}/epoch{{}}.pt'.format(sfd,afl,afd),      #None or dir
            })

config61l1 = {}
config61l2 = {}
config61l1[100] = config61[32][2][8]
config61l2[100] = config61[32][4][8]
for l in [200,300,400,500,600,700,800,900,1000]:
    config61l1[l] = config61[32][2][8].copy({
        "seqLen":               l,
        "fileCorpus":           'corpusPrimes/corpus9l_{}.npz'.format(l),
        "fileSplit":            'corpusPrimes/split9l_{}.p'.format(l),
        "tensorboard":          'tensorBoardPytorchPrimes/conf61l1_{}'.format(l),    #None or logdir
        "modelFile":            'modelsPytorchPrimes/conf61l1_{}/epoch{{}}.pt'.format(l),      #None or dir
        })
    config61l2[l] = config61[32][4][8].copy({
        "seqLen":               l,
        "fileCorpus":           'corpusPrimes/corpus9l_{}.npz'.format(l),
        "fileSplit":            'corpusPrimes/split9l_{}.p'.format(l),
        "tensorboard":          'tensorBoardPytorchPrimes/conf61l2_{}'.format(l),    #None or logdir
        "modelFile":            'modelsPytorchPrimes/conf61l2_{}/epoch{{}}.pt'.format(l),      #None or dir
        })

config61l1Continue = {}
config61l1Continue[900] = config61l1[900].copy({
    "earlyStopping":        None,                   #None or patience
    "startEpoch":           24,
    "epochs":               76,
    "modelFileLoad":        'modelsPytorchPrimes/conf61l1_900/epoch23.pt',      #None or dir
    })
c

config62 = config61l1[100].copy({
    "bidirectionalMerge":   'avg',      #avg or max
    "tensorboard":          'tensorBoardPytorchPrimes/conf62',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf62/epoch{}.pt',      #None or dir
})

config63 = Conf({
    "seqLen":               100,
    "numDigits":            10,
    "modelType":            'modelPrimeTest',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       32,
    "seqDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   None,      #avg or max
    "afterFeaturesLayers":  1,
    "afterFeaturesDim":     0,
    "afterFeaturesOutDim":  1,
    "classLayers":          0,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus9.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split9.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf63',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf63/epoch{}.pt',      #None or dir
    "modelFileLoad":        'modelsPytorchPrimes/conf63/epoch99.pt',      #None or dir
})

config63l = {}
config63l[100] = config63
for l in [200,300,400,500,600,700,800,900,1000]:
    config63l[l] = config63.copy({
        "seqLen":               l,
        "fileCorpus":           'corpusPrimes/corpus9l_{}.npz'.format(l),
        "fileSplit":            'corpusPrimes/split9l_{}.p'.format(l),
        "tensorboard":          'tensorBoardPytorchPrimes/conf63l_{}'.format(l),    #None or logdir
        "modelFile":            'modelsPytorchPrimes/conf63l_{}/epoch{{}}.pt'.format(l),      #None or dir
        "modelFileLoad":        'modelsPytorchPrimes/conf63l_{}/epoch99.pt'.format(l),      #None or dir
        })

config64 = Conf({
    "seqLen":               100,
    "numDigits":            10,
    "modelType":            'modelPrimeTestSoftmax',
    "seqFeaturesLayers":    1,
    "seqFeaturesDim":       32,
    "seqDropout":           0.,
    "featuresDropout":      0.,
    "rnnType":              'GRU',                 #LSTM or GRU
    "bidirectionalMerge":   None,      #avg or max
    "afterFeaturesLayers":  1,
    "afterFeaturesDim":     0,
    "afterFeaturesOutDim":  1,
    "attentionDim":         1,
    "classLayers":          0,
    "hiddenDim":            0,                     #None equals outDim
    "outDim":               1,
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "loss":                 'binary_crossentropy',
    "dataMethod":           'load',
    "fileCorpus":           'corpusPrimes/corpus9.npz',
    "trainSplit":           80,
    "validSplit":           50,
    "loadSplit":            True,
    "saveSplit":            False,
    "fileSplit":            'corpusPrimes/split9.p',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "earlyStopping":        None,                   #None or patience
    "tensorboard":          'tensorBoardPytorchPrimes/conf64',    #None or logdir
    "modelFile":            'modelsPytorchPrimes/conf64/epoch{}.pt',      #None or dir
    "modelFileLoad":        'modelsPytorchPrimes/conf64/epoch55.pt',      #None or dir
    "getAttentionType":     'weighted',     #features, weights or weighted
})

config64l = {}
config64l[100] = config64
for l in [200,300,400,500,600,700,800,900,1000]:
    config64l[l] = config64.copy({
        "seqLen":               l,
        "fileCorpus":           'corpusPrimes/corpus9l_{}.npz'.format(l),
        "fileSplit":            'corpusPrimes/split9l_{}.p'.format(l),
        "tensorboard":          'tensorBoardPytorchPrimes/conf64l_{}'.format(l),    #None or logdir
        "modelFile":            'modelsPytorchPrimes/conf64l_{}/epoch{{}}.pt'.format(l),      #None or dir
        #"modelFileLoad":        'modelsPytorchPrimes/conf64l_{}/epoch99.pt'.format(l),      #None or dir
        })


configRun1 = config64l[1000]
configRun2 = config64l[200]
configRun3 = config64l[300]
configRun4 = config64l[400]
configRun5 = config64l[500]
configRun6 = config64l[600]
configRun7 = config64l[700]
configRun8 = config64l[800]
configRun9 = config64l[900]

configRun10 = config63l[1000]
configRun11 = config61l2[300]
configRun12 = config61l2[400]
configRun13 = config61l2[500]
configRun14 = config61l2[600]
configRun15 = config61l2[700]
configRun16 = config61l2[800]
configRun17 = config61l2[900]
configRun18 = config61l2[1000]

configRun19 = config61[32][4][8]
configRun20 = config61[32][4][16]
configRun21 = config61[32][8][2]
configRun22 = config61[32][8][4]
configRun23 = config61[32][8][8]
configRun24 = config61[32][8][16]


configRun1Continue = config61l1Continue[900] 
configRun2Continue = config57Continue
configRun3Continue = config51lContinue[600]
configRun4Continue = config51lContinue[400]
configRun5Continue = config51lContinue[700]

