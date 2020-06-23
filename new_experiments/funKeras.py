import numpy as np 
import random
import pickle
import keras.models
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Masking
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.layers import TimeDistributed
from keras.layers import Reshape
from keras.layers import Bidirectional
from keras.layers import Dropout
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras import initializers
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from keras.models import load_model
from keras.callbacks import EarlyStopping, TensorBoard

def makeModel(conf):
    model = Sequential()

    model.add(Reshape(target_shape = (conf.numFields * conf.numPhrases, conf.phraseLen, conf.vecLen)))

    #phrase level
    for layer in range(conf.featuresLayers):
        model.add(TimeDistributed(Bidirectional(LSTM(conf.featuresDim, dropout=conf.dropout, recurrent_dropout=conf.dropout, return_sequences=True))))

        for _ in range(conf.afterFeaturesLayers):
            model.add(TimeDistributed(Dense(conf.featuresDim*2, activation='relu')))

    model.add(TimeDistributed(GlobalMaxPooling1D()))

    model.add(Reshape(target_shape = (conf.numFields, conf.numPhrases, conf.featuresDim*2)))

    #field level
    for layer in range(conf.featuresLayers):
        model.add(TimeDistributed(Bidirectional(LSTM(conf.featuresDim, dropout=conf.dropout, recurrent_dropout=conf.dropout, return_sequences=True))))

        for _ in range(conf.afterFeaturesLayers):
            model.add(TimeDistributed(Dense(conf.featuresDim*2, activation='relu')))

    model.add(TimeDistributed(GlobalMaxPooling1D()))

    #doc level
    for layer in range(conf.featuresLayers):
        model.add(Bidirectional(LSTM(conf.featuresDim, dropout=conf.dropout, recurrent_dropout=conf.dropout, return_sequences=True)))

        for _ in range(conf.afterFeaturesLayers):
            model.add(Dense(conf.featuresDim*2, activation='relu'))

    model.add(GlobalMaxPooling1D())

    #classification
    model.add(Dropout(conf.dropout))
    for _ in range(conf.hiddenLayers):
        if conf.hiddenDim is None:
            model.add(Dense(conf.outDim, activation='relu'))
        else:
            model.add(Dense(conf.hiddenDim, activation='relu'))

    if conf.outDim == 1:
        model.add(Dense(conf.outDim, activation='sigmoid'))
    else:
        model.add(Dense(conf.outDim, activation='softmax'))

    #opt = RMSprop(lr=conf.learningRate, decay=conf.learningRateDecay)
    opt = Adam(lr=conf.learningRate, decay=conf.learningRateDecay)
    model.compile(loss=conf.loss, optimizer=opt, metrics=['accuracy'])
    
    return model

def loadData(conf):
    X = np.load(conf.fileX)
    y = np.load(conf.fileY)

    if conf.loadSplit:
        split = pickle.load(open(conf.fileSplit, 'rb'))
        train = split['train']
        valid = split['valid']
        test = split['test']
    else:
        if conf.outDim == 1:
            #TODO: works only on binary datasets
            i0 = []
            i1 = []
            for i in range(len(y)):
                if y[i] == 0.:
                    i0.append(i)
                else:
                    i1.append(i)
            random.shuffle(i0)
            random.shuffle(i1)

            trainSplit0 = int(len(i0)/100*conf.trainSplit)
            trainSplit1 = int(len(i0)/100*conf.trainSplit)

            validSplit0 = trainSplit0 + int((len(i0) - trainSplit0)/100*conf.validSplit)
            validSplit1 = trainSplit1 + int((len(i1) - trainSplit1)/100*conf.validSplit)

            train = i0[:trainSplit0] + i1[:trainSplit1]
            valid = i0[trainSplit0:validSplit0] + i1[trainSplit1:validSplit1]
            test = i0[validSplit0:] + i1[validSplit1:]

            random.shuffle(train)
            random.shuffle(valid)
            random.shuffle(test)

        if conf.saveSplit:
            split = {
                "train":train,
                "valid":valid,
                "test":test,
            }

            pickle.dump(split, open(conf.fileSplit, 'wb'))

    return(X,y,train,valid,test)

def runTrain(conf, model, XTrain, yTrain, XValid, yValid):
    callbacks = []
    
    if not conf.earlyStopping is None:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=conf.earlyStopping))

    if not conf.tensorboard is None:
        callbacks.append(TensorBoard(log_dir=conf.tensorboard))

    model.fit(XTrain, yTrain, validation_data=(XValid, yValid), batch_size=conf.batchSize, epochs=conf.epochs, callbacks=callbacks)

def runTest(conf, model, XTest, yTest):
    loss, acc = model.evaluate(XTest, yTest)
    print("test_loss: {:0.4f} - test_acc: {:0.4f}".format(loss, acc))

def save(conf, model):
    model.save(conf.modelFile)

def load(conf):
    model = keras.models.load_model(conf.modelFile)
    return model

def main(conf):
    print("======= CREATE MODEL")
    model = makeModel(conf)
    print("======= LOAD DATA")
    X, y, train, valid, test = loadData(conf)
    print("======= TRAIN MODEL")
    runTrain(conf, model, X[train], y[train], X[valid], y[valid]) 
    print("======= TEST MODEL")
    runTest(conf, model, X[test], y[test])
    print("======= SAVE MODEL")
    save(conf, model)


if __name__== "__main__":
    print("======= CREATE MODEL")
    model = makeModel(config)
    print("======= LOAD DATA")
    X, y, train, valid, test = loadData(config)
    print("======= TRAIN MODEL")
    runTrain(config, model, X[train], y[train], X[valid], y[valid]) 
    print("======= TEST MODEL")
    runTest(config, model, X[test], y[test])
    print("======= SAVE MODEL")
    save(conf, model)


