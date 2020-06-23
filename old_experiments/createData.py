#! /usr/bin/env python

import numpy as np
import sys
import datetime
import notifier
import statisticLogger
import parameterLogger
import importer
import logger
import os
import pickle

now = datetime.datetime.now()
timestamp = now.strftime("%Y_%m_%d-%H_%M_%S")

saveDir = "./savedData/"
fileIsto = "./data/ISTOLOGIE_corr.csv"
fileNeop = "./data/RTRT_NEOPLASI_corr.csv"
fileErr = saveDir+"errors.txt"
fileLog = saveDir+"log.txt"
fileXSede = saveDir+"xSede.npy"
fileXMorfo = saveDir+"xMorfo.npy"
fileYSede = saveDir+"ySede.npy"
fileYMorfo = saveDir+"yMorfo.npy"
fileYUnbSede = saveDir+"yUnbSede.npy"
fileYUnbMorfo = saveDir+"yUnbMorfo.npy"
fileCommonClassesSede = saveDir+"commonClassesSede.npy"
fileCommonClassesMorfo = saveDir+"commonClassesMorfo.npy"
fileCommonClassesIndexesSede = saveDir+"commonClassesIndexesSede.npy"
fileCommonClassesIndexesMorfo = saveDir+"commonClassesIndexesMorfo.npy"
fileUncommonClassesSede = saveDir+"uncommonClassesSede.npy"
fileUncommonClassesMorfo = saveDir+"uncommonClassesMorfo.npy"
fileUncommonClassesIndexesSede = saveDir+"uncommonClassesIndexesSede.npy"
fileUncommonClassesIndexesMorfo = saveDir+"uncommonClassesIndexesMorfo.npy"
fileStatisticLogger = saveDir+"statisticLogger.bin"
fileParameterLogger = saveDir+"parameterLogger.bin"


sys.stderr = open(fileErr, 'w')

randomState = np.random.RandomState(0)

statisticLogger = statisticLogger.StatisticLogger()
parameterLogger = parameterLogger.ParameterLogger()
importer = importer.Importer(statisticLogger, parameterLogger, plotter)

with logger.Logger(fileLog) as log:
    log.write("Start execution "+timestamp)
    log.write("Importing csv")
    importer.importCsv(fileIsto, fileNeop)
    #importer.cutDataset(500)

    log.write("Filter classes")
    importer.filterClasses(fileClassesSedeOrig, fileClassesMorfoOrig, fileClassesSedeFilt, fileClassesMorfoFilt, filePlotClassesSede, filePlotClassesMorfo)

    log.write("Tokenize text fields")
    importer.tokenize(fileTokensSede, fileTokensMorfo)

    log.write("Binarize classes")
    importer.binarize()

    xSede, xMorfo = importer.getX()
    ySede, yMorfo = importer.getY()
    ySedeUnb, yMorfoUnb = importer.getYUnb()
    labelUnbinarizerSede, labelUnbinarizerMorfo = importer.getLabelUnbinarizers()
    commonClassesSede, commonClassesMorfo = importer.getCommonClasses()
    commonClassesIndexesSede, commonClassesIndexesMorfo = importer.getCommonClassesIndexes()

    uncommonClassesSede, uncommonClassesMorfo = importer.getUncommonClasses()
    uncommonClassesIndexesSede, uncommonClassesIndexesMorfo = importer.getUncommonClassesIndexes()

    log.write("Save objects")
    np.save(fileXSede, xSede)
    np.save(fileXMorfo, xMorfo)
    np.save(fileYSede, ySede)
    np.save(fileYMorfo, yMorfo)
    np.save(fileYUnbSede, ySedeUnb)
    np.save(fileYUnbMorfo, yMorfoUnb)
    np.save(fileCommonClassesSede, commonClassesSede)
    np.save(fileCommonClassesMorfo, commonClassesMorfo)
    np.save(fileCommonClassesIndexesSede, commonClassesIndexesSede)
    np.save(fileCommonClassesIndexesMorfo, commonClassesIndexesMorfo)
    np.save(fileUncommonClassesSede, uncommonClassesSede)
    np.save(fileUncommonClassesMorfo, uncommonClassesMorfo)
    np.save(fileUncommonClassesIndexesSede, uncommonClassesIndexesSede)
    np.save(fileUncommonClassesIndexesMorfo, uncommonClassesIndexesMorfo)
    pickle.dump(statisticLogger, open(fileStatisticLogger, "wb"))
    pickle.dump(parameterLogger, open(fileParameterLogger, "wb"))

    notifier.sendMessage("Terminated "+timestamp+" execution (terminated in "+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+")")
    log.write("End")

