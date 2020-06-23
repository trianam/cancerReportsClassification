#! /usr/bin/env python

import numpy as np
import sys
import datetime
import notifier
import statisticLogger
import parameterLogger
import metricsLogger
import importer
import svm
import crossValidator
import plotter
import logger

analysisType = 1 
numFolds = 10

now = datetime.datetime.now()
timestamp = now.strftime("%Y_%m_%d-%H_%M_%S")

outputDir = "outSvm"
fileIsto = "./data/ISTOLOGIE_corr.csv"
fileNeop = "./data/RTRT_NEOPLASI_corr.csv"
fileErr = "./"+outputDir+"/"+timestamp+"-errors.txt"
fileClassesSedeOrig = "./"+outputDir+"/"+timestamp+"-classesSedeOrig.txt"
fileClassesSedeFilt = "./"+outputDir+"/"+timestamp+"-classesSedeFilt.txt"
fileClassesMorfoOrig = "./"+outputDir+"/"+timestamp+"-classesMorfoOrig.txt"
fileClassesMorfoFilt = "./"+outputDir+"/"+timestamp+"-classesMorfoFilt.txt"
filePlotClassesSede = "./"+outputDir+"/"+timestamp+"-plotClassesSede.pdf"
filePlotClassesMorfo = "./"+outputDir+"/"+timestamp+"-plotClassesMorfo.pdf"
fileMetrics = "./"+outputDir+"/"+timestamp+"-metrics.txt"
fileOldMetrics = "./"+outputDir+"/"+timestamp+"-oldMetrics.txt"
#fileMetricsSede = "./"+outputDir+"/"+timestamp+"-metricsSede.txt"
#fileMetricsDetailSede = "./"+outputDir+"/"+timestamp+"-metricsDetailsSede.txt"
#fileMetricsMorfo = "./"+outputDir+"/"+timestamp+"-metricsMorfo.txt"
#fileMetricsDetailMorfo = "./"+outputDir+"/"+timestamp+"-metricsDetailsMorfo.txt"
fileTokensNotizieSede = "./"+outputDir+"/"+timestamp+"-tokensNotizieSede.txt"
fileTokensDiagnosiSede = "./"+outputDir+"/"+timestamp+"-tokensDiagnosiSede.txt"
fileTokensMacroscopiaSede = "./"+outputDir+"/"+timestamp+"-tokensMacroscopiaSede.txt"
fileTokensNotizieMorfo = "./"+outputDir+"/"+timestamp+"-tokensNotizieMorfo.txt"
fileTokensDiagnosiMorfo = "./"+outputDir+"/"+timestamp+"-tokensDiagnosiMorfo.txt"
fileTokensMacroscopiaMorfo = "./"+outputDir+"/"+timestamp+"-tokensMacroscopiaMorfo.txt"
filePrecRecallSede = "./"+outputDir+"/"+timestamp+"-precisionRecallSede.pdf"
filePrecRecallMorfo = "./"+outputDir+"/"+timestamp+"-precisionRecallMorfo.pdf"
fileRocSede = "./"+outputDir+"/"+timestamp+"-rocSede.pdf"
fileRocMorfo = "./"+outputDir+"/"+timestamp+"-rocMorfo.pdf"
fileLearningCurveSede = "./"+outputDir+"/"+timestamp+"-learningCurveSede.pdf"
fileLearningCurveMorfo = "./"+outputDir+"/"+timestamp+"-learningCurveMorfo.pdf"
fileKdeSede = "./"+outputDir+"/"+timestamp+"-kdeSede.pdf"
fileKdeMorfo = "./"+outputDir+"/"+timestamp+"-kdeMorfo.pdf"
fileFreq = "./"+outputDir+"/"+timestamp+"-freqFeatures.pdf"
fileLog = "./"+outputDir+"/"+timestamp+"-executionLog.txt"
fileStatistics = "./"+outputDir+"/"+timestamp+"-statistics.txt"
fileParameters = "./"+outputDir+"/"+timestamp+"-parameters.txt"


sys.stderr = open(fileErr, 'w')

randomState = np.random.RandomState(0)

statisticLogger = statisticLogger.StatisticLogger()
parameterLogger = parameterLogger.ParameterLogger()
oldMetricsLogger = metricsLogger.MetricsLogger()
metricsLogger = metricsLogger.MetricsLogger()
plotter = plotter.Plotter()
importer = importer.Importer(statisticLogger, parameterLogger, plotter)
svmSede = svm.Svm(randomState)
svmMorfo = svm.Svm(randomState)
crossValidatorSede = crossValidator.CrossValidator(randomState, numFolds)
crossValidatorMorfo = crossValidator.CrossValidator(randomState, numFolds)

parameterLogger.numFolds = numFolds
parameterLogger.printParameters(fileParameters)

with logger.Logger(fileLog) as log:
    log.write("Start execution "+timestamp)
    log.write("Importing csv")
    importer.importCsv(fileIsto, fileNeop)
    #importer.cutDataset(500)

    log.write("Filter classes")
    importer.filterClasses(fileClassesSedeOrig, fileClassesMorfoOrig, fileClassesSedeFilt, fileClassesMorfoFilt, filePlotClassesSede, filePlotClassesMorfo)

    log.write("Tokenize text fields")
    importer.tokenize(fileTokensNotizieSede, fileTokensDiagnosiSede, fileTokensMacroscopiaSede, fileTokensNotizieMorfo, fileTokensDiagnosiMorfo, fileTokensMacroscopiaMorfo)

    log.write("Binarize classes")
    importer.binarize()

    xSede, xMorfo, ySede, yMorfo, ySedeUnb, yMorfoUnb = importer.getXY()
    yAladarSede, yAladarMorfo = importer.getYAladar()
    labelUnbinarizerSede, labelUnbinarizerMorfo = importer.getLabelUnbinarizers()
    commonClassesSede, commonClassesMorfo = importer.getCommonClasses()
    commonClassesIndexesSede, commonClassesIndexesMorfo = importer.getCommonClassesIndexes()

    uncommonClassesSede, uncommonClassesMorfo = importer.getUncommonClasses()
    uncommonClassesIndexesSede, uncommonClassesIndexesMorfo = importer.getUncommonClassesIndexes()

    log.write("Calculate old metrics")
    oldCurves, oldMetrics = importer.getOldMetrics()

    if analysisType == 0:
        log.write("Set and split dataset")
        svmSede.setData(xSede, ySede)
        svmMorfo.setData(xMorfo, yMorfo)

        log.write("Train svm")
        svmSede.train()
        svmMorfo.train()

        log.write("Calculate precision recall curves")
        plotter.plotPrecisionRecallCurve(filePrecRecallSede, svmSede.calculatePrecisionRecallCurve(), oldCurves['sede'])
        plotter.plotPrecisionRecallCurve(filePrecRecallMorfo, svmMorfo.calculatePrecisionRecallCurve(), oldCurves['morfo'])

        log.write("Calculate ROC curves")
        plotter.plotRocCurve(fileRocSede, svmSede.calculateRocCurve(), oldCurves['sede'])
        plotter.plotRocCurve(fileRocMorfo, svmMorfo.calculateRocCurve(), oldCurves['morfo'])

        log.write("Calculate learning curves")
        svmSede.calculateLearningCurve(fileLearningCurveSede)
        svmMorfo.calculateLearningCurve(fileLearningCurveMorfo)

        #log.write("Calculate metrics")
        #svmSede.calculateMetrics(fileMetricsSede, fileMetricsDetailSede)
        #svmMorfo.calculateMetrics(fileMetricsMorfo, fileMetricsDetailMorfo)

    elif analysisType == 1:
        log.write("Cross validate")

        crossValidatorSede.setSvm(svmSede)
        crossValidatorMorfo.setSvm(svmMorfo)

        crossValidatorSede.setData(xSede, ySede, ySedeUnb, yAladarSede)
        crossValidatorMorfo.setData(xMorfo, yMorfo, yMorfoUnb, yAladarMorfo)

        crossValidatorSede.setLabelUnbinarizer(labelUnbinarizerSede)
        crossValidatorMorfo.setLabelUnbinarizer(labelUnbinarizerMorfo)
        
        crossValidatorSede.setCommonClasses(commonClassesSede, commonClassesIndexesSede)
        crossValidatorMorfo.setCommonClasses(commonClassesMorfo, commonClassesIndexesMorfo)

        crossValidatorSede.setUncommonClasses(uncommonClassesSede, uncommonClassesIndexesSede)
        crossValidatorMorfo.setUncommonClasses(uncommonClassesMorfo, uncommonClassesIndexesMorfo)

        curvesSede, metricsSede = crossValidatorSede.validate()
        curvesMorfo, metricsMorfo = crossValidatorMorfo.validate()

        log.write("Write statistics and metrics")
        statisticLogger.numMisclassifiedAladarSede = crossValidatorSede.getNumMisclassifiedAladar()
        statisticLogger.numMisclassifiedAladarMorfo = crossValidatorMorfo.getNumMisclassifiedAladar()
        statisticLogger.numMisclassifiedNewSede = crossValidatorSede.getNumMisclassifiedNew()
        statisticLogger.numMisclassifiedNewMorfo = crossValidatorMorfo.getNumMisclassifiedNew()
        statisticLogger.numMisclassifiedRespectAladarSede = crossValidatorSede.getNumMisclassifiedRespectAladar()
        statisticLogger.numMisclassifiedRespectAladarMorfo = crossValidatorMorfo.getNumMisclassifiedRespectAladar()
        statisticLogger.numMisclassifiedRespectNewSede = crossValidatorSede.getNumMisclassifiedRespectNew()
        statisticLogger.numMisclassifiedRespectNewMorfo = crossValidatorMorfo.getNumMisclassifiedRespectNew()

        statisticLogger.printStatistics(fileStatistics)

        metricsLogger.setMetrics(metricsSede, metricsMorfo)
        metricsLogger.printMetrics(fileMetrics)

        oldMetricsLogger.setMetrics(oldMetrics['sede'], oldMetrics['morfo'])
        oldMetricsLogger.printMetrics(fileOldMetrics)

        log.write("Plot precision recall curves")
        plotter.plotPrecisionRecallCurve(filePrecRecallSede, curvesSede['precisionRecall'], oldCurves['sede'])
        plotter.plotPrecisionRecallCurve(filePrecRecallMorfo, curvesMorfo['precisionRecall'], oldCurves['morfo'])

        log.write("Plot ROC curves")
        plotter.plotRocCurve(fileRocSede, curvesSede['roc'], oldCurves['sede'])
        plotter.plotRocCurve(fileRocMorfo, curvesMorfo['roc'], oldCurves['morfo'])


    notifier.sendMessage("Terminated "+timestamp+" execution (terminated in "+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+")")
    log.write("End")

