import sys
import numpy as np
import pickle
import pandas as pd
import csv
import sklearn as sl
import sklearn.metrics
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score, precision_recall_fscore_support, f1_score, accuracy_score, cohen_kappa_score

fileIsto = "./data/ISTOLOGIE_corr.csv"
fileNeop = "./data/RTRT_NEOPLASI_corr.csv"
fileOut = "./metricsOld.txt"
fileOutPickle = "./metricsOld.p"

print('Importing csv')
#encoding = 'iso-8859-1' or encoding = 'cp1252' or encoding = 'latin1'
dfIsto = pd.read_csv(fileIsto, header = 0, encoding='iso-8859-1', quoting=csv.QUOTE_NONNUMERIC, low_memory=False, usecols = ['ICDO3_T', 'ICDO3_M', 'id_neopl'], dtype={'id_neopl':pd.np.float64, 'ICDO3_T':pd.np.str, 'ICDO3_M':pd.np.str})#, converters={'ICDO1_T':str, 'ICDO1_M':str})
dfNeop = pd.read_csv(fileNeop, header = 0, usecols = ['id_neopl', 'sede_icdo3', 'morfologia_icdo3'], low_memory=False, dtype={'id_neopl':pd.np.float64, 'sede_icdo3':pd.np.str, 'morfologia_icdo3':pd.np.str})

print('Merging csv')
dfMerge = pd.merge(dfIsto, dfNeop, on='id_neopl')

dfMerge = dfMerge.dropna(subset = ['sede_icdo3', 'morfologia_icdo3'])

yt = {}
yt['sede1'] = dfMerge['sede_icdo3'].map(lambda s: s[0:3])
yt['sede12']  = dfMerge['sede_icdo3']
yt['morfo1']  = dfMerge['morfologia_icdo3'].map(lambda s: s[0:4])
yt['morfo2']  = dfMerge['morfologia_icdo3'].map(lambda s: s[4])

yal = {}
yal['sede1'] = dfMerge['ICDO3_T'].map(lambda s: "" if pd.isnull(s) else s[0:3])
yal['sede12'] = dfMerge['ICDO3_T'].map(lambda s: "" if pd.isnull(s) else s)
yal['morfo1'] = dfMerge['ICDO3_M'].map(lambda s: "" if pd.isnull(s) else  s[0:4])
yal['morfo2'] = dfMerge['ICDO3_M'].map(lambda s: "" if pd.isnull(s) else  s[4])

print('Calculating metrics')

out_file = open(fileOut, 'w')

metrics = {}
for t in ['sede1', 'sede12', 'morfo1', 'morfo2']:
    metrics[t] = {'MAPs':np.nan, 'MAPc':np.nan}
    acc = accuracy_score(yt[t], yal[t])
    metrics[t]['accuracy'] = acc
    out_file.write("\naccuracy score {}: ".format(t))
    out_file.write(str(acc))

    kappa = cohen_kappa_score(yt[t], yal[t])
    metrics[t]['kappa'] = kappa
    out_file.write("\nkappa score {}: ".format(t))
    out_file.write(str(kappa))

    metrics[t]['precision'] = {}
    metrics[t]['recall'] = {}
    metrics[t]['f1score'] = {}
    for avg in ['micro', 'macro', 'weighted']:
        prec,rec,f1,_ = precision_recall_fscore_support(yt[t], yal[t], average=avg)
        metrics[t]['precision'][avg] = prec
        metrics[t]['recall'][avg] = rec
        metrics[t]['f1score'][avg] = f1
        out_file.write("\nprecision score {} {} avg: ".format(t,avg))
        out_file.write(str(prec))
        out_file.write("\nrecall score {} {} avg: ".format(t,avg))
        out_file.write(str(rec))
        out_file.write("\nf1 score {} {} avg: ".format(t,avg))
        out_file.write(str(f1))

    out_file.write("\n")

out_file.close()
pickle.dump(metrics, open(fileOutPickle, 'wb'))



sys.exit()

print('Calculating metrics')

out_file = open(fileOut, 'w')
out_file.write(sl.metrics.classification_report(dfMerge.sede.map(str), dfMerge.ICDO1_T.map(str)))
out_file.write("\n")
out_file.write(sl.metrics.classification_report(dfMerge.morfologia.map(str), dfMerge.ICDO1_M.map(str)))

out_file.write("\nf1 score sede micro average: ")
out_file.write(str(sl.metrics.f1_score(dfMerge.sede.map(str), dfMerge.ICDO1_T.map(str), average='micro')))
out_file.write("\nf1 score sede macro average: ")
out_file.write(str(sl.metrics.f1_score(dfMerge.sede.map(str), dfMerge.ICDO1_T.map(str), average='macro')))
out_file.write("\nf1 score morfologia micro average: ")
out_file.write(str(sl.metrics.f1_score(dfMerge.morfologia.map(str), dfMerge.ICDO1_M.map(str), average='micro')))
out_file.write("\nf1 score morfologia macro average: ")
out_file.write(str(sl.metrics.f1_score(dfMerge.morfologia.map(str), dfMerge.ICDO1_M.map(str), average='macro')))
out_file.write("\n")
out_file.close()

print('End')

#metrics.f1_score(dfMerge.sede.map(str), dfMerge.ICDO1_T.map(str))
#metrics.precision_score(dfMerge.sede.map(str), dfMerge.ICDO1_T.map(str))
#metrics.accuracy_score(dfMerge.sede.map(str), dfMerge.ICDO1_T.map(str))
#metrics.f1_score(dfMerge.sede.map(str), dfMerge.ICDO1_T.map(str), average='micro')
#metrics.f1_score(dfMerge.sede.map(str), dfMerge.ICDO1_T.map(str), average='macro')
#metrics.f1_score(dfMerge.sede.map(str), dfMerge.ICDO1_M.map(str), average='micro')
#metrics.f1_score(dfMerge.sede.map(str), dfMerge.ICDO1_M.map(str), average='macro')

