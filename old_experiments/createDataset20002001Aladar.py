#! /usr/bin/env python

from collections import Counter
import numpy as np
import pandas as pd
import csv
import sys

fileIsto = "./data/ISTOLOGIE_corr.csv"
fileNeop = "./data/RTRT_NEOPLASI_corr.csv"
csvfile2000 = "./aladar2000.csv"
csvfile2001 = "./aladar2001.csv"
csvfile2000aa = "./aladar2000aa.csv"
csvfile2001aa = "./aladar2001aa.csv"
csvfileAll = "./aladarAll.csv"
encoding = 'iso-8859-1'

dfIsto = pd.read_csv(fileIsto, header = 0, encoding=encoding, quoting=csv.QUOTE_NONNUMERIC, low_memory=False, usecols = ['ICDO1_T', 'ICDO1_M', 'ICDO3_T', 'ICDO3_M', 'ICD_IX', 'id_neopl', 'id_tot'], dtype={'id_tot':pd.np.float64, 'id_neopl':pd.np.float64, 'ICDO1_T':pd.np.str, 'ICDO1_M':pd.np.str, 'ICDO3_T':pd.np.str, 'ICDO3_M':pd.np.str, 'ICD_IX':pd.np.str})#, converters={'ICDO1_T':str, 'ICDO1_M':str})
dfNeop = pd.read_csv(fileNeop, header = 0, encoding=encoding, quoting=csv.QUOTE_NONNUMERIC, low_memory=False, usecols = ['anno_archivio', 'data_incidenza', 'id_tot', 'id_neopl', 'data_ricovero', 'data_intervento', 'data_altro_intervento', 'inserimento_data_neoplasi', 'aggiornamento_data', 'sede', 'morfologia', 'sede_icdo3', 'morfologia_icdo3', 'ICD_IX'], dtype={'anno_archivio':pd.np.str, 'data_incidenza':pd.np.str, 'id_tot':pd.np.float64, 'id_neopl':pd.np.float64, 'sede':pd.np.str, 'morfologia':pd.np.str, 'sede_icdo3':pd.np.str, 'morfologia_icdo3':pd.np.str, 'ICD_IX':pd.np.str})

#df = pd.merge(dfIsto, dfNeop, on='id_tot')
df = pd.merge(dfIsto, dfNeop, on='id_neopl')

#df.drop('id_neopl', 1)

sys.exit()

tab2000 = []
tab2001 = []
tab2000aa = []
tab2001aa = []
tabAll = []

noDataIncidenza = 0
noAnnoArchivio = 0

with open(csvfile2000, 'w', newline='') as f2000, open(csvfile2001, 'w', newline='') as f2001, open(csvfile2000aa, 'w', newline='') as f2000aa, open(csvfile2001aa, 'w', newline='') as f2001aa, open(csvfileAll, 'w', newline='') as fAll:
    w2000 = csv.writer(f2000, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    w2001 = csv.writer(f2001, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    w2000aa = csv.writer(f2000aa, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    w2001aa = csv.writer(f2001aa, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    wAll = csv.writer(fAll, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    w2000.writerow(["ICDO1sede1", "ICDO1sede2", "ICDO1morfo1", "ICDO1morfo2", "aladarICDO1sede1", "aladarICDO1sede2", "aladarICDO1morfo1", "aladarICDO1morfo2"])
    w2001.writerow(["ICDO1sede1", "ICDO1sede2", "ICDO1morfo1", "ICDO1morfo2", "aladarICDO1sede1", "aladarICDO1sede2", "aladarICDO1morfo1", "aladarICDO1morfo2"])
    w2000aa.writerow(["ICDO1sede1", "ICDO1sede2", "ICDO1morfo1", "ICDO1morfo2", "aladarICDO1sede1", "aladarICDO1sede2", "aladarICDO1morfo1", "aladarICDO1morfo2"])
    w2001aa.writerow(["ICDO1sede1", "ICDO1sede2", "ICDO1morfo1", "ICDO1morfo2", "aladarICDO1sede1", "aladarICDO1sede2", "aladarICDO1morfo1", "aladarICDO1morfo2"])
    wAll.writerow(["ICDO1sede1", "ICDO1sede2", "ICDO1morfo1", "ICDO1morfo2", "aladarICDO1sede1", "aladarICDO1sede2", "aladarICDO1morfo1", "aladarICDO1morfo2"])

    for i in range(len(df)):
        if i%1000 == 0:
            print("Processed {}/{}".format(i, len(df)))

        if str(df.iloc[i]['data_incidenza']) == "nan":
            noDataIncidenza += 1
        if str(df.iloc[i]['anno_archivio']) == "nan":
            noAnnoArchivio += 1

        if not str(df.iloc[i]['data_incidenza']) == "nan" and df.iloc[i]['data_incidenza'][:4] == '2000' and not str(df.loc[i]['sede']) == "nan" and not str(df.loc[i]['morfologia']) == "nan" and not str(df.loc[i]['ICDO1_T']) == "nan" and not str(df.loc[i]['ICDO1_M']) == "nan":
            tab2000.append([df.loc[i]['sede'][:3], df.loc[i]['sede'][3:], df.loc[i]['morfologia'][:4], df.loc[i]['morfologia'][4:], df.loc[i]['ICDO1_T'][:3], df.loc[i]['ICDO1_T'][3:], df.loc[i]['ICDO1_M'][:4], df.loc[i]['ICDO1_M'][4:]])
            w2000.writerow([df.loc[i]['sede'][:3], df.loc[i]['sede'][3:], df.loc[i]['morfologia'][:4], df.loc[i]['morfologia'][4:], df.loc[i]['ICDO1_T'][:3], df.loc[i]['ICDO1_T'][3:], df.loc[i]['ICDO1_M'][:4], df.loc[i]['ICDO1_M'][4:]])

        if not str(df.iloc[i]['data_incidenza']) == "nan" and df.iloc[i]['data_incidenza'][:4] == '2001' and not str(df.loc[i]['sede']) == "nan" and not str(df.loc[i]['morfologia']) == "nan" and not str(df.loc[i]['ICDO1_T']) == "nan" and not str(df.loc[i]['ICDO1_M']) == "nan":
            tab2001.append([df.loc[i]['sede'][:3], df.loc[i]['sede'][3:], df.loc[i]['morfologia'][:4], df.loc[i]['morfologia'][4:], df.loc[i]['ICDO1_T'][:3], df.loc[i]['ICDO1_T'][3:], df.loc[i]['ICDO1_M'][:4], df.loc[i]['ICDO1_M'][4:]])
            w2001.writerow([df.loc[i]['sede'][:3], df.loc[i]['sede'][3:], df.loc[i]['morfologia'][:4], df.loc[i]['morfologia'][4:], df.loc[i]['ICDO1_T'][:3], df.loc[i]['ICDO1_T'][3:], df.loc[i]['ICDO1_M'][:4], df.loc[i]['ICDO1_M'][4:]])

        if not str(df.iloc[i]['anno_archivio']) == "nan" and df.iloc[i]['anno_archivio'] == '00' and not str(df.loc[i]['sede']) == "nan" and not str(df.loc[i]['morfologia']) == "nan" and not str(df.loc[i]['ICDO1_T']) == "nan" and not str(df.loc[i]['ICDO1_M']) == "nan":
            tab2000aa.append([df.loc[i]['sede'][:3], df.loc[i]['sede'][3:], df.loc[i]['morfologia'][:4], df.loc[i]['morfologia'][4:], df.loc[i]['ICDO1_T'][:3], df.loc[i]['ICDO1_T'][3:], df.loc[i]['ICDO1_M'][:4], df.loc[i]['ICDO1_M'][4:]])
            w2000aa.writerow([df.loc[i]['sede'][:3], df.loc[i]['sede'][3:], df.loc[i]['morfologia'][:4], df.loc[i]['morfologia'][4:], df.loc[i]['ICDO1_T'][:3], df.loc[i]['ICDO1_T'][3:], df.loc[i]['ICDO1_M'][:4], df.loc[i]['ICDO1_M'][4:]])

        if not str(df.iloc[i]['anno_archivio']) == "nan" and df.iloc[i]['anno_archivio'] == '01' and not str(df.loc[i]['sede']) == "nan" and not str(df.loc[i]['morfologia']) == "nan" and not str(df.loc[i]['ICDO1_T']) == "nan" and not str(df.loc[i]['ICDO1_M']) == "nan":
            tab2001aa.append([df.loc[i]['sede'][:3], df.loc[i]['sede'][3:], df.loc[i]['morfologia'][:4], df.loc[i]['morfologia'][4:], df.loc[i]['ICDO1_T'][:3], df.loc[i]['ICDO1_T'][3:], df.loc[i]['ICDO1_M'][:4], df.loc[i]['ICDO1_M'][4:]])
            w2001aa.writerow([df.loc[i]['sede'][:3], df.loc[i]['sede'][3:], df.loc[i]['morfologia'][:4], df.loc[i]['morfologia'][4:], df.loc[i]['ICDO1_T'][:3], df.loc[i]['ICDO1_T'][3:], df.loc[i]['ICDO1_M'][:4], df.loc[i]['ICDO1_M'][4:]])

        if not str(df.loc[i]['sede']) == "nan" and not str(df.loc[i]['morfologia']) == "nan" and not str(df.loc[i]['ICDO1_T']) == "nan" and not str(df.loc[i]['ICDO1_M']) == "nan":
            tabAll.append([df.loc[i]['sede'][:3], df.loc[i]['sede'][3:], df.loc[i]['morfologia'][:4], df.loc[i]['morfologia'][4:], df.loc[i]['ICDO1_T'][:3], df.loc[i]['ICDO1_T'][3:], df.loc[i]['ICDO1_M'][:4], df.loc[i]['ICDO1_M'][4:]])
            wAll.writerow([df.loc[i]['sede'][:3], df.loc[i]['sede'][3:], df.loc[i]['morfologia'][:4], df.loc[i]['morfologia'][4:], df.loc[i]['ICDO1_T'][:3], df.loc[i]['ICDO1_T'][3:], df.loc[i]['ICDO1_M'][:4], df.loc[i]['ICDO1_M'][4:]])

tab2000 = np.array(tab2000)
tab2001 = np.array(tab2001)
tab2000aa = np.array(tab2000aa)
tab2001aa = np.array(tab2001aa)
tabAll = np.array(tabAll)


