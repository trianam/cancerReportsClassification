
class StatisticLogger():
    def __init__(self):
        self.numDocuments = 0
        self.numDocumentsDelSede = 0
        self.numDocumentsDelMorfo = 0
        self.numTermsNotizieSede = 0
        self.numTermsDiagnosiSede = 0
        self.numTermsMacroscopiaSede = 0
        self.numTermsNotizieMorfo = 0
        self.numTermsDiagnosiMorfo = 0
        self.numTermsMacroscopiaMorfo = 0
        self.numClassesSede = 0
        self.numClassesMorfo = 0
        self.numClassesDelSede = 0
        self.numClassesDelMorfo = 0
        self.numMisclassifiedAladarSede = 0
        self.numMisclassifiedAladarMorfo = 0
        self.numMisclassifiedNewSede = 0
        self.numMisclassifiedNewMorfo = 0
        self.numMisclassifiedRespectAladarSede = 0
        self.numMisclassifiedRespectAladarMorfo = 0
        self.numMisclassifiedRespectNewSede = 0
        self.numMisclassifiedRespectNewMorfo = 0

    def printStatistics(self, filename):
        formatter = [
            {'desc':"num documents", 'field': self.numDocuments},
            {'desc':"num documents deleted sede", 'field':self.numDocumentsDelSede},
            {'desc':"num documents deleted morfo", 'field':self.numDocumentsDelMorfo},
            {'desc':"num terms notizie sede", 'field':self.numTermsNotizieSede},
            {'desc':"num terms diagnosi sede", 'field':self.numTermsDiagnosiSede},
            {'desc':"num terms macroscopia sede", 'field':self.numTermsMacroscopiaSede},
            {'desc':"num terms notizie morfo", 'field':self.numTermsNotizieMorfo},
            {'desc':"num terms diagnosi morfo", 'field':self.numTermsDiagnosiMorfo},
            {'desc':"num terms macroscopia morfo", 'field':self.numTermsMacroscopiaMorfo},
            {'desc':"num classes sede", 'field':self.numClassesSede},
            {'desc':"num classes morfo", 'field':self.numClassesMorfo},
            {'desc':"num classes deleted sede", 'field':self.numClassesDelSede},
            {'desc':"num classes deleted morfo", 'field':self.numClassesDelMorfo},
            {'desc':"num misclassified Aladar sede", 'field':self.numMisclassifiedAladarSede},
            {'desc':"num misclassified Aladar morfo", 'field':self.numMisclassifiedAladarMorfo},
            {'desc':"num misclassified New sede", 'field':self.numMisclassifiedNewSede},
            {'desc':"num misclassified New morfo", 'field':self.numMisclassifiedNewMorfo},
            {'desc':"num misclassified respect Aladar sede", 'field':self.numMisclassifiedRespectAladarSede},
            {'desc':"num misclassified respect Aladar morfo", 'field':self.numMisclassifiedRespectAladarMorfo},
            {'desc':"num misclassified respect New sede", 'field':self.numMisclassifiedRespectNewSede},
            {'desc':"num misclassified respect New morfo", 'field':self.numMisclassifiedRespectNewMorfo}
        ]

        with open(filename, 'w') as handler:
            for form in formatter:
                handler.write("{}\t{}\n".format(form['field'], form['desc']))


