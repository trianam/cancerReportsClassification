
class ParameterLogger():
    def __init__(self):
        self.numFolds = 0
        self.cutOff = 0
        self.maxDf = 0
        self.mostCommonClassesNum = 0
        self.leastCommonClassesNum = 0
        self.useStemmer = 0


    def printParameters(self, filename):
        formatter = [
            {'desc':"num folds", 'field': self.numFolds},
            {'desc':"cut off", 'field': self.cutOff},
            {'desc':"term max document frequency", 'field': self.numFolds},
            {'desc':"num most common classes", 'field': self.mostCommonClassesNum},
            {'desc':"num least common classes", 'field': self.leastCommonClassesNum},
            {'desc':"use stemmer", 'field': self.useStemmer},
        ]

        with open(filename, 'w') as handler:
            for form in formatter:
                handler.write("{}\t{}\n".format(form['field'], form['desc']))


