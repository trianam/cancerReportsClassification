import time

class Logger:
    def __init__(self, fileName):
        self._logFile = open(fileName, 'w')
        self._time0 = time.time()


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._logFile.close()

    def write(self, description):
        strToPrint = "{0:.1f} sec - {1:s}\n".format(time.time()-self._time0, description)
        self._logFile.write(strToPrint)
        print(strToPrint, end='')

