import numpy as np
import random
import sympy

def subsequences(ts, window):
    shape = (ts.size - window + 1, window)
    strides = ts.strides * 2
    return np.lib.stride_tricks.as_strided(ts, shape=shape, strides=strides)

def checkSample(x, window):
    return np.apply_along_axis(lambda s: np.array(list(map(lambda l:sympy.isprime(int(l)), s))).all(), 1, subsequences(x, window)).any()

def checkSampleV2(x, primesStr):
    xStr = ''.join([str(d) for d in x])
    return any([e in xStr for e in primesStr])

def primesInSample(x):
    return np.array(list(map(lambda l:sympy.isprime(int(l)), x)))

def primesInSampleV2(x, primesStr):
    xStr = ''.join([str(d) for d in x])
    p = np.zeros_like(x, dtype=np.bool)
    for prime in primesStr:
        f = xStr.find(prime)
        if f > -1:
            p[f:f+len(prime)] = True
    return p

def createDataset(numSamples, rangeMin, rangeMax, seqLen, subseqLen, difficult=True, clean=False):
    """
    Create balanced dataset of 'numSamples' sequences of 'seqLen' integers in range ['rangeMin', 'rangeMax').
    Samples are positives if they have a subsequence of at least length 'subseqLen' composed entirely
    of prime numbers.
    """

    positivesX = []
    negativesX = []
    print("Create negatives (and some positives)", flush=True)
    while len(negativesX) < numSamples/2:
        if len(negativesX)%1000 is 0:
            print("neg: {};  pos: {}          ".format(len(negativesX), len(positivesX)), end='\r', flush=True)
        x = np.random.randint(rangeMin, rangeMax, seqLen)
        if difficult:
            #put subseqLen (or less) random primes anyway
            for _ in range(subseqLen):
                x[np.random.randint(seqLen)] = sympy.randprime(rangeMin, rangeMax)

        if clean:
            for i in range(seqLen - subseqLen +1):
                if np.array(list(map(lambda l:sympy.isprime(int(l)), [x[j] for j in range(i, i+subseqLen)]))).all():
                    x[i] = np.random.randint(rangeMin, rangeMax)       
                    while sympy.isprime(int(x[i])):
                        x[i] = np.random.randint(rangeMin, rangeMax)       

            negativesX.append(x)

        else:
            if checkSample(x, subseqLen):
                if len(positivesX) < numSamples/2:
                    positivesX.append(x)
            else:
                negativesX.append(x)

    print("neg: {};  pos: {}          ".format(len(negativesX), len(positivesX)), flush=True)
    print("Complete positives (if necessary)", flush=True)

    while len(positivesX) < numSamples/2:
        if len(positivesX)%1000 is 0:
            print("neg: {};  pos: {}          ".format(len(negativesX), len(positivesX)), end='\r', flush=True)
        x = np.random.randint(rangeMin, rangeMax, seqLen)
        iStart = np.random.randint(seqLen - subseqLen + 1)
        for i in range(iStart, iStart+3):
            x[i] = sympy.randprime(rangeMin, rangeMax)

        positivesX.append(x)

    print("neg: {};  pos: {}          ".format(len(negativesX), len(positivesX)), flush=True)

    positivesY = np.ones((len(positivesX), 1), dtype=np.int32)
    negativesY = np.zeros((len(negativesX), 1), dtype=np.int32)

    X = np.array(negativesX + positivesX, dtype=np.int32)
    y = np.concatenate((negativesY, positivesY), axis=0)

    p = np.apply_along_axis(lambda x: primesInSample(x), 0, X)

    return (X, y, p)


def createDatasetV2(numSamples, rangeMin, rangeMax, seqLen, minPrimes, numPrimes, maxPrimes):
    """
    Create balanced dataset of 'numSamples' sequences of 'seqLen' integers in range ['rangeMin', 'rangeMax'),
    each sample has at least 'minPrimes' and maximum 'maxPrimes' prime numbers
    Samples are positives if they have at least 'numPrimes' prime numbers.
    """
    negativesX = np.zeros(shape=(int(numSamples/2), seqLen), dtype=np.int32)
    positivesX = np.zeros(shape=(int(numSamples/2), seqLen), dtype=np.int32)
    negativesY = np.zeros((len(negativesX), 1), dtype=np.int32)
    positivesY = np.ones((len(positivesX), 1), dtype=np.int32)

    
    print("Create negatives", flush=True)
    for s in range(int(numSamples/2)):
        if s%1000 is 0:
            print("neg: {}          ".format(s), end='\r', flush=True)

        for t in range(seqLen):
            v = np.random.randint(rangeMin, rangeMax)
            while sympy.isprime(v):
                v = np.random.randint(rangeMin, rangeMax)
            negativesX[s][t] = v

        for t in random.sample(range(seqLen), np.random.randint(minPrimes, numPrimes)):
            negativesX[s][t] = sympy.randprime(rangeMin, rangeMax)

    print("neg: {}          ".format(s+1), flush=True)
    print("Create positives", flush=True)
    for s in range(int(numSamples/2)):
        if s%1000 is 0:
            print("pos: {}          ".format(s), end='\r', flush=True)

        for t in range(seqLen):
            v = np.random.randint(rangeMin, rangeMax)
            while sympy.isprime(v):
                v = np.random.randint(rangeMin, rangeMax)
            positivesX[s][t] = v

        for t in random.sample(range(seqLen), np.random.randint(numPrimes, maxPrimes+1)):
            positivesX[s][t] = sympy.randprime(rangeMin, rangeMax)

    print("pos: {}          ".format(s+1), flush=True)
    
    X = np.concatenate((negativesX, positivesX), axis=0)
    y = np.concatenate((negativesY, positivesY), axis=0)
    p = np.apply_along_axis(lambda x: primesInSample(x), 0, X)

    return (X, y, p)

def createDatasetV3(numSamples, seqLen, minSubseqLen, maxSubseqLen=None, singlePrime=True):
    """
    Create balanced dataset of 'numSamples' sequences of 'seqLen' integers in range [0, 9].
    Samples are positives if they have one (and only one if 'singlePrime' is true) 
    subsequence of length in ['minSubseqLen','maxSubseqLen']
    whose digits compose a prime number
    """
    
    if maxSubseqLen is None:
        maxSubseqLen = seqLen

    print("Calculate primes", flush=True)
    primes = sympy.sieve.primerange(10**(minSubseqLen-1), 10**maxSubseqLen)
    primesStr = [str(p) for p in primes]

    positivesX = []
    negativesX = []
    posWherePrimes = []
    negWherePrimes = []

    print("Create negatives", flush=True)
    while len(negativesX) < numSamples/2:
        if len(negativesX)%1000 is 0:
            print("neg: {};  pos: {}          ".format(len(negativesX), len(positivesX)), end='\r', flush=True)
        x = np.random.randint(0, 10, seqLen)

        for i in range(minSubseqLen, seqLen+1):
            while checkSampleV2(x[max(0,i-maxSubseqLen) : i], primesStr):
                x[i-1] = np.random.randint(0,10)

        if checkSampleV2(x, primesStr):
            if len(positivesX) < numSamples/2:
                positivesX.append(x)
                posWherePrimes.append(primesInSampleV2(x, primesStr))
        else:
            negativesX.append(x)
            negWherePrimes.append(primesInSampleV2(x, primesStr))


    print("neg: {};  pos: {}          ".format(len(negativesX), len(positivesX)), flush=True)
    print("Create positives", flush=True)

    while len(positivesX) < numSamples/2:
        if len(positivesX)%1000 is 0:
            print("neg: {};  pos: {}          ".format(len(negativesX), len(positivesX)), end='\r', flush=True)
        x = np.random.randint(0, 10, seqLen)

        if singlePrime:
            for i in range(minSubseqLen, seqLen+1):
                while checkSampleV2(x[max(0,i-maxSubseqLen) : i], primesStr):
                    x[i-1] = np.random.randint(0,10)

            prime = primesStr[np.random.randint(len(primesStr))]
            i = np.random.randint(seqLen-len(prime))
            for j in range(len(prime)):
                x[i+j] = int(prime[j])

            #TODO:remove bordering primes?

        if checkSampleV2(x, primesStr):
            if len(positivesX) < numSamples/2:
                positivesX.append(x)
                posWherePrimes.append(primesInSampleV2(x, primesStr))

        #TODO:check if necessary to force positiveness

    print("neg: {};  pos: {}          ".format(len(negativesX), len(positivesX)), flush=True)

    positivesY = np.ones((len(positivesX), 1), dtype=np.int32)
    negativesY = np.zeros((len(negativesX), 1), dtype=np.int32)

    X = np.array(negativesX + positivesX, dtype=np.int32)
    y = np.concatenate((negativesY, positivesY), axis=0)
    p = np.array(negWherePrimes + posWherePrimes)

    return (X, y, p)


def dataset1():
    np.random.seed(42)
    X,y,p = createDataset(100000, 0, 100, 100, 3, True)
    np.savez_compressed("corpusPrimes/corpus1.npz", X=X, y=y, p=p)

def dataset2():
    np.random.seed(42)
    X,y,p = createDataset(100000, 0, 100, 1000, 3, False, True)
    np.savez_compressed("corpusPrimes/corpus2.npz", X=X, y=y, p=p)

def dataset3():
    np.random.seed(42)
    X,y,p = createDataset(100000, 0, 100, 1000, 10, False, True)
    np.savez_compressed("corpusPrimes/corpus3.npz", X=X, y=y, p=p)

def dataset4():
    np.random.seed(42)
    X,y,p = createDataset(100000, 0, 100, 1000, 5, False, True)
    np.savez_compressed("corpusPrimes/corpus4.npz", X=X, y=y, p=p)

def dataset5():
    np.random.seed(42)
    X,y,p = createDatasetV2(100000, 0, 100, 1000, 10, 15, 20)
    np.savez_compressed("corpusPrimes/corpus5.npz", X=X, y=y, p=p)

def dataset6():
    np.random.seed(42)
    X,y,p = createDatasetV2(100000, 0, 100, 100, 10, 15, 20)
    np.savez_compressed("corpusPrimes/corpus6.npz", X=X, y=y, p=p)

def dataset7():
    np.random.seed(42)
    X,y,p = createDatasetV2(100000, 0, 100, 100, 0, 50, 100)
    np.savez_compressed("corpusPrimes/corpus7.npz", X=X, y=y, p=p)

def dataset8():
    np.random.seed(42)
    X,y,p = createDatasetV2(100000, 0, 100, 1000, 95, 100, 105)
    np.savez_compressed("corpusPrimes/corpus8.npz", X=X, y=y, p=p)

def dataset58():
    np.random.seed(42)
    centers = [20,30,40,50,60,70,80,90]
    for c in centers:
        print("========== center {}".format(c))
        X,y,p = createDatasetV2(100000, 0, 100, 1000, c-5, c, c+5)
        np.savez_compressed("corpusPrimes/corpus58_{}.npz".format(c), X=X, y=y, p=p)

def dataset58p2():
    np.random.seed(42)
    centers = [110, 120, 130, 140, 150, 160, 170, 180]
    for c in centers:
        print("========== center {}".format(c))
        X,y,p = createDatasetV2(100000, 0, 100, 1000, c-5, c, c+5)
        np.savez_compressed("corpusPrimes/corpus58_{}.npz".format(c), X=X, y=y, p=p)

def dataset58p3():
    np.random.seed(42)
    centers = [190, 200, 210, 220, 230, 240, 250, 260]
    for c in centers:
        print("========== center {}".format(c))
        X,y,p = createDatasetV2(100000, 0, 100, 1000, c-5, c, c+5)
        np.savez_compressed("corpusPrimes/corpus58_{}.npz".format(c), X=X, y=y, p=p)

def dataset9():
    np.random.seed(42)
    X,y,p = createDatasetV3(100000, 100, 3, 3, singlePrime=True)
    np.savez_compressed("corpusPrimes/corpus9.npz", X=X, y=y, p=p)

def dataset10():
    np.random.seed(42)
    X,y,p = createDatasetV3(100000, 100, 3, 3, singlePrime=False)
    np.savez_compressed("corpusPrimes/corpus10.npz", X=X, y=y, p=p)

def dataset9l():
    np.random.seed(42)
    lengths = [200,300,400,500,600,700,800,900,1000]
    for l in lengths:
        print("========== length {}".format(l))
        X,y,p = createDatasetV3(100000, l, 3, 3, singlePrime=True)
        np.savez_compressed("corpusPrimes/corpus9l_{}.npz".format(l), X=X, y=y, p=p)

