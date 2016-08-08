from toy import *
import numpy as np
from timeit import default_timer as timer

n = 3  # size of grid
lattice = [[Node() for x in range(n)] for y in range(n)]

coupl = .9
W = [[[coupl, coupl, coupl, coupl] for x in range(n)] for y in range(n)]
#W = [[[np.random.rand()/2 + 0.5, np.random.rand()/2 + 0.5, np.random.rand()/2 + 0.5, np.random.rand()/2 + 0.5] for x in range(n)] for y in range(n)]
#W = [[ np.random.rand((4)) * 2 for x in range(n) ] for y in range(n)]

getMag(lattice, n)

ww = [-x/10 for x in range(10)]
for ii in ww:
    start = timer()
    coupl = ii
    W = [[[coupl, coupl, coupl, coupl] for x in range(n)] for y in range(n)]
    # W = [[[np.random.rand()/2 + 0.5 + ii, np.random.rand()/2 + 0.5 + ii, np.random.rand()/2 + 0.5 + ii, np.random.rand()/2 + 0.5 + ii] for x in range(n)] for y in range(n)]
    # W = [[[(1 - -1) * np.random.rand() + -1,(1 - -1) * np.random.rand() + -1, (1 - -1) * np.random.rand() + -1, (1 - -1) * np.random.rand() + -1] for x in range(n)] for y in range(n)]

    for iter in range(10):
        updateMF(lattice, W, n)
    mf = partitionMF(lattice, W, n)
    mfbytap = partition(lattice, W, n, "tap2")
    magMF = getMag(lattice, n)
    reset(lattice, W, n)
    for iter in range(10):
        updateTAP2(lattice, W, n)
    tap = partitionTAP(lattice, W, n)
    magTAP = getMag(lattice, n)
    Z = magnetization(lattice, W, n)
    magReal = getMag(lattice, n)
    print(magMSE(magReal, magMF), magMSE(magReal, magTAP))
    print(-np.log(Z), mf, tap)


