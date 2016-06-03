import numpy as np

# get the next binary vector
def inc(x):
    for i in range(len(x)):
        x[i]+=1
        if x[i]<=1: return True
        x[i]=0

    return False
s = [0, 0, 0]
inc(s)
s
inc([0, 0, 1])
#compute the energy for a single x,h pair
def lh_one(x,h):
 return -np.dot(np.dot(x,W),h)-np.dot(b,x)-np.dot(d,h)

#input is a list of 1d arrays, X
def lh(X):

    x=np.zeros(K)
    h=np.zeros(K)

    logZ=-np.inf

    #compute the normalizing constant
    while True:
        logZ=np.logaddexp(logZ, lh_one(x))
        if not inc(x): break

 #compute the log-likelihood
 lh=0
 for x in X: # iterate over elements in the dataset
  lhp=-np.inf
  while True: #sum over all possible values of h
   lhp=np.logaddexp(lhp,lh_one(x,h))
   if not inc(h): break
  lh+=lhp-logZ

 return lh