"""
Support Vector Machine implemented using Stochastic Gradient Ascent
Jun Dan, 11/26/2014
"""

import sys
import numpy as np

###   read input file   ###
myData = np.genfromtxt(sys.argv[1], delimiter=',')
n = myData.shape[0] # sample size

###   build new X by adding 1 as the last column   ###
X = np.ones(shape=(n, 3))
X[:,(0,1)] = myData.loc[:,(0,1)]

###   build kernel matrix for hinge loss   ###
def kernel_matrix(x, typ):
    K_ = np.zeros(shape=(x.shape[0], x.shape[0]))
    if typ == 'linear':
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                K_[i,j] = np.dot(x[i,], x[j,])
    else:
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                K_[i,j] = np.dot(x[i,], x[j,])*np.dot(x[i,], x[j,])
    return K_

K = kernel_matrix(X, sys.argv[2])
print 'SVM using gradient ascent with %s kernel' % sys.argv[2]


###   calculate step size   ###
eta = 1/diag(K)

iter = 0

###   initial alphas   ###
alpha = np.zeros(n)

###   difference   ###
diff = 1
eps = 0.0001
C = 10

while (diff > eps):
    alpha0 = alpha.copy()
    for k in range(n):
        alpha[k] = alpha[k] + eta[k]*(1-myData.loc[k,2]*sum(alpha*myData.loc[:,2]*K[:,k]))
        if alpha[k] < 0:
            alpha[k] = 0
        if alpha[k] > C:
            alpha[k] = C
    iter = iter+1
    diff = sum((alpha-alpha0)*(alpha-alpha0))
    print (iter, diff)

print 'support vectors\n'
for k in range(n):
    if alpha[k] != 0:
        print 'sample %d : %.1f, %.1f; class : %d; a : %f\n' % (k+1, myData.loc[k,0], myData.loc[k,1], myData.loc[k,2], alpha[k])

print 'total number of support vectors : %d\n' % sum(alpha!=0)
