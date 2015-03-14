#import RandomConnMat as cm
import matplotlib.pylab as plt
import numpy as np
from scipy import linalg as la
import math as m
import decimal

def balance(l):
	N = len(l)
	meanP, meanN = 0,0
	c1, c2 = 0,0
	for i in range(N):
		if l[i] > 0:
			meanP += l[i]
			c1+=1
		if l[i] < 0:
			meanN += l[i]
			c2+=1
	diff = abs(meanP)-abs(meanN)
	for i in range(N):
		if l[i] < 0:
			l[i] -= diff/c2
	return l

N = 480.
sigma = m.sqrt(1/decimal.Decimal(N))
mu = 0
q = 2
b = np.random.normal(mu,sigma,size=(N,N))
c = np.random.normal(mu,q*sigma,size=(N,N))
d = np.random.normal(mu,2*q*sigma,size=(N,N))

#f2,((ax1,ax2)) = plt.subplots(2,sharex=False, sharey=False)
eb = la.eigvals(b)
ec = la.eigvals(c)
ed = la.eigvals(d)

colors = ['c','g','b','y','r']

plt.subplot(4,1,1)
plt.title("Random Matrices with mean ="+str(mu)+" and variance starting from %.8s" %sigma**2)

plt.scatter(eb.real,eb.imag, color=colors[0])
plt.scatter(ec.real,ec.imag, color=colors[1])
plt.axis('equal')
plt.subplot(4,1,2)
plt.hist(eb.real)

plt.subplot(4,1,3)
for i in range(480):
	balance(b[i,:])
	balance(c[i,:])
#plt.subplot(3,1,3)
plt.scatter(eb.real,eb.imag, color=colors[0])
plt.scatter(ec.real,ec.imag, color=colors[1])
plt.axis('equal')
plt.subplot(4,1,4)
plt.hist(eb.real)
#ax2.title("stuf")
#ax2.pcolor(c)
plt.show()
