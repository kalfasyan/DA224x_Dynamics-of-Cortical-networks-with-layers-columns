import random
import numpy as np
import itertools
import matplotlib.pylab as plt
from scipy import linalg as la
import time
from progressbar import *
from collections import Counter
import decimal
import math
output = open('matrixExport.txt', 'wb')
widgets = ['Working: ', Percentage(), ' ', Bar(marker='=',
			left='[',right=']'), ' ', ETA(), ' ', FileTransferSpeed()]


""" Checks if 2 neurons belong in the same hypercolumn """
def same_hypercolumn(q,w):
	for i in hypercolumns:
		if q in i and w in i:
			return True
	return False

""" Checks if 2 neurons belong in the same minicolumn """
def same_minicolumn(q,w):
	if same_hypercolumn(q,w):
		for mc in minicolumns:
			if q in mc and w in mc:
				return True
	return False

""" Checks if 2 neurons belong in the same layer """
def same_layer(q,w):
	if same_hypercolumn(q,w):
		if q in layers23 and w in layers23:
			return True
		elif q in layers4 and w in layers4:
			return True
		elif q in layers5 and w in layers5:
			return True
	return False

def next_hypercolumn(q,w):
	if same_hypercolumn(q,w):
		return False
	for i in range(len(split_hc)):
		for j in split_hc[i]:
			if j < len(split_hc):	
				if (q in split_hc[i] and w in split_hc[i+1]):
					return True
	return False
	
def prev_hypercolumn(q,w):
	if same_hypercolumn(q,w):
		return False
	for i in range(len(split_hc)):
		for j in split_hc[i]:
			if i >0:
				if (q in split_hc[i] and w in split_hc[i-1]):
					return True
	return False
		
def diff_hypercolumns(q,w):
	if next_hypercolumn(q,w):
		if (q in layers5 and w in layers4):
			return flip(0.20,q)
	elif prev_hypercolumn(q,w):
		if (q in layers5 and w in layers23):
			return flip(0.20,q)
	return 0

def both_exc(q,w):
	if same_layer(q,w):
		if (q in excitatory_nrns and w in excitatory_nrns):
			return True
	return False
	
def both_inh(q,w):
	if same_layer(q,w):
		if (q in inhibitory_nrns and w in inhibitory_nrns):
			return True
	return False

""" Returns 1 under probability 'p', else 0  (0<=p<=1)"""

def flipAdj(p,q):
	if q in excitatory_nrns:	
		return 1 if random.random() < p else 0
	elif q in inhibitory_nrns:
		return -1 if random.random() < p else 0

def flip(p,q):
	if q in excitatory_nrns:	
		return (np.random.normal(0,sigma)+.5) if random.random() < p else 0
	elif q in inhibitory_nrns:
		return (np.random.normal(0,sigma)-.5) if random.random() < p else 0

def check_zero(z):
	unique, counts = np.unique(z, return_counts=True)
	occurence = np.asarray((unique, counts)).T
	for i in range(len(z)):
		if np.sum(z) != 0:
			if len(occurence)==3 and occurence[0][1]>occurence[2][1]:
				if z[i] == -1:
					z[i] = 0
			elif len(occurence)==3 and occurence[2][1]>occurence[0][1]:
				if z[i] == 1:
					z[i] = 0
			elif len(occurence) < 3:
				if z[i] == -1:
					z[i] += 1
				if z[i] == 1:
					z[i] -= 1
		else:
			return z

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
			l[i] -= diff/(c2)
	return l

def balanceN(mat):
	N = len(mat)
	sumP,sumN = 0,0
	c,c2=0,0
	for i in range(N):
		for j in range(N):
			if mat[j][i] > 0:
				sumP += mat[j][i]
				c+=1
			elif mat[j][i] < 0:
				sumN += mat[j][i]
				c2+=1	
	diff = sumP + sumN
	for i in range(N):
		for j in range(N):
			if mat[j][i] < 0:
				mat[j][i] -= diff/c2


#########################################################
"""              1. INITIALIZATIONS                """
exc_nrns_mc = 1
inh_nrns_mc = 1
lr_mc = 3
mc_hc = 4
hc = 2
nrns = (exc_nrns_mc+inh_nrns_mc)*hc*mc_hc*lr_mc
pbar = ProgressBar(widgets=widgets, maxval=nrns)
q = 1
sigma = math.sqrt(q/decimal.Decimal(nrns))
sigma2 = math.sqrt(1/decimal.Decimal(nrns))
mu = 0
nrns_hc = nrns/hc
nrns_mc = nrns_hc/mc_hc
nrns_l23 = nrns_mc/3
nrns_l4 = nrns_l23
nrns_l5 = nrns_l23
print nrns,"neurons."
print nrns_hc, "per hypercolumn in %s" %hc,"hypercolumns." 
print nrns_mc, "per minicolumn in %s" %mc_hc,"minicolumns."
print nrns_l23, "in each layer in %s" %lr_mc,"layers"
##############################################################
""" 2. Creating list of Hypercolumns, list of minicolumns within
	hypercolumns, list of layers within minicolumns within 
	hypercolumns"""
split = [i for i in range(nrns)]
split_hc = zip(*[iter(split)]*nrns_hc)
split_mc = []
split_lr = []
for i in range(len(split_hc)):
	split_mc.append(zip(*[iter(split_hc[i])]*nrns_mc))
	for j in range(len(split_mc[i])):
		split_lr.append(zip(*[iter(split_mc[i][j])]*nrns_l23))
split_exc = []
split_inh = []
for i in range(len(split_lr)):
	for j in split_lr[i]:
		split_exc.append(j[0:exc_nrns_mc])
		split_inh.append(j[exc_nrns_mc:])
##############################################################
""" 3. Creating sets for all minicolumns and all layers  """
hypercolumns = set(split_hc)

minitemp = []
for i in range(len(split_mc)):
	for j in split_mc[i]:
		minitemp.append(j)
minicolumns = set(minitemp)

l23temp = []
l4temp = []
l5temp = []
for i in range(len(split_lr)):
	for j in range(len(split_lr[i])):
		if j == 0:
			l23temp.append(split_lr[i][j])
		if j == 1:
			l4temp.append(split_lr[i][j])
		if j == 2:
			l5temp.append(split_lr[i][j])
layers23 = set(list(itertools.chain.from_iterable(l23temp)))
layers4 = set(list(itertools.chain.from_iterable(l4temp)))
layers5 = set(list(itertools.chain.from_iterable(l5temp)))

excitatory_nrns = set(list(itertools.chain.from_iterable(split_exc)))
inhibitory_nrns = set(list(itertools.chain.from_iterable(split_inh)))

"""            4. Connection matrix operations             """
##############################################################
start_time = time.time()
print "Initializing and creating connection matrix..."
conn_matrix = np.zeros((nrns,nrns))
pbar.start()
for i in range(nrns):
	for j in range(nrns):
		#conn_matrix[j][i] = flip(0.35,i)
		#"""	
		# SAME HYPERCOLUMN 
		if same_hypercolumn(i,j):
			#if same_minicolumn(i,j) or same_layer(i,j):
		#conn_matrix[j][i] = flip(0.2,i)

			# LAYER 2/3
			if i in layers23 and j in layers23:
				if both_exc(i,j):
					conn_matrix[j][i]= 0.26
				elif both_inh(i,j):
					conn_matrix[j][i]= 0.25
				elif i in excitatory_nrns and j in inhibitory_nrns:
					conn_matrix[j][i]= 0.21
				else:
					conn_matrix[j][i]= 0.16
			# LAYER 4
			elif i in layers4 and j in layers4:	
				if both_exc(i,j):
					conn_matrix[j][i]= 0.17
				elif both_inh(i,j):
					conn_matrix[j][i]= 0.50
				elif i in excitatory_nrns and j in inhibitory_nrns:
					conn_matrix[j][i]= 0.19
				else:
					conn_matrix[j][i]=  0.10 
			# LAYER 5
			elif i in layers5 and j in layers5:
				if both_exc(i,j):
					conn_matrix[j][i]=  0.09 
				elif both_inh(i,j):
					conn_matrix[j][i]=  0.60 
				elif i in excitatory_nrns and j in inhibitory_nrns:
					conn_matrix[j][i]=  0.10 
				else:
					conn_matrix[j][i]=  0.12 
			# FROM LAYER4 -> LAYER2/3
			elif i in layers4 and j in layers23:
				if both_exc(i,j):
					conn_matrix[j][i]=  0.28 
				elif both_inh(i,j):
					conn_matrix[j][i]=  0.20 
				elif i in excitatory_nrns and j in inhibitory_nrns:
					conn_matrix[j][i]=  0.10 
				else:
					conn_matrix[j][i]=  0.50 
			# FROM LAYER2/3 -> LAYER5
			elif i in layers23 and j in layers5:
				if both_exc(i,j):
					conn_matrix[j][i]=  0.55 
				elif both_inh(i,j):
					conn_matrix[j][i]=  0.0001 
				elif i in excitatory_nrns and j in inhibitory_nrns:
					conn_matrix[j][i]=  0.001 
				else:
					conn_matrix[j][i]=  0.20 	
							
		# DIFFERENT HYPERCOLUMN	
		elif next_hypercolumn(i,j):
			if (i in layers5 and j in layers4):
				conn_matrix[j][i]=  0.55
		elif prev_hypercolumn(i,j):
			if (i in layers5 and j in layers23):
				conn_matrix[j][i]=  0.55
		else:
			conn_matrix[j][i] = 0.00001
#"""
		pbar.update(i)
pbar.finish()
#"""
plt.pcolor(conn_matrix)
plt.show()
#"""
print ("Matrix Created in %.5s seconds." % (time.time() - start_time))
np.savetxt('matrixExport.txt', conn_matrix, fmt='%.4s')
print "\nWrote to matrixExport.txt"
