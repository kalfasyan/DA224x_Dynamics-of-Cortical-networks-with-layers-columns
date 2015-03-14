#
import random
import numpy as np
import itertools
import matplotlib.pylab as plt
from scipy import linalg as la
import time
from progressbar import *
from collections import Counter
import decimal
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

def flip2(p,q):
	if q in excitatory_nrns:	
		return 1 if random.random() < p else 0
	elif q in inhibitory_nrns:
		return -1 if random.random() < p else 0

def flip(p,q):
	a = decimal.Decimal(0.002083333)
	if q in excitatory_nrns:	
		return (abs(np.random.normal(0,a))) if random.random() < p else 0
	elif q in inhibitory_nrns:
		return (-abs(np.random.normal(0,a))) if random.random() < p else 0


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

	
#########################################################
"""              1. INITIALIZATIONS                """
exc_nrns_mc = 10
inh_nrns_mc = 10
lr_mc = 3
mc_hc = 4
hc = 2
nrns = (exc_nrns_mc+inh_nrns_mc)*hc*mc_hc*lr_mc
#bar = Bar('Processing', max=nrns)
pbar = ProgressBar(widgets=widgets, maxval=nrns)
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
#conn_matrix = np.random.random_integers(0,0,size=(nrns,nrns))
conn_matrix = np.zeros((nrns,nrns))
print conn_matrix
pbar.start()
for i in range(nrns):
	for j in range(nrns):
		#conn_matrix[j][i] = flip(0.35,i)
		# SAME HYPERCOLUMN 
		if same_hypercolumn(i,j):
			# LAYER 2/3
			if i in layers23 and j in layers23:
				if both_exc(i,j):
					conn_matrix[j][i]= flip(0.26,i)
				elif both_inh(i,j):
					conn_matrix[j][i]= flip(0.25,i)
				elif i in excitatory_nrns and j in inhibitory_nrns:
					conn_matrix[j][i]= flip(0.21,i)
				else:
					conn_matrix[j][i]= flip(0.16,i)
			# LAYER 4
			elif i in layers4 and j in layers4:	
				if both_exc(i,j):
					conn_matrix[j][i]= flip(0.17,i)
				elif both_inh(i,j):
					conn_matrix[j][i]= flip(0.50,i)
				elif i in excitatory_nrns and j in inhibitory_nrns:
					conn_matrix[j][i]= flip(0.19,i)
				else:
					conn_matrix[j][i]= flip(0.10,i)
			# LAYER 5
			elif i in layers5 and j in layers5:
				if both_exc(i,j):
					conn_matrix[j][i]= flip(0.09,i)
				elif both_inh(i,j):
					conn_matrix[j][i]= flip(0.60,i)
				elif i in excitatory_nrns and j in inhibitory_nrns:
					conn_matrix[j][i]= flip(0.10,i)
				else:
					conn_matrix[j][i]= flip(0.12,i)
			# FROM LAYER4 -> LAYER2/3
			elif i in layers4 and j in layers23:
				if both_exc(i,j):
					conn_matrix[j][i]= flip(0.28,i)
				elif both_inh(i,j):
					conn_matrix[j][i]= flip(0.20,i)
				elif i in excitatory_nrns and j in inhibitory_nrns:
					conn_matrix[j][i]= flip(0.10,i)
				else:
					conn_matrix[j][i]= flip(0.50,i)
			# FROM LAYER2/3 -> LAYER5
			elif i in layers23 and j in layers5:
				if both_exc(i,j):
					conn_matrix[j][i]= flip(0.55,i)
				elif both_inh(i,j):
					conn_matrix[j][i]= flip(0.0001,i)
				elif i in excitatory_nrns and j in inhibitory_nrns:
					conn_matrix[j][i]= flip(0.001,i)
				else:
					conn_matrix[j][i]= flip(0.20,i)	
		# DIFFERENT HYPERCOLUMN	
		elif next_hypercolumn(i,j):
			if (i in layers5 and j in layers4):
				conn_matrix[j][i]=  flip(0.30,i)
		elif prev_hypercolumn(i,j):
			if (i in layers5 and j in layers23):
				conn_matrix[j][i]=  flip(0.30,i)
		else:
			conn_matrix[j][i] = flip(0.0001,i)
	pbar.update(i)
pbar.finish()

print ("Matrix Created in %.5s seconds." % (time.time() - start_time))
#print "Loading plot..."
ee = la.eigvals(conn_matrix)
plt.scatter(ee.real,ee.imag)
#plt.pcolor(conn_matrix)
plt.show()
#np.savetxt('matrixExport.txt', conn_matrix, fmt='%.1s')
#print "\nWrote to matrixExport.txt"
