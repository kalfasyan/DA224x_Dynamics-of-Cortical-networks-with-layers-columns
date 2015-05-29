import random
import numpy as np
import itertools
import decimal
import math

nrn_type = "iaf_neuron"

exc_nrns_mc = 64
inh_nrns_mc = 16
lr_mc = 3
mc_hc = 4
hc = 3
nrns = (exc_nrns_mc+inh_nrns_mc)*hc*mc_hc*lr_mc
q = 1
sigma = math.sqrt(q/decimal.Decimal(nrns))
sigma2 = math.sqrt(1/decimal.Decimal(nrns))
mu = 0
nrns_hc = nrns/hc
nrns_mc = nrns_hc/mc_hc
nrns_l23 = nrns_mc*34/100
nrns_l4 = nrns_mc*33/100
nrns_l5 = nrns_mc*33/100
print nrns,"neurons."
print nrns_hc, "per hypercolumn in %s" %hc,"hypercolumns."
print nrns_mc, "per minicolumn in %s" %mc_hc,"minicolumns."
print nrns_l23, nrns_l4, nrns_l5, "in layers23 layer4 and layer5 respectively"
##############################################################
""" 2. Creating list of Hypercolumns, list of minicolumns within
    hypercolumns, list of layers within minicolumns within
    hypercolumns"""
split = [i for i in range(nrns)]
split_hc = zip(*[iter(split)]*nrns_hc)
split_mc = []
split_lr23,split_lr4,split_lr5 = [],[],[]
for i in range(len(split_hc)):
    split_mc.append(zip(*[iter(split_hc[i])]*nrns_mc))
    for j in range(len(split_mc[i])):
        split_lr23.append(split_mc[i][j][0:nrns_l23])
        split_lr4.append(split_mc[i][j][nrns_l23:nrns_l23+nrns_l4])
        split_lr5.append(split_mc[i][j][nrns_l23+nrns_l4:])

split_exc,split_inh = [],[]
for i in range(len(split_lr23)):
    split_exc.append(split_lr23[i][0:int(round(80./100.*(len(split_lr23[i]))))])
    split_inh.append(split_lr23[i][int(round(80./100.*(len(split_lr23[i])))):])
for i in range(len(split_lr4)):
    split_exc.append(split_lr4[i][0:int(round(80./100.*(len(split_lr4[i]))))])
    split_inh.append(split_lr4[i][int(round(80./100.*(len(split_lr4[i])))):])
for i in range(len(split_lr5)):
    split_exc.append(split_lr5[i][0:int(round(80./100.*(len(split_lr5[i]))))])
    split_inh.append(split_lr5[i][int(round(80./100.*(len(split_lr5[i])))):])

##############################################################
""" 3. Creating sets for all minicolumns and all layers  """
hypercolumns = set(split_hc)

minitemp = []
for i in range(len(split_mc)):
    for j in split_mc[i]:
        minitemp.append(j)
minicolumns = set(minitemp)

layers23 = set(list(itertools.chain.from_iterable(split_lr23)))
layers4 = set(list(itertools.chain.from_iterable(split_lr4)))
layers5 = set(list(itertools.chain.from_iterable(split_lr5)))

exc_nrns_set = set(list(itertools.chain.from_iterable(split_exc)))
inh_nrns_set = set(list(itertools.chain.from_iterable(split_inh)))

exc = [None for i in range(len(exc_nrns_set))]
inh = [None for i in range(len(inh_nrns_set))]

#################### FUNCTIONS #####################################



""" Checks if 2 neurons belong in the same hypercolumn """
def same_hypercolumn(q,w):
	for i in hypercolumns:
		if q in i and w in i:
			return True
	return False

""" Checks if 2 neurons belong in the same minicolumn """
def same_minicolumn(q,w):
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
		if (q in exc_nrns_set and w in exc_nrns_set):
			return True
	return False

def both_inh(q,w):
	if same_layer(q,w):
		if (q in inh_nrns_set and w in inh_nrns_set):
			return True
	return False

""" Returns 1 under probability 'p', else 0  (0<=p<=1)"""
def flipAdj(p,q):
	if q in exc_nrns_set:
		return 1 if random.random() < p else 0
	elif q in inh_nrns_set:
		return -1 if random.random() < p else 0

def flip(p,q):
    #p+=.2
    r=0# np.random.uniform(0,sigma)
    if q in exc_nrns_set:
        return (np.random.normal(0,sigma)+.5) if random.random() < p-r else 0
    elif q in inh_nrns_set:
        return (np.random.normal(0,sigma)-.5) if random.random() < p-r else 0

def flip2(p,q):
    a = decimal.Decimal(0.002083333)
    if q in exc_nrns_set:
        return (abs(np.random.normal(0,a))) if random.random() < p else 0
    elif q in inh_nrns_set:
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

""" Total sum of conn_matrix weights becomes zero """
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

""" Returns a counter 'c' in case a number 'n' is not (close to) zero """
def check_count(c, n):
    if n <= -1e-4 or n>= 1e-4:
        c+=1
    return c

