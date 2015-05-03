from scipy import stats
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
import params011 as pm
output = open('matrixExport.txt', 'wb')

# Progress bar stuff
# --------------------------------------------------------
widgets = ['Working: ', Percentage(), ' ', Bar(marker='=',
            left='[',right=']'), ' ', ETA(), ' ', FileTransferSpeed()]
pbar = ProgressBar(widgets=widgets, maxval=pm.nrns)
#---------------------------------------------------------

start_time = time.time()
print "Initializing and creating connection matrix..."
conn_matrix = np.zeros((pm.nrns,pm.nrns))
#pbar.start()
count23,count4,count5,countA,countB,countAz,countQ,countW = 0,0,0,0,0,0,0,0
for i in range(pm.nrns):
    for j in range(pm.nrns):
        #http://stackoverflow.com/questions/481144/equation-for-testing-if-a-point-is-inside-a-circle
        # SAME HYPERCOLUMN
        if pm.same_hypercolumn(i,j):
        #"""
            # SAME MINICOLUMN
            if pm.same_minicolumn(i,j):
                conn_matrix[j][i] = pm.flip(0.028,i)
                countA = pm.check_count(countA, conn_matrix[j][i])
            #:::
            elif pm.same_hypercolumn(i,j) and not pm.same_minicolumn(i,j):
                conn_matrix[j][i] = pm.flip(0.11,i)
                countQ = pm.check_count(countQ,conn_matrix[j][i])
        # DIFFERENT HYPERCOLUMN
        elif not pm.same_hypercolumn(i,j):
            conn_matrix[j][i] = pm.flip(0.11,i)
            countAz = pm.check_count(countAz, conn_matrix[j][i])
        #pbar.update(i)
#pbar.finish()

        #pbar.update(i)
#pbar.finish()

print ("Matrix Created in %.5s seconds." % (time.time() - start_time))

#_________________________________________________________________________________________

#"""
print "connections 2/3 = ", count23
print "connections 4 = ", count4
print "connections 5 = ", count5
print "countA", countA
print "countB", countB
print "athroisma: ", count23+count4+count5+countA+countB
print "not same minicolumn, same hypercolumn = ",countQ
print "not same hypercolumn = ",countAz
print "connections 30%", countA+countQ
print "connections 70%", countAz
#"""


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

k=[]
for i in range(pm.nrns):
    if np.sum(conn_matrix[i,:]) > 1e-5:
        k.append(i)
print "Row sums not zero", len(k)
pm.balanceN(conn_matrix)
#for i in range(len(k)):
#    pm.balance(conn_matrix[k[i],:])
delta =0
for i in range(pm.nrns):
    if np.sum(conn_matrix[i,:]) > 1e-5:
        delta+=1
        #print np.sum(conn_matrix[i,:])
        #print i
print "sum of all matrix",np.sum(conn_matrix)
print "Row sums not to zero after balance",delta

g = 0
for i in range(pm.nrns):
    for j in range(pm.nrns):
        if conn_matrix[j][i] < -1e-4 or conn_matrix[j][i] > 1e-4:
            g+=1
print "total connections: ", g

print "% connections within a minicolumn= ", round((countA+countQ)/float(g)*100.)
print "% connections outside a minicolumn= ", round(float(float(countAz)/float(g))*100.)
print
#"""

h,z=0,0
for i in range(pm.nrns):
    if i in pm.exc_nrns_set:
        for j in conn_matrix[:,i]:
            if j < 0:
                h+=1
    if i in pm.inh_nrns_set:
        for j in conn_matrix[:,i]:
            if j > 0:
                z+=1
print h,"negatives in exc"
print z,"positives in inh"

"""
gh = 0
for i in conn_matrix[1][:]:
    if i < -1e-4 or i > 1e-4:
        gh+=1
print gh
"""
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#"""
ee = la.eigvals(conn_matrix)
conn_matrix.dump("011.dat")
print "done"


"""
xl = np.linspace(-30,30,61)
yl = np.linspace(-30,30,61)

con_density = np.zeros((len(xl),len(yl)))
st = []

for i in range(len(trials)):
    ee = trials[i]
    for ii in range(len(ee)):
        xid = plt.find(xl>=ee[ii].imag)[0]
        if len(plt.find(yl>=ee[ii].real)) > 0:
            yid = plt.find(yl>=ee[ii].real)[0]
        else:
            yid = abs(trials[i][0])
        con_density[xid,yid] = con_density[xid,yid]+1
    st.append(con_density)
for i in st:
    pcolor(i)
colorbar()

ed_ev = np.linspace(-20,20,1e2)
hh_real,ed1 = np.histogram(ee.real,ed_ev)
hh_imag,ed1 = np.histogram(ee.imag,ed_ev)

tt = np.linspace(np.pi,-np.pi,1e2)
sx = np.sin(tt)
sy = np.cos(tt)


#plt.figure(1)
#plt.clf()
#plt.subplot(3,2,1)
plt.scatter(ee.real,ee.imag)
plt.plot(sx,sy,'r')
plt.plot(20*sx,20*sy,'g')
#plt.pcolor(conn_matrix, cmap=plt.cm.Blues)
plt.title("%.8s variance," % pm.sigma**2 +str(pm.mu)+" mean")
plt.axis('equal')
plt.xlim(min(ed_ev),max(ed_ev))
plt.ylim(min(ed_ev),max(ed_ev))
#plt.xlabel("two largest eigenvalues by magnitude %s, " % abs(ee)[0]+ str(abs(ee)[1]))
plt.show()



plt.subplot(3,2,2)
plt.plot(hh_imag,ed_ev[0:-1])
plt.ylim(min(ed_ev),max(ed_ev))
#plt.ylim(0,100)
plt.xlabel("max ee.real %.5s" % np.max(ee.real) + " max ee.imag %.5s" %np.max(ee.imag))

plt.subplot(3,2,3)
plt.plot(ed_ev[0:-1],hh_real)
plt.xlim(min(ed_ev),max(ed_ev))


plt.subplot(3,2,4)
#plt.plot(noB_var_row)#, cmap=plt.cm.RdYlBu)
#plt.plot(noB_var_col)#, cmap=plt.cm.RdYlBu)
#plt.plot(B_var_row)#, cmap=plt.cm.RdYlBu)
plt.plot(B_var_col)#, cmap=plt.cm.RdYlBu)


plt.subplot(3,2,5)
plt.pcolor(conn_matrix)#, cmap=plt.cm.RdYlBu)

plt.subplot(3,2,6)
plt.plot(ed[0:-1],hh)
#plt.ylim(0,800)

plt.show()
#"""
#np.savetxt('matrixExport.txt', conn_matrix, fmt='%.1s')
#print "\nWrote to matrixExport.txt"

"""
cmaps(['indexed','Blues','OrRd','PiYG','PuOr',
                'RdYlBu','RdYlGn','afmhot','binary','copper',
                'gist_ncar','gist_rainbow','own1','own2'])
"""
