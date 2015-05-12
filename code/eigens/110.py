from scipy import stats
import random
import numpy as np
import itertools
import matplotlib.pylab as plt
from scipy import linalg as la
import time
from collections import Counter
import decimal
import math
import params110 as pm

start_time = time.time()
print "Initializing and creating connection matrix..."
conn_matrix = np.zeros((pm.nrns,pm.nrns))
#pbar.start()
count23,count4,count5,countA,countB,countAz,countQ,countW = 0,0,0,0,0,0,0,0
for i in range(pm.nrns):
    for j in range(pm.nrns):
        if pm.same_minicolumn(i,j):
            if i in pm.layers23 and j in pm.layers23:
                if pm.both_exc(i,j):
                    conn_matrix[j][i]= pm.flip(0.26,i)
                    count23 = pm.check_count(count23, conn_matrix[j][i])
                elif pm.both_inh(i,j):
                    conn_matrix[j][i]= pm.flip(0.25,i)
                    count23 = pm.check_count(count23, conn_matrix[j][i])
                elif i in pm.exc_nrns_set and j in pm.inh_nrns_set:
                    conn_matrix[j][i]= pm.flip(0.21,i)
                    count23 = pm.check_count(count23, conn_matrix[j][i])
                else:
                    conn_matrix[j][i]= pm.flip(0.16,i)
                    count23 = pm.check_count(count23, conn_matrix[j][i])
            # LAYER 4
            elif i in pm.layers4 and j in pm.layers4:
                if pm.both_exc(i,j):
                    conn_matrix[j][i]= pm.flip(0.17,i)
                    count4 = pm.check_count(count4, conn_matrix[j][i])
                elif pm.both_inh(i,j):
                    conn_matrix[j][i]= pm.flip(0.50,i)
                    count4 = pm.check_count(count4, conn_matrix[j][i])
                elif i in pm.exc_nrns_set and j in pm.inh_nrns_set:
                    conn_matrix[j][i]= pm.flip(0.19,i)
                    count4 = pm.check_count(count4, conn_matrix[j][i])
                else:
                    conn_matrix[j][i]= pm.flip(0.10,i)
                    count4 = pm.check_count(count4, conn_matrix[j][i])
            # LAYER 5
            elif i in pm.layers5 and j in pm.layers5:
                if pm.both_exc(i,j):
                    conn_matrix[j][i]= pm.flip(0.09,i)
                    count5 = pm.check_count(count5, conn_matrix[j][i])
                elif pm.both_inh(i,j):
                    conn_matrix[j][i]= pm.flip(0.60,i)
                    count5 = pm.check_count(count5, conn_matrix[j][i])
                elif i in pm.exc_nrns_set and j in pm.inh_nrns_set:
                    conn_matrix[j][i]= pm.flip(0.10,i)
                    count5 = pm.check_count(count5, conn_matrix[j][i])
                else:
                    conn_matrix[j][i]= pm.flip(0.12,i)
                    count5 = pm.check_count(count5, conn_matrix[j][i])
            # FROM LAYER4 -> LAYER2/3
            elif i in pm.layers4 and j in pm.layers23:
                if pm.both_exc(i,j):
                    conn_matrix[j][i]= pm.flip(0.28,i)
                    countA = pm.check_count(countA, conn_matrix[j][i])
                elif pm.both_inh(i,j):
                    conn_matrix[j][i]= pm.flip(0.20,i)
                    countA = pm.check_count(countA, conn_matrix[j][i])
                elif i in pm.exc_nrns_set and j in pm.inh_nrns_set:
                    conn_matrix[j][i]= pm.flip(0.10,i)
                    countA = pm.check_count(countA, conn_matrix[j][i])
                else:
                    conn_matrix[j][i]= pm.flip(0.50,i)
                    countA = pm.check_count(countA, conn_matrix[j][i])
            # FROM LAYER2/3 -> LAYER5
            elif i in pm.layers23 and j in pm.layers5:
                if pm.both_exc(i,j):
                    conn_matrix[j][i]= pm.flip(0.55,i)
                    countB = pm.check_count(countB, conn_matrix[j][i])
                elif pm.both_inh(i,j):
                    conn_matrix[j][i]= pm.flip(0.0001,i)
                    countB = pm.check_count(countB, conn_matrix[j][i])
                elif i in pm.exc_nrns_set and j in pm.inh_nrns_set:
                    conn_matrix[j][i]= pm.flip(0.001,i)
                    countB = pm.check_count(countB, conn_matrix[j][i])
                else:
                    conn_matrix[j][i]= pm.flip(0.20,i)
                    countB = pm.check_count(countB, conn_matrix[j][i])
        #:::
        elif not pm.same_minicolumn(i,j):
            conn_matrix[j][i] = pm.flip(0.102,i)
            countQ = pm.check_count(countQ,conn_matrix[j][i])

print ("Matrix Created in %.5s seconds." % (time.time() - start_time))

#_________________________________________________________________________________________

#"""
print "connections 2/3 = ", count23
print "connections 4 = ", count4
print "connections 5 = ", count5
print "not same minicolumn, same hypercolumn = ",countQ
print "not same hypercolumn = ",countAz
print "connections 30%", count23+count4+count5+countQ
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
#    balance(conn_matrix[k[i],:])
#"""
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
        if abs(conn_matrix[j][i]) > 1e-4:
            g+=1
print "total connections: ", g

print "% connections within a minicolumn= ", round((count23+count4+count5+countA+countB)/float(g)*100.)
print "% connections outside a minicolumn= ", round(float(float(countQ)/float(g))*100.)
print

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


gh = 0
for i in conn_matrix[1][:]:
    if i < -1e-4 or i > 1e-4:
        gh+=1
print gh
#"""
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#"""


ee = la.eigvals(conn_matrix)
conn_matrix.dump("110.dat")
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
        yid = plt.find(yl>=ee[ii].real)[0]
        con_density[xid,yid] = con_density[xid,yid]+1
    st.append(con_density)
for i in st:
    pcolor(i)
colorbar()

#con_density = con_density#/len(ee)
pcolor(con_density)

print ("Matrix Created in %.5s seconds." % (time.time() - start_time))

print str(100.-(len(plt.find(abs(ee) > 15))/float(len(ee)) *100.))+" of the eigenvalues are in the 15 radius circle"

print "two largest eigenvalues by magnitude %.5s, " % abs(ee)[0]+"%.5s " %abs(ee)[1]
print "Loading plot..."



plt.figure(1)
#plt.subplot(1,2,1)
hist5,xedges,yedges = np.histogram2d(ee.real,ee.imag,bins=30,normed=False)
extent = [-20, 20, -20, 20 ]
plt.imshow(hist5.T,extent=extent,interpolation='nearest',origin='lower')
#plt.colorbar()
#plt.show()

ed = np.linspace(-4.,1,1e3)
hh,ed= np.histogram(conn_matrix.flatten(),ed)

tt = np.linspace(np.pi,-np.pi,1e2)
sx = np.sin(tt)
sy = np.cos(tt)



ed_ev = np.linspace(-20,20,1e2)
hh_real,ed1 = np.histogram(ee.real,ed_ev)
hh_imag,ed1 = np.histogram(ee.imag,ed_ev)


#plt.clf()
#plt.subplot(1,2,2)
plt.scatter(ee.real,ee.imag)
plt.plot(sx,sy,'r')
plt.plot(15*sx,15*sy,'g')
#plt.pcolor(conn_matrix, cmap=plt.cm.Blues)
plt.title("%.8s variance," % pm.sigma**2 +str(pm.mu)+" mean")
#plt.axis('equal')
plt.xlim(min(ed_ev),max(ed_ev))
plt.ylim(min(ed_ev),max(ed_ev))
plt.show()


noB_var_row = np.var(conn_matrix,1)
noB_var_col = np.var(conn_matrix,0)
B_var_row = np.var(conn_matrix,1)
B_var_col = np.var(conn_matrix,0)


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