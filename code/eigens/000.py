import numpy as np
from scipy import linalg as la
import time
import params000 as pm



start_time = time.time()
print "Initializing and creating connection matrix..."
conn_matrix = np.zeros((pm.nrns,pm.nrns))
#pbar.start()
count23,count4,count5,countA,countB,countAz,countQ,countW = 0,0,0,0,0,0,0,0
for i in range(pm.nrns):
    for j in range(pm.nrns):
        conn_matrix[j][i] = pm.flip(.1027,i)
        countAz = pm.check_count(countAz, conn_matrix[j][i])

#"""

#_________________________________________________________________________________________

"""
print "connections 2/3 = ", count23
print "connections 4 = ", count4
print "connections 5 = ", count5
print "not same minicolumn, same hypercolumn = ",countQ
print "not same hypercolumn = ",countAz
print "connections 30%", count23+count4+count5+countQ
print "connections 70%", countAz
"""



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

k=[]
for i in range(pm.nrns):
    if np.sum(conn_matrix[i,:]) > 1e-5:
        k.append(i)
print "Row sums not zero", len(k)
pm.balanceN(conn_matrix)
#for i in range(len(k)):
#    pm.balance(conn_matrix[k[i],:])

g = 0
for i in range(pm.nrns):
    for j in range(pm.nrns):
        if conn_matrix[j][i] < -1e-4 or conn_matrix[j][i] > 1e-4:
            g+=1
print g

print "% connections within a hypercolumn= ", round((count23+count4+count5+countQ+countA+countB)/float(g)*100.)
print "% connections outside a hypercolumn= ", round(float(float(countAz)/float(g))*100.)
print

delta =0
for i in range(pm.nrns):
    if np.sum(conn_matrix[i,:]) > 1e-5:
        delta+=1
        #print np.sum(conn_matrix[i,:])
        #print i
print "sum of all matrix",np.sum(conn_matrix)
print "Row sums not to zero after balance",delta
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
"""
gh = 0
for i in conn_matrix[:,15]:
    print i

    if abs(i) > 1e-4:
        gh+=1
print "gh", gh
#"""
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#"""
ee = la.eigvals(conn_matrix)

conn_matrix.dump("000.dat")
print "Done!"

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

print ("Matrix Created in %.5s seconds." % (time.time() - start_time))

print str(100.-(len(plt.find(abs(ee) > 15))/float(len(ee)) *100.))+" of the eigenvalues are in the 15 radius circle"

print "two largest eigenvalues by magnitude %.5s, " % abs(ee)[0]+"%.5s " %abs(ee)[1]
print "Loading plot..."

noB_var_row = np.var(conn_matrix,1)
noB_var_col = np.var(conn_matrix,0)
B_var_row = np.var(conn_matrix,1)
B_var_col = np.var(conn_matrix,0)

ed = np.linspace(-8.,3,1e3)
hh,ed= np.histogram(conn_matrix.flatten(),ed)

tt = np.linspace(np.pi,-np.pi,1e2)
sx = np.sin(tt)
sy = np.cos(tt)


#ee = la.eigvals(conn_matrix)

ed_ev = np.linspace(-20,20,1e2)
hh_real,ed1 = np.histogram(ee.real,ed_ev)
hh_imag,ed1 = np.histogram(ee.imag,ed_ev)

plt.figure(1)
#plt.clf()
plt.subplot(3,2,1)
plt.scatter(ee.real,ee.imag)
plt.plot(sx,sy,'r')
plt.plot(15*sx,15*sy,'g')
#plt.pcolor(conn_matrix, cmap=plt.cm.Blues)
plt.title("%.8s variance," % pm.sigma**2 +str(pm.mu)+" mean")
plt.axis('equal')
plt.xlim(min(ed_ev),max(ed_ev))
plt.ylim(min(ed_ev),max(ed_ev))
#plt.show()

plt.subplot(3,2,2)
plt.plot(hh_imag,ed_ev[0:-1])
plt.ylim(min(ed_ev),max(ed_ev))
#plt.ylim(0,100)
plt.xlabel("two largest eigenvalues by magnitude %.5s, " % abs(ee)[0]+"%.5s " %abs(ee)[1])

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
