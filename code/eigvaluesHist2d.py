"""
Script to create a 2d histogram of eigenvalue spectra for each model
fnames : contains the names for each model

Instructions:   Set 't' equal to the fnames' list item corresponding to the
                model that you want to plot its 2d histogram of its eigenvalue
                spectra.
"""
import numpy as np
from scipy import linalg as la
import pylab as py


fnames= []
for i in [0,1]:
    for j in [0,1]:
        for k in [0,1]:
           fnames.append(str(i)+str(j)+str(k))

conn_matrix = [[] for i in range(len(fnames))]
eig = conn_matrix


for i in range(len(fnames)):
    conn_matrix[i] = np.load('./dumps/'+fnames[i]+'conmat.dat')
    eig[i] = np.load('./dumps/'+fnames[i]+'eigvals.dat')


print fnames
t=7
print fnames[t]
trials = [eig[t]]

xl = np.linspace(-30,30,61)
yl = np.linspace(-30,30,61)

con_density = np.zeros((len(xl),len(yl)))
st = []


for i in range(len(trials)):
    ee = trials[i]
    for ii in range(len(ee)):
        xid = py.find(xl>=ee[ii].imag)[0]
        if len(py.find(yl>=ee[ii].real)) > 0:
            yid = py.find(yl>=ee[ii].real)[0]
        else:
            yid = abs(trials[i][0])
        con_density[xid,yid] = con_density[xid,yid]+1
    st.append(con_density)


fig,ax = py.subplots()
# Format
fig = py.gcf()
for i in st:
    ax.grid(False)
    py.pcolor(i)
py.show()