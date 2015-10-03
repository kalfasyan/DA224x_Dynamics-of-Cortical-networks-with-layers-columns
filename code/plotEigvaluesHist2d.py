"""
Example:
run plotEigvaluesHist2d.py 111 LMH

- This created the 2d histogram for eig-distribution for model 111 (LMH)
- LMH is the model's codename, make sure to use the correct respective codenames
    after the model
"""
import numpy as np
from scipy import linalg as la
import pylab as py
import sys


modelname = sys.argv[1]
figurename = sys.argv[2]
nr_models = 8

eig = np.load('./dumps/'+modelname+'eigvals.dat')


linX = np.linspace(-30,30,61)
linY = np.linspace(-30,30,61)

con_density = np.zeros((len(linX),len(linY)))
st = []


eig = [eig]
for i in range(len(eig)):
    ee = eig[i]
    for jj in range(len(ee)):
        # Finds
        xid = py.find(linX>=ee[jj].imag)[0]
        if len(py.find(linY>=ee[jj].real)) > 0:
            yid = py.find(linY>=ee[jj].real)[0]
        else:
            if abs(eig[i][0])>60.:
                yid = 60.
            else:
                yid = abs(eig[i][0])
        con_density[xid,yid] = con_density[xid,yid]+1
    st.append(con_density)


fig,ax = py.subplots()
# Format
fig = py.gcf()
for i in st:
    #ax.grid(False)
    py.imshow(i,interpolation='nearest',aspect='auto')#,cmap='seismic')
#py.axis('off')

py.gcf().tight_layout()
py.axis('equal')
py.colorbar()
py.title(figurename+' eigenvalue spectrum')
#py.savefig('eig'+figurename+'.png', bbox_inches='tight')
py.show()
