"""
Finds and plots percentage of pairs and number of shared connections of neurons
Specifically tweaked now to plot two models in the same plot.
"""
import numpy as np
import pylab as py
import sys
from sys import stdout
from time import sleep

modelname1  = '100'#sys.argv[1]#001
figurename1 = 'L'

modelname2  = '101'
figurename2 = 'LH'

""" -------------------------------------------------------- """
def targets_of(mat):
    """ Finds the targets of each neuron """
    neuron_conns = [[] for i in range(len(mat))]
    for jj in range(len(mat)):
        target_ids = py.find((np.abs(mat[:,jj])>0.))
        neuron_conns[jj] = target_ids.tolist()
    return neuron_conns

def testor(modelname,figurename,clr):
    c1 = np.load('./eigens/'+modelname+'.dat')
    print len(targets_of(c1))
    a = targets_of(c1)
    degcor = np.zeros((len(a),len(a)))
    geo_mean = np.zeros((len(a),len(a)))
    norm_deg_corr = np.zeros((len(a),len(a)))


    for i in range(len(a)):
        for j in range(len(a)):
            if i > j:
                ax = len(list(set(a[i]) & set(a[j])))
                bx = np.mean((len(a[i]), len(a[j])))
                degcor[i][j] = ax
                geo_mean[i][j] = bx
                norm_deg_corr[i][j] = float(ax)/float(bx)
            #else:
            #    degcor[i][j] = len(list(set(a[i]) & set(a[j])))
        if i % 100 == 0.:
            stdout.write("\r%d" % int(float(i)/float(len(a))*100))
            stdout.write("%")
            stdout.flush()
            sleep(1)
    stdout.write("\n")

    N= 2880
    total_pairs = N*(N-1)/2.
    print degcor

    x1 = np.reshape(degcor,(1,N**2))
    x2 = x1.flatten()
    ed = np.arange(1,np.max(x2),1)
    hh,edx = np.histogram(x2,ed)
    #py.figure(modelname)
    py.plot(ed[0:-1],100*hh/total_pairs, clr)
    py.xlim(0,np.max(x2))
""" ------------------------------------------------------- """

py.figure(modelname1+" "+modelname2)
testor(modelname1,figurename1,'r')
testor(modelname2,figurename2, 'b')
py.title(figurename1+" and "+ figurename2+' degree correlations')
py.ylabel('Percentage % of pairs')
py.xlabel('Number of shared connections')
py.legend([figurename1,figurename2])
#py.savefig(modelname+'.png', bbox_inches='tight')
py.show()