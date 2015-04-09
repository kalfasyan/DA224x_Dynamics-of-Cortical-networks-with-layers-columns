import numpy as np
import pylab as pl

spr = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

no_con = 1e5

wt_var = np.zeros(len(spr))
for ii in range(len(spr)):
	wt = np.zeros(no_con)
	oc = int(no_con*spr[ii])
	wt[0:oc] = 1.
	wt_var[ii] = np.var(wt)


pl.figure(1)
pl.clf()
pl.plot(spr,wt_var)
pl.show()
