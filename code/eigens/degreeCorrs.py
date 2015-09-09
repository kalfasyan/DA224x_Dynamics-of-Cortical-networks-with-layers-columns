import parameters_v1 as pm
import numpy as np
import pylab as py
import itertools

modelname = '101'
c1 = np.load(modelname+'.dat')
#c1 = np.random.random_integers(-5,5,size=(10,10))

""" Finds the targets of laminar component """
def targets_of(mat, comp):
    cons = [[] for i in range(len(mat))]
    for j in range(len(mat)):
        d = py.find((np.abs(mat[:,j]>0.)))#& (j in comp)))
        cons[j] = d.tolist()
    return cons

comps = pm.laminar_components(modelname)[0]

print len(targets_of(c1,comps[0]))
a = targets_of(c1,comps[0])
d = np.zeros((len(a),len(a)))

for i in range(len(a)):
    for j in range(len(a)):
        d[i][j] = len(list(set(a[i]) & set(a[j])))

print d
x1 = np.reshape(d,(1,2880**2))
x2 = x1.flatten()
ed = np.arange(0,400,1)
hh,edx = np.histogram(x2,ed)
py.figure(1004)
py.plot(ed[0:-1],hh)
py.show()
"""
for i in range(len(components)):
    A = targets_of(c1,components[i])
    for j in range(len(components)):
        B = targets_of(c1,components[j])
        aset = set([tuple(x) for x in A])
        bset = set([tuple(x) for x in B])
        print len(np.array([x for x in aset & bset]))-1



print "Done!"

intersect = np.zeros((len(components),len(components)))

for i in range(len(components)):
    t1 = targets_of(c1,components[i])
    for j in range(len(components)):
        t2 = targets_of(c1,components[j])
        intersect[i][j] = len(list(set(map(tuple,t1)).intersection(set(map(tuple,t2)))))

#print list(set(map(tuple,t1)).intersection(set(map(tuple,t2))))

py.imshow(intersect,interpolation='none',extent=[0,len(components),0,len(components)])
py.show()



fig,ax = py.subplots()
#py.imshow(intersect,interpolation='none',extent=[0,len(s),0,len(s)])

# turn off the frame
ax.set_frame_on(False)
# Format
fig = py.gcf()
# put the major ticks at the middle of each cell
ax.set_yticks(np.arange(intersect.shape[0]) + 0.5, minor=False)
ax.set_xticks(np.arange(intersect.shape[1]) + 0.5, minor=False)
# setting the labels
ax.set_xticklabels(labels, minor=False)
ax.set_yticklabels(labels,minor=False)

im = ax.pcolor(intersect,cmap=py.cm.Blues)
for y in range(intersect.shape[0]):
    for x in range(intersect.shape[1]):
        py.text(x + 0.5, y + 0.5, '%.0f' % intersect[y, x],
                 horizontalalignment='center',
                 verticalalignment='center',
                 )
for label in im.axes.xaxis.get_ticklabels():
    label.set_rotation(90)
#fig.colorbar(im)
ax.grid(False)
ax = py.gca()
py.show()

"""