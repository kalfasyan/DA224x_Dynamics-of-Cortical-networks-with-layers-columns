import parameters_v1 as pm
import numpy as np
import pylab as py
import itertools

modelname = '011'
c1 = np.load(modelname+'.dat')
counter = np.zeros((3,3))

""" Finds the targets of laminar component """
def targets_of_comp(mat, layer):
    targets = []
    # Iterating sources and finding targets
    for j in range(len(mat)):
        targets.append( py.find( (np.abs(c1[:,j])>0.) & (j in layer) ) )
    return set(list(itertools.chain(*targets)))

s,labels = [],[]
for i in range(len(pm.laminar_components(modelname)[0])):
    s.append(targets_of_comp(c1,pm.laminar_components(modelname)[0][i]))
    labels.append(pm.laminar_components(modelname)[1][i])

intersect = np.zeros((len(s),len(s)))

for i in range(len(s)):
    for j in range(len(s)):
        intersect[i][j] = len(set.intersection(s[i],s[j]))

print intersect


""" PLOTTING """
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