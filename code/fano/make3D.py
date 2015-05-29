from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import proj3d


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fname = '101'

try:
    with open("fanoInh"+fname+".txt","r") as f, open("fanoExt"+fname+".txt","r") as g, open("fanoExc"+fname+".txt","r") as h, open("../ffnames.txt","r") as j:
        inh = [float(i) for i in f.readlines()]
        ext = [float(i) for i in g.readlines()]
        exc = [float(i) for i in h.readlines()]
        labels = [i[0:-1] for i in j.readlines()]
except IOError as e:
    print 'Operation Failed: %s' % e.strerror

xt =exc
yt =inh
zt =ext

ax.scatter(xt, yt, zt, c='b', marker='^')

ax.set_xlabel('exc')
ax.set_ylabel('inh')
ax.set_zlabel('ext')

plt.show()