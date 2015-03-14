import pylab
import numpy as np
import sys
import os

def plot_conn_list(conn_list_fn, params=None, clim=None, src_cell_type=None):
    print 'Loading data from:', conn_list_fn, '\n......'
    data = np.loadtxt(conn_list_fn)

    src_min, src_max= np.min(data[:, 0]), np.max(data[:, 0])
    n_src = src_max - src_min + 1
    tgt_min, tgt_max= np.min(data[:, 1]), np.max(data[:, 1])
    n_tgt = tgt_max - tgt_min + 1
    conn_mat = np.zeros((n_src, n_tgt))
    print 'src_min, src_max', src_min, src_max
    print 'tgt_min, tgt_max', tgt_min, tgt_max
    for c in xrange(data[:,0].size):
        src = data[c, 0] - src_min
        tgt = data[c, 1] - tgt_min
        conn_mat[src, tgt] = data[c, 2]

    plot_matrix(conn_mat, clim=clim)
    return conn_mat



def plot_matrix(data, clim=None):
    cmap_name = 'bwr'
    # find other colormaps here: http://matplotlib.org/examples/color/colormaps_reference.html

    if clim != None:
        norm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])#, clip=True)
        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_name)
        m.set_array(np.arange(clim[0], clim[1], 0.01))

    fig = pylab.figure()
    ax = fig.add_subplot(111)
    print "plotting .... "

    if clim != None:
        cax = ax.pcolormesh(data, cmap=cmap_name, vmin=clim[0], vmax=clim[1])
    else:
        cax = ax.pcolormesh(data, cmap=cmap_name)
    pylab.ylim(0, data.shape[0])
    pylab.xlim(0, data.shape[1])
    pylab.colorbar(cax)



if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print("Please enter data folder to be analysed after the script name")
        print("e.g.\npython analyse_data_folder.py data_today/ ")
        exit(1)
    else:
        fn = sys.argv[1]

    plot_conn_list(fn)
    # saving the figure
    #plot_fn = "testfig.png"
    #print "saving ....", plot_fn
    #pylab.savefig(plot_fn)
    pylab.show()
