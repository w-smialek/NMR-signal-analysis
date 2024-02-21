import signals as sig

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi

##
## Effects of non-stationary frequency
##

n = 200

## 1D

form  = [1*np.exp(2j*pi*(0.1+0.05*t/n)*t - t/(n/2)) for t in range(n)]
noise = sig.whitenoise_complex(0.1,n)

sig_1D_nonstat = sig.Signal(form)
sig_1D_nonstat.add(noise)
sig_1D_nonstat.dt = 1/n

plt.style.use('ggplot')

# sig_1D_nonstat.plot("freq", linewidth = 0.8)
# # plt.savefig("1d_nonstat_example.png",dpi=300)

## 2D

def non_stationary_frequency(t1,t2,treal):
    f1, f2 = 0.1 + 0.07*treal/n, 0.2 + 0.07*treal/n
    tau1 = n#n/2
    return np.exp(2j*pi*(f1*t1+f2*t2) - t1/tau1)

###
###
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def block_perm_sampling(n_pieces, perm, signs):
    pieces_forw = [list(i) for i in list(split(range(n), n_pieces))]
    pieces_backw = [list(i) for i in list(split(range(n-1,-1,-1), n_pieces))]
    pieces_backw.reverse()
    sampling = []
    for p,s in zip(perm,signs):
        if s:
            sampling += pieces_forw[p]
        else:
            sampling += pieces_backw[p]
    return sampling

def invert_permutation(p):
    """Return an array s with which np.array_equal(arr[p][s], arr) is True.
    The array_like argument p must be some permutation of 0, 1, ..., len(p)-1.
    """
    p = np.asanyarray(p) # in case p is a tuple, etc.
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s

t_indir = block_perm_sampling(4,[3,2,1,0],4*[True])

plt.plot(t_indir)
plt.show()

###
###

# t_indir = np.random.choice(n,n,replace=False)

signal_lin = sig.Signal(sig.waveform2(n,non_stationary_frequency,range(n)))

signal_perp = sig.Signal(sig.waveform2(n,non_stationary_frequency,t_indir))

signal_lin.dt1 = signal_lin.dt2 = signal_perp.dt1 = signal_perp.dt2 = 1/(n/2)

plt.style.use('classic')

signal_lin.plot("freq",cmap="PiYG")
# # plt.savefig("indirect_lin.png",dpi=300)
plt.show()
plt.close()
signal_perp.plot("freq",cmap="PiYG")
# plt.savefig("indirect_perp_big.png",dpi=300)
plt.show()
plt.close()

##
## Time-resolved NUS
##

def snapshot_2d(n, F, t_real, t_indir):
    '''n - number of direct time datapoints \n
    treal - range of real time values \n
    t_indir - corresponding indirect time values'''
    form = np.zeros((n,n)).astype(complex)
    for t2, tr in zip(t_indir, t_real):
        form[t2,:] = np.array([F(t1,t2,tr) for t1 in range(n)])
    return form

def average_over_neighbours(array):
    ars = [np.roll(array,t,(0,1)) for t in [(-1,0),(0,-1),(0,0),(0,1),(1,0),(-1,-1),(1,1),(-1,1),(1,-1)]]
    return sum(ars)

n_snapshots = 80
treal_interval = 20

snaphot_sampling_ratio = treal_interval/n

snapshots = []
for i in range(n_snapshots):
    t_real_interval = (int(treal_interval*i),int(treal_interval*(i+1)))
    # t_indir = np.random.choice(n,int(n*snaphot_sampling_ratio),replace=False)
    ss = sig.Signal(snapshot_2d(n,non_stationary_frequency,range(*t_real_interval),t_indir))
    ss.freqdom = average_over_neighbours(ss.freqdom)
    snapshots.append(ss)

maxes = []

for snap in snapshots:
    # snap.plot(type="time")
    # snap.plot(type="freq")
    maxes.append(np.argmax(snap.freqdom))

mm = np.zeros((n,n))
for m in maxes:
    mm[(m//n,m%n)] = 1

plt.imshow(mm)
plt.show()

snapshots = []
for i in range(n_snapshots):
    t_real_interval = (int(treal_interval*i),int(treal_interval*(i+1)))
    # t_indir = np.random.choice(n,int(n*snaphot_sampling_ratio),replace=False)
    ss = sig.Signal(snapshot_2d(n,non_stationary_frequency,range(*t_real_interval),range(n)))
    ss.freqdom = average_over_neighbours(ss.freqdom)
    snapshots.append(ss)

maxes = []

for snap in snapshots:
    # snap.plot(type="time")
    # snap.plot(type="freq")
    maxes.append(np.argmax(snap.freqdom))

mm = np.zeros((n,n))
for m in maxes:
    mm[(m//n,m%n)] = 1

plt.imshow(mm)
plt.show()
