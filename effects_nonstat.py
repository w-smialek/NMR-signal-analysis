import signals as sig

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi

##
## Effects of non-stationary frequency
##

n = 100

## 1D

form  = [1*np.exp(2j*pi*(0.1+0.05*t/n)*t - t/(n/2)) for t in range(n)]
noise = sig.whitenoise_complex(0.1,n)

sig_1D_nonstat = sig.Signal(form)
sig_1D_nonstat.add(noise)
sig_1D_nonstat.dt = 1/n

plt.style.use('ggplot')

sig_1D_nonstat.plot("freq", linewidth = 0.8)
plt.savefig("1d_nonstat_example.png",dpi=300)

## 2D

def non_stationary_frequency(t1,t2,treal):
    f1, f2 = 0.1, 0.2 + 0.02*treal/n
    tau1 = n/2
    return np.exp(2j*pi*(f1*t1+f2*t2) - t1/tau1)

t_indir = np.random.choice(n,n,replace=False)

signal_lin = sig.Signal(sig.waveform2(n,non_stationary_frequency,range(n)))

signal_perp = sig.Signal(sig.waveform2(n,non_stationary_frequency,t_indir))

signal_lin.dt1 = signal_lin.dt2 = signal_perp.dt1 = signal_perp.dt2 = 1/(n/2)

plt.style.use('classic')

signal_lin.plot("freq",cmap="PiYG")
# plt.savefig("indirect_lin.png",dpi=300)
plt.show()
signal_perp.plot("freq",cmap="PiYG")
plt.savefig("indirect_perp_big.png",dpi=300)
plt.show()

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
    t_indir = np.random.choice(n,int(n*snaphot_sampling_ratio),replace=False)
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
