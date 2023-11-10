import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from colorsys import hls_to_rgb

def colorize(z):
    r = np.abs(z)
    arg = np.angle(z) 

    h = (arg + pi)  / (2 * pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2) 
    return c

def waveform2(n, F, t_indir):
    '''t_indir - indirect time values in terms of real time'''
    form = np.zeros((n,n)).astype(complex)
    for t2, tr in zip(t_indir, range(n)):
        form[t2,:] = np.array([F(t1,t2,tr) for t1 in range(n)])
    return form

class Signal():
    def __init__(self, form=np.array([0])):
        self.timedom = np.array(form)
        self.shape = self.timedom.shape
        self.len = self.shape[0]
        self.dim = len(self.shape)
        self.freqdom = np.fft.fft2(self.timedom,None,[i for i in range(self.dim)])

    def update(self):
        self.freqdom = np.fft.fft2(self.timedom,None,[i for i in range(self.dim)])

    def add(self,form):
        if self.timedom.any():
            self.timedom = form
            return
        self.timedom += form

    def deshuffle(self,t_indir):
        inverse_perm = np.empty_like(t_indir)
        inverse_perm[t_indir] = np.arange(t_indir.size)
        self.timedom = self.timedom[inverse_perm]
        self.update()

    
    def plot(self, type="time"):
        if type == "time":
            if self.dim == 1:
                plt.plot(self.timedom.real)
            else:
                plt.imshow(self.timedom.real)
            plt.show()
        if type == "freq":
            if self.dim == 1:
                plt.plot(self.freqdom.real)
            else:
                plt.imshow(abs(self.freqdom))
            plt.show()


n = 512

###
### Effects of non-stationary frequency
###

def non_stationary_frequency(t1,t2,treal):
    f1, f2 = 0.1+0.02*treal/n , 0.2 + 0.02*treal/n
    tau1, tau2 = n/2, n/2
    return np.exp(2j*pi*(f1*t1+f2*t2))

t_indir = np.random.choice(n,n,replace=False)

signal_lin = Signal(waveform2(n,non_stationary_frequency,range(n)))
signal_perp = Signal(waveform2(n,non_stationary_frequency,t_indir))


# signal_lin.plot("freq")
# signal_perp.plot("freq")

###
### Time-resolved NUS
###

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
    ss = Signal(snapshot_2d(n,non_stationary_frequency,range(*t_real_interval),t_indir))
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


###
### CS reconstruction and NUS - non-stationarity compromise
###

