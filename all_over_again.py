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
    return np.array([[F(t1,t2,treal) for t1 in range(n)] for t2, treal in zip(t_indir,range(n))])

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
                plt.imshow(self.freqdom.real)
            plt.show()


n = 500

def non_stationary_frequency(t1,t2,treal):
    f1, f2 = 0.1*treal/n , 0.1*treal/n
    tau1, tau2 = n/2, n/2
    return np.exp(2j*pi*(f1*t1+f2*t2))

t_indir = np.random.choice(n,n,replace=False)

signal_lin = Signal(waveform2(n,non_stationary_frequency,range(n)))
signal_perp = Signal(waveform2(n,non_stationary_frequency,t_indir))


signal_lin.plot("time")
signal_lin.plot("freq")

signal_perp.deshuffle(t_indir)
signal_perp.plot("time")
signal_perp.plot("freq")

###
### Time-resolved NUS
###

def snapshot_2d(n, F, treal, t_indir):
    '''n - number of direct time datapoints \n
    treal - range of real time values \n
    t_indir - corresponding indirect time values'''
    
