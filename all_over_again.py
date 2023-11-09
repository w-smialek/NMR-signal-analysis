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

class Signal():
    def __init__(self, form=np.array([0])):
        self.timedom = np.array(form)
        self.shape = self.timedom.shape
        self.len = self.shape[0]
        self.dim = len(self.shape)
        self.freqdom = np.fft.fft2(self.timedom,None,[i for i in range(self.dim)])

    def add(self,form):
        if self.timedom.any():
            self.timedom = form
            return
        self.timedom += form
    
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


n = 100

form = [[np.exp(2j*pi*(t1*(0.1+t2*0.1/n)+t2*(0.3+t2/n*0.1))) for t1 in range(n)] for t2 in range(n)]


signal = Signal()

signal.plot("time")
signal.plot("freq")

