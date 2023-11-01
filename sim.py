import numpy as np
import matplotlib.pyplot as plt

class Waveform:
    def __init__(self,l):
        self.form = np.array(l)
        self.size = self.form.size

    def real(self):
        return np.real(self.form)
    
    def imag(self):
        return np.imag(self.form)
    
    def ft(self):
        return Waveform(np.fft.fft(self.form))
    
class Waveform2D(Waveform):

    def ft(self):
        return Waveform2D(np.fft.fft2(self.form))
    
    def n_dir(self):
        return np.shape(self.form)[0]
    
    def n_indir(self):
        return np.shape(self.form)[1]
    
    
class Signal:
    def __init__(self,delta1):
        self.dt1 = delta1
        self.timedom = None

    def update(self):
        self.freqdom = self.timedom.ft()
    
    def set_signal(self, signal):
        self.timedom = Waveform(signal)
        self.update()

    def add_signal(self, signal):
        if self.timedom == None:
            self.set_signal(signal)
        else:
            self.timedom.form += np.array(signal)
            self.update()

    def plot(self, type="time", part = "complex"):
        if type == "time":
            if not part == "imag":
                plt.plot([i*self.dt1 for i in range(self.timedom.size)], self.timedom.real())
            if not part == "real":
                plt.plot([i*self.dt1 for i in range(self.timedom.size)], self.timedom.imag())
            plt.xlabel("[s]")
            plt.show()
        if type == "freq":
            if not part == "imag":
                plt.plot([i/self.freqdom.size/self.dt1 for i in range(self.freqdom.size)], self.freqdom.real())
            if not part == "real":
                plt.plot([i/self.freqdom.size/self.dt1 for i in range(self.freqdom.size)], self.freqdom.imag())
            plt.xlabel("[Hz]")
            plt.show()

class Signal2D(Signal):
    def __init__(self, delta1, delta2):
        super().__init__(delta1)
        self.dt2 = delta2

    def set_signal(self, signal):
        self.timedom = Waveform2D(signal)
        self.update()

    def direct_slice(self,indirect_t):
        dir_sl = Signal(self.dt1)
        dir_sl.set_signal(self.timedom.form[indirect_t,:])
        return dir_sl
    
    def set_direct_slice(self,l,indirect_t):
        self.timedom.form[indirect_t,:] = np.array(l)
        self.update()
    
    def add_direct_slice(self,l,indirect_t):
        self.timedom.form[indirect_t,:] += np.array(l)
        self.update()

    def sparse_sample(self, samples):
        for k in range(self.timedom.n_indir()):
            if not k in samples:
                self.set_direct_slice(np.zeros(self.timedom.n_indir()),k)
        self.update()

    def plot_direct(self, indirect_t ,type="time", part = "complex"):
        print(self.direct_slice(indirect_t).timedom.form)
        self.direct_slice(indirect_t).plot(type,part)

    def plot2D(self):
        plt.imshow(self.timedom.real(), interpolation='none')
        plt.show()

    def freqplot2D(self):
        plt.imshow(self.freqdom.real(), interpolation='none')
        plt.show()


def sinexp(x,ampl,freq,tau):
    return ampl*np.exp(x*(2*np.pi*freq*1j - 1/tau))

def whitenoise_complex(stdev,n):
    rng = np.random.default_rng()
    return rng.normal(0,stdev/2,n) + 1j* rng.normal(0,stdev/2,n)

def sinexp_2d_list(ampls,freqs,taus,ns):
    return [[sinexp(t1,ampls[0],freqs[0],taus[0])*sinexp(t2,ampls[1],freqs[1],taus[1]) for t1 in range(ns[0])] for t2 in range(ns[1])]


# s = Signal(0.0001)

# n = 300
# ampls = (1,3,2)
# freqs = (0.1,0.04,0.3)
# taus = (35,60,6)
# for (a,f,t) in zip(ampls,freqs,taus):
#     s.add_signal([sinexp(x,a,f,t) for x in range(n)])

# s.plot("time")
# s.plot("freq","real")

# noise = whitenoise_complex(stdev=20,n=n)

# s.add_signal(noise)

# s.plot("time")
# s.plot("freq","real")

n = 100

form2d = [[sinexp(t1,1,0.3+t2*0.3/100,30)*sinexp(t2,1,0.15,60) for t1 in range(n)] for t2 in range(n)]

s2 = Signal2D(0.0001,0.0001)
# s2.add_signal(form2d)
# s2.plot2D()
# s2.freqplot2D()

# form2d = [[sinexp(t1,1,0.3,30)*sinexp(t2,np.exp(-t2/4),0.15,20) for t1 in range(n)] for t2 in range(n)]

# s2.set_signal(form2d)

# s2.plot2D()
# s2.freqplot2D()


# form2d = sinexp_2d_list((1,1),(0.3,0.15),(30,60),(n,n))

s2.add_signal(sinexp_2d_list((1,1),(0.3,0.15),(30,60),(n,n)))
s2.add_signal(sinexp_2d_list((1,0.6),(0.5,0.8),(20,24),(n,n)))
s2.add_signal(sinexp_2d_list((1.3,0.4),(0.26,0.15),(15,45),(n,n)))
s2.add_signal(whitenoise_complex(0.2,(n,n)))
s2.plot2D()
s2.freqplot2D()

sampling = []
i = 0
while True:
    sampling.append(int(i**1.8))
    i += 1
    if max(sampling) > 100:
        del sampling[-1]
        break

s2.sparse_sample(sampling)

s2.plot2D()
s2.freqplot2D()