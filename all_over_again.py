import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from colorsys import hls_to_rgb
import cvxpy as cp

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

n = 30

###
### Effects of non-stationary frequency
###

# def non_stationary_frequency(t1,t2,treal):
#     f1, f2 = 0.1+0.02*treal/n , 0.2 + 0.02*treal/n
#     tau1, tau2 = n/2, n/2
#     return np.exp(2j*pi*(f1*t1+f2*t2))

def non_stationary_frequency(t1,t2,treal):
    f1, f2 = 0.1, 0.2
    f3, f4 = 0.3, 0.6
    tau1, tau2 = n/2, n/2
    return np.exp(2j*pi*(f1*t1+f2*t2)-t1/tau1) + np.exp(2j*pi*(f3*t1+f4*t2)-t1/tau2)

t_indir = np.random.choice(n,n,replace=False)

signal_lin = Signal(waveform2(n,non_stationary_frequency,range(n)))

signal_perp = Signal(waveform2(n,non_stationary_frequency,t_indir))

# signal_lin.plot("freq")
# signal_perp.plot("freq")

###
### Time-resolved NUS
###

# def snapshot_2d(n, F, t_real, t_indir):
#     '''n - number of direct time datapoints \n
#     treal - range of real time values \n
#     t_indir - corresponding indirect time values'''
#     form = np.zeros((n,n)).astype(complex)
#     for t2, tr in zip(t_indir, t_real):
#         form[t2,:] = np.array([F(t1,t2,tr) for t1 in range(n)])
#     return form

# def average_over_neighbours(array):
#     ars = [np.roll(array,t,(0,1)) for t in [(-1,0),(0,-1),(0,0),(0,1),(1,0),(-1,-1),(1,1),(-1,1),(1,-1)]]
#     return sum(ars)

# n_snapshots = 80
# treal_interval = 20

# snaphot_sampling_ratio = treal_interval/n

# snapshots = []
# for i in range(n_snapshots):
#     t_real_interval = (int(treal_interval*i),int(treal_interval*(i+1)))
#     t_indir = np.random.choice(n,int(n*snaphot_sampling_ratio),replace=False)
#     ss = Signal(snapshot_2d(n,non_stationary_frequency,range(*t_real_interval),t_indir))
#     ss.freqdom = average_over_neighbours(ss.freqdom)
#     snapshots.append(ss)

# maxes = []

# for snap in snapshots:
#     # snap.plot(type="time")
#     # snap.plot(type="freq")
#     maxes.append(np.argmax(snap.freqdom))

# mm = np.zeros((n,n))
# for m in maxes:
#     mm[(m//n,m%n)] = 1

# plt.imshow(mm)
# plt.show()


###
### CS reconstruction and NUS - non-stationarity compromise
###

### 1D

sampling_ratio = 0.3

def sampling_matrix(sampling_mask):
    '''rectangular sampling matrix from a vector 0 or 1 sampling mask'''
    sampling_mat = np.array([row for row in np.diag(sampling_mask) if np.sum(row) == True])
    return sampling_mat

def vectorization(matrix):
    '''flattening of a 2d array'''
    n,m = np.shape(matrix)
    return np.array([matrix[i//m,i%m] for i in range(n*m)])

def matricization(tensor):
    '''2d matrix from a 4d tensor'''
    a,b,c,d = np.shape(tensor)
    return np.array([[tensor[j%a,j//a,i//d,i%d] for i in range(c*d)] for j in range(a*b)])

def cs_reconstruct_1d(sig_sampled,sampling_matrix,delta):
    l = np.shape(sampling_matrix)[1]
    sig_sampled = sig_sampled.timedom
    ift_matrix = np.fromfunction(lambda w, k: 1/l*np.exp(2*np.pi*1j/l*w*k),(l,l))
    x = cp.Variable(l, complex=True)
    objective = cp.Minimize(cp.norm(x,1))
    print(sig_sampled.shape, sampling_matrix.shape, ift_matrix.shape)
    constraints = [cp.abs(sampling_matrix@ift_matrix@x - sig_sampled) <= delta]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(verbose=True)
    return x.value

def cs_reconstruct_2d(sig_sampled,sampling_matricized,delta):
    l = int(np.sqrt(np.shape(sampling_matricized)[1]))
    print(l)
    sig_sampled = sig_sampled.timedom
    sig_sampled_vectorized = vectorization(sig_sampled)
    ift_tensor = np.fromfunction(lambda t1, t2, k1, k2: 1/(l**2)*np.exp(2*np.pi*1j/l*(t1*k1+t2*k2)),(l,l,l,l))
    # ift_tensor_matricized = np.fromfunction(lambda t, k: 1/(l**2)*np.exp(2*np.pi*1j/l*((t%l)*(k//l)+(t//l)*(k%l))),(l,l))
    ift_tensor_matricized = matricization(ift_tensor)
    # sampling_matricized = matricization(sampling)
    x = cp.Variable(l**2, complex=True)
    objective = cp.Minimize(cp.norm(x,1))
    constraints = [cp.abs(sampling_matricized@ift_tensor_matricized@x - sig_sampled_vectorized) <= delta]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(verbose=True)
    return x.value

# mask = np.random.choice([0,1],n,replace=True,p=(1-sampling_ratio,sampling_ratio))
# sampl_matrix = sampling_matrix(mask)

# test_sig = np.array([np.exp(2j*pi*0.4*t1-t1/(n/2))+np.exp(2j*pi*0.7*t1-t1/(n/2)) for t1 in range(n)])
# test_sig_full = Signal(test_sig)
# test_sig = test_sig[mask.astype(bool)]
# test_sig = Signal(test_sig)

# test_sig_full.plot(type="time")
# test_sig_full.plot(type="freq")
# test_sig.plot(type="time")
# test_sig.plot(type="freq")

# x = cs_reconstruct_1d(test_sig,sampl_matrix,0.1)

# rec_sig = Signal(np.fft.fft(np.flip(x)))
# rec_sig.plot(type="time")
# rec_sig.plot(type="freq")

### 2D

# def non_stationary_frequency(t1,t2,treal):
#     f1, f2 = 0.1+0.02*treal/n , 0.2 + 0.02*treal/n
#     tau1, tau2 = n/2, n/2
#     return np.exp(2j*pi*(f1*t1+f2*t2))

def non_stationary_frequency(t1,t2,treal):
    f1, f2 = 0.1, 0.2
    f3, f4 = 0.3, 0.6
    tau1, tau2 = n/2, n/2
    return np.exp(2j*pi*(f1*t1+f2*t2)-t1/tau1) + np.exp(2j*pi*(f3*t1+f4*t2)-t1/tau2)


sig = waveform2(n,non_stationary_frequency,range(n))
signal = Signal(sig)

signal.plot("freq")

sampling_rows = np.random.choice([0,1],n,replace=True,p=(1-sampling_ratio, sampling_ratio))
sampling_mask = np.array([np.ones(n) if ifrow else np.zeros(n) for ifrow in sampling_rows])

sampling_mat = sampling_matrix(vectorization(sampling_mask))

sig_sam = sig[sampling_mask.astype(bool)].reshape((np.count_nonzero(sampling_rows),n))
signal_sampled = Signal(sig_sam)

x = cs_reconstruct_2d(signal_sampled,sampling_mat,0.1)

signal_rec = Signal(np.fft.fft2(np.flip(x.reshape((n,n)),(0,1))))
signal_rec.plot("time")
signal_rec.plot("freq")