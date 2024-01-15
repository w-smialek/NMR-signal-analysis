import numpy as np
import matplotlib.pyplot as plt
import itertools

### Functions

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

n = 500

def signal(f, fbar, dec, sampling):
    return [np.cos(2*np.pi*(f + fbar * tau)*t)*np.exp(-t*dec) for t, tau in zip(range(n),sampling)]

def block_perm_sampling(n_pieces, perm):
    pieces = [list(i) for i in list(split(range(n), n_pieces))]
    sampling = []
    for p in perm:
        sampling += pieces[p]
    return sampling

def fitness_function(sampling, f, fbar, d_fbar, dec):
    fit = 0
    for k in np.linspace(-5,5,20):
        sig = signal(f,fbar+k*d_fbar,dec,sampling)
        sig_ft = abs(np.fft.fft(sig))
        fit += max(abs(sig_ft))*np.exp(-alpha*abs(k))
    return fit

### Parameters

f = 0.15
fbar = 0.001/n
dec = 0

n_pieces = 6
delta_f = 0.0002/n
alpha = 0

### Loop

perms = itertools.permutations(range(n_pieces))

max_all = 0
sampling_max = None
indx = 0

maxes = []

for p in perms:
    sampling = block_perm_sampling(n_pieces,p)
    fit_current = fitness_function(sampling,f,fbar,delta_f,dec)
    maxes.append(fit_current)
    indx+=1
    if fit_current > max_all:
        print(fit_current, indx)
        max_all = fit_current
        sampling_max = sampling

### Plots

# Best sampling
plt.plot(sampling_max)
plt.show()

# Trivial sampling fitness
max_control = fitness_function(range(n),f,fbar,delta_f,dec)
print(max_control)

# Fitness of all permutations and reference trivial sampling
plt.plot(maxes)
plt.plot([max_control for m in maxes])
plt.show()

# Trivial and optimal sampling spectra
sig_ft = abs(np.fft.fft(signal(f,fbar,dec,range(n))))
plt.plot(sig_ft)
plt.show()

sig_ft = abs(np.fft.fft(signal(f,fbar,dec,sampling_max)))
plt.plot(sig_ft)
plt.show()

###