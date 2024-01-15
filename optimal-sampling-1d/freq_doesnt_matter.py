import numpy as np
import matplotlib.pyplot as plt
import itertools
import math

### Functions

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

n = 500

def signal(f, fbar, dec, sampling):
    return [np.exp(2j*np.pi*(f + fbar * tau)*t)*np.exp(-t*dec) for t, tau in zip(range(n),sampling)]

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

def fitness_function(sampling, f, fbar, d_fbar, dec):
    fit = 0
    for k in np.linspace(-5,5,20):
        sig = signal(f,fbar+k*d_fbar,dec,sampling)
        sig_ft = abs(np.fft.fft(sig))
        fit += max(abs(sig_ft))*np.exp(-alpha*abs(k))
    return fit

### Parameters

f = 0.2
fbar = 0.02/n
dec = 0

n_pieces = 4
delta_f = fbar/20#0.003/n
alpha = 0

### Loop

perms = itertools.permutations(range(n_pieces))

max_all = 0
sampling_max = None
indx = 0

maxes = []

for p in perms:
    for s in itertools.product([True,False], repeat=n_pieces):
        sampling = block_perm_sampling(n_pieces,p,s)
        fit_current = fitness_function(sampling,f,fbar,delta_f,dec)
        maxes.append(fit_current)
        if fit_current > max_all:
            print(fit_current)
            max_all = fit_current
            sampling_max = sampling
    indx+=1
    percent = int(indx/(math.factorial(n_pieces))*100)
    if percent%10 <= 2:
        print(percent)

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
sig = signal(f,fbar,dec,range(n))
sig_ft = abs(np.fft.fft(sig))
plt.plot(sig)
plt.show()
plt.plot(sig_ft)
plt.show()

sig = signal(f,fbar,dec,sampling_max)
sig_ft = abs(np.fft.fft(sig))
plt.plot(sig)
plt.show()
plt.plot(sig_ft)
plt.show()

###