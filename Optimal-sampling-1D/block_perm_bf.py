import itertools
import numpy as np
import matplotlib.pyplot as plt
import math

def cos_sig_sampled(amplitude,frequency,inverse_decay,t_sampling,tau_sampling):
    return np.array([amplitude*np.cos(2*np.pi*frequency(tau)*t - inverse_decay*t) for t, tau in zip(t_sampling,tau_sampling)])

def whitenoise(stdev,n):
    rng = np.random.default_rng()
    return rng.normal(0,stdev,n)

n = 500
sigma = 0.3
f = 0.4
fbar = 0.03
# fbar = 0.04

freq = lambda tau: f-fbar*tau/n

n_pieces = 4
pieces_forw = []
pieces_backw = []

for i in range(n_pieces):
    pieces_forw.append([i*(n//n_pieces)+j for j in range(n//n_pieces)])
    pieces_backw.append([i*(n//n_pieces)+j for j in range(n//n_pieces,0,-1)])


maxes = []

i = 0
for perm in itertools.permutations(range(n_pieces)):
    sampling_forw = []
    for p in perm:
        sampling_forw += pieces_forw[p]
    # plt.plot(sampling)
    # plt.show()

    form = cos_sig_sampled(1, freq, 0, range(-n//2,n//2), sampling_forw)
    formft = np.fft.fft(form).real
    maxes.append((max(abs(formft)), np.argmax(abs(formft))/n, perm))
    # plt.plot(formft)
    # plt.show()
    i += 1
    if (i % 1000) == 0:
        print(i)

i = 0
for perm in itertools.permutations(range(n_pieces)):
    sampling_backw = []
    for p in perm:
        sampling_backw += pieces_backw[p]
    # plt.plot(sampling)
    # plt.show()

    form = cos_sig_sampled(1, freq, 0, range(-n//2,n//2), sampling_backw)
    formft = np.fft.fft(form).real
    maxes.append((max(abs(formft)), np.argmax(abs(formft))/n, perm))
    # plt.plot(formft)
    # plt.show()
    i += 1
    if (i % 1000) == 0:
        print(i)


maxes1 = [m[0] for m in maxes]
maxes2 = [min(m[1],1-m[1]) for m in maxes]
perms = [m[2] for m in maxes]

plt.plot(maxes1)
plt.show()
plt.close()
plt.ylim(0, 1)
plt.plot(maxes2)
plt.show()

mmax = np.argmax(maxes1)
mperm = perms[mmax]

sampling = []

print(mmax)
print(math.factorial(n_pieces))

if mmax > math.factorial(n_pieces):
    for p in mperm:
        sampling += pieces_backw[p]
else:
    for p in mperm:
        sampling += pieces_forw[p]

print(mperm)
plt.plot(sampling)
plt.show()