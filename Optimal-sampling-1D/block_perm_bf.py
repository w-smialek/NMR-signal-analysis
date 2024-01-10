import itertools
import numpy as np
import matplotlib.pyplot as plt

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

n_pieces = 8
pieces = []

for i in range(n_pieces):
    pieces.append([i*(n//n_pieces)+j for j in range(n//n_pieces)])

maxes = []

for perm in itertools.permutations(range(n_pieces)):
    sampling = []
    for p in perm:
        sampling += pieces[p]
    # plt.plot(sampling)
    # plt.show()

    form = cos_sig_sampled(1, freq, 0, range(-n//2,n//2), sampling)
    formft = np.fft.fft(form).real
    maxes.append((max(abs(formft)), np.argmax(abs(formft))/n, perm))
    # plt.plot(formft)
    # plt.show()

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
for p in mperm:
    sampling += pieces[p]
print(mperm)
plt.plot(sampling)
plt.show()