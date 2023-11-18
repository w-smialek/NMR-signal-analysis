import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
import imageio

import signals as sig

##
## NUS - non-stationarity compromise
##

n = 40

## 1D

frate = 0.01

def waveform(F, t_real, t_sampled):
    return [F(t_r,t_s) for t_r, t_s in zip(t_real, t_sampled)]

def non_stat_freq(t_r, t_s):
    a1, a2 = 2, 1
    f1, f2 = 0.2 + frate*t_r/n, 0.5 + frate*t_r/n
    tau1, tau2 = n/3, n/3
    return a1*np.exp(2j*pi*t_s*f1 -t_s/tau1) + a2*np.exp(2j*pi*t_s*f2 -t_s/tau2)

plt.style.use("ggplot")

sig = sig.Signal(waveform(non_stat_freq,range(n),range(n)))
sig.add(sig.whitenoise_complex(0.3,n))
sig.dt1 = 1/n
# sig.plot(type="time")
sig.plot(type="freq", linewidth = 0.8)
plt.savefig("1d_compromise_%s_orig.png"%int(100*frate),dpi=300)
plt.close()

for i in np.linspace(0.1,1,10):
    sampling_ratio = i
    mask = np.zeros(n)
    filter = np.random.choice(np.arange(n),int(sampling_ratio*n),replace=False)
    mask[filter] = 1

    form = waveform(non_stat_freq, range(int(n*sampling_ratio)), filter)

    sig_sampled = sig.Signal(form)
    sig_sampled.deshuffle(filter)
    # sig_sampled.plot(type="time")
    # sig_sampled.plot(type="freq")
    # plt.show()
    
    a = np.fft.ifft(sig.cs_reconstruct_1d(sig_sampled, sig.sampling_matrix(mask), 0.5))

    if i ==0.99:
        sig_rec = sig.Signal(a)
    else:
        sig_rec = sig.Signal(a)
    # sig_rec.plot(type="time")
    plt.style.use("ggplot")
    sig_rec.plot(type="freq", linewidth = 0.8)
    # plt.plot(sig_rec.freqdom)
    plt.savefig("1d_compromise_%s_%s.png"%(int(100*frate),(int(i*100))),dpi=300)
    plt.close()

filenames = ["1d_compromise_%s_%s.png"%(int(100*frate),(int(i*100))) for i in np.linspace(0.1,1,10)]

with imageio.get_writer("1d_compromise_%s.gif"%int(100*frate), mode='I',duration=1) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)


## 2D

frate = 0.02

def non_stationary_frequency(t1,t2,treal):
    f1, f2, f3, f4 = 0.2+frate*treal/n, 0.4+frate*treal/n, 0.6+frate*treal/n, 0.3+frate*treal/n
    tau1, tau2 = n, n
    return 2*np.exp(2j*pi*(f1*t1+f2*t2)-t1/tau1) + np.exp(2j*pi*(f3*t1+f4*t2)-t1/tau2)#-t1/(tau1)-t2/(tau2))

def snapshot_2d(n, F, t_real, t_indir):
    '''n - number of direct time datapoints \n
    treal - range of real time values \n
    t_indir - corresponding indirect time values'''
    form = np.zeros((n,n)).astype(complex)
    for t2, tr in zip(t_indir, t_real):
        form[t2,:] = np.array([F(t1,t2,tr) for t1 in range(n)])
    return form[t_indir]

sig_original = sig.Signal(snapshot_2d(n,non_stationary_frequency,range(n),range(n)))
sig_original.add(sig.whitenoise_complex(0.3,(n,n)))
# sig_original.plot()
sig_original.plot("freq")
# plt.matshow(sig_original.freqdom.real)
plt.savefig("2d_compromise_%s_orig.png"%int(frate*100),dpi=300)
plt.close()

for i in np.linspace(0.1,1,10):
    t_r_schedule = np.random.choice(np.arange(int(n*i)),int(n*i),replace=False)

    filter = np.random.choice(np.arange(n),int(n*i),replace=False)
    mask = np.zeros((n,n))
    mask[filter] = np.ones(n)
    sampling_mat = sig.sampling_matrix(sig.vectorization(mask))

    sig_padded = np.zeros((n,n)).astype(complex)
    for t2, treal in zip(filter, t_r_schedule):
        sig_padded[t2,:] = [non_stationary_frequency(t1,t2,treal) for t1 in range(n)]

    sig_subsampled = sig_padded[mask.astype(bool)].reshape((filter.size,n))

    sig_padded = sig.Signal(sig_padded)
    sig_subsampled = sig.Signal(sig_subsampled)
    # sig_padded.plot()
    # sig_subsampled.plot()

    sig_reconstructed = sig.cs_reconstruct_2d(sig_subsampled,sampling_mat,0.1*i)
    sig_reconstructed = sig.Signal(np.fft.ifft2(sig_reconstructed.reshape((n,n)).T))

    sig_reconstructed.plot("freq")
    # plt.matshow(sig_reconstructed.freqdom.real)
    plt.savefig("2d_compromise_%s_%s.png"%(int(100*frate),int(i*100)),dpi=300)

filenames = ["2d_compromise_%s_%s.png"%(int(100*frate),(int(i*100))) for i in np.linspace(0.1,1,10)]

with imageio.get_writer("2d_compromise_%s.gif"%int(100*frate), mode='I',duration=1) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)


n_snapshots = 10
t_r_length = 5

sampling_ratio = t_r_length/n

for i in range(n_snapshots):
    t_r_schedule = np.random.choice(np.arange(i*t_r_length,(i+1)*t_r_length),t_r_length,replace=False)

    filter = np.random.choice(np.arange(n),int(n*sampling_ratio),replace=False)
    mask = np.zeros((n,n))
    mask[filter] = np.ones(n)
    sampling_mat = sig.sampling_matrix(sig.vectorization(mask))

    sig_padded = np.zeros((n,n)).astype(complex)
    for t2, treal in zip(filter, t_r_schedule):
        sig_padded[t2,:] = [non_stationary_frequency(t1,t2,treal) for t1 in range(n)]

    sig_subsampled = sig_padded[mask.astype(bool)].reshape((filter.size,n))

    sig_padded = sig.Signal(sig_padded)
    sig_subsampled = sig.Signal(sig_subsampled)
    sig_padded.plot()
    sig_subsampled.plot()

    sig_reconstructed = sig.cs_reconstruct_2d(sig_subsampled,sampling_mat,0.1)
    sig_reconstructed = sig.Signal(np.fft.ifft2(sig_reconstructed.reshape((n,n)).T))

    sig_reconstructed.plot("freq")
