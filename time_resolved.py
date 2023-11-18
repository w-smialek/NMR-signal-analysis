import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
import imageio

import signals as sig

###
### 2D time-resolved NUS with CS reconstruction
###

n = 40

def non_stationary_frequency(t1,t2,treal):
    f1, f2 = 0.2+0.025*treal/n, 0.4+0.015*treal/n
    tau1, tau2 = n/2, n/2
    return np.exp(2j*pi*(f1*t1+f2*t2))#-t1/(tau1)-t2/(tau2))

def snapshot_2d(n, F, t_real, t_indir):
    '''n - number of direct time datapoints \n
    treal - range of real time values \n
    t_indir - corresponding indirect time values'''
    form = np.zeros((n,n)).astype(complex)
    for t2, tr in zip(t_indir, t_real):
        form[t2,:] = np.array([F(t1,t2,tr) for t1 in range(n)])
    return form[t_indir]

sampling_ratio = 0.2
t_r_length = int(sampling_ratio*n)

n_samples = 30

for i in range(n_samples):
    t_r_schedule = np.random.choice(np.arange(int(t_r_length*i),int(t_r_length*(i+1))),t_r_length,replace=False)
    filter = np.random.choice(np.arange(n),t_r_length,replace=False)
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

    sig_reconstructed = sig.cs_reconstruct_2d(sig_subsampled,sampling_mat,0.6)
    sig_reconstructed = sig.Signal(np.fft.ifft2(sig_reconstructed.reshape((n,n)).T))

    sig_reconstructed.plot("freq")
    # plt.matshow(sig_reconstructed.freqdom.real)
    plt.savefig("2d_TS1_%s.png"%(int(i*100)),dpi=300)
    plt.close()


def average_over_neighbours(array):
    ars = [np.roll(array,t,(0,1)) for t in [(-1,0),(0,-1),(0,0),(0,1),(1,0),(-1,-1),(1,1),(-1,1),(1,-1)]]
    return sum(ars)

n_snapshots = 10
treal_interval = 10

snaphot_sampling_ratio = treal_interval/n

snapshots = []
masks = []
for i in range(n_snapshots):
    t_real_interval = (int(treal_interval*i),int(treal_interval*(i+1)))
    t_indir = np.random.choice(n,int(n*snaphot_sampling_ratio),replace=False)
    ss = sig.Signal(snapshot_2d(n,non_stationary_frequency,range(*t_real_interval),t_indir))
    ss.deshuffle(t_indir)
    mask = np.array([np.ones(n) if (i in t_indir) else np.zeros(n) for i in range(n)])
    masks.append(mask)
    snapshots.append(ss)

maxes = []
recs = []

for snap, mask in zip(snapshots,masks):
    snap.plot(type="freq")
    x = sig.cs_reconstruct_2d(snap,sig.sampling_matrix(sig.vectorization(mask)),1)
    rec = sig.Signal(np.fft.ifft2(x.reshape((n,n))).T)
    rec.plot("freq")
    recs.append(rec)
    maxes.append(np.argmax(rec.freqdom))

mm = np.zeros((n,n))
for m in maxes:
    mm[(m//n,m%n)] = 1

plt.imshow(mm)
plt.show()

# filenames = ["2d_TS1_%s.png"%(int(i*100)) for i in range(n_samples)]

# with imageio.get_writer("2d_TS1.gif", mode='I',duration=0.5) as writer:
#     for filename in filenames:
#         image = imageio.imread(filename)
#         writer.append_data(image)