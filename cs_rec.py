import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
import matplotlib as mpl

import signals as sig

###
### CS reconstruction
###

n = 40

### 1D

newcolors = np.array([[1,0,0,1-np.exp(-i/150)] for i in range(256)])
newcmp = mpl.colors.ListedColormap(newcolors)


sampling_ratio = 0.1

mask = np.random.choice([0,1],n,replace=True,p=(1-sampling_ratio,sampling_ratio))
sampl_matrix = sig.sampling_matrix(mask)

test_sig = np.array([np.exp(2j*pi*0.4*t1-t1/(n/2))+np.exp(2j*pi*0.7*t1-t1/(n/2)) for t1 in range(n)])
test_sig += sig.whitenoise_complex(0.3,n)
test_sig_full = sig.Signal(test_sig)
test_sig = test_sig[mask.astype(bool)]
test_sig = sig.Signal(test_sig)

plt.style.use("ggplot")
# test_sig_full.plot(type="time")
test_sig_full.plot(type="freq", linewidth = 0.8)
plt.savefig("1d_rec_orig.png",dpi=300)
plt.show()
# test_sig.plot(type="time")
test_sig.plot(type="freq")
plt.show()


x = sig.cs_reconstruct_1d(test_sig,sampl_matrix,0.1)

rec_sig = sig.Signal(np.fft.ifft(x))
# rec_sig.plot(type="time")
rec_sig.plot(type="freq", linewidth = 0.8)
plt.savefig("1d_rec_rec.png",dpi=300)
plt.show()


## 2D

sampling_ratio = 0.2

# def non_stationary_frequency(t1,t2,treal):
#     f1, f2 = 0.1+0.02*treal/n , 0.2 + 0.02*treal/n
#     tau1, tau2 = n/2, n/2
#     return np.exp(2j*pi*(f1*t1+f2*t2)-t1/(tau1)-t2/(tau2))

def non_stationary_frequency(t1,t2,treal):
    f1, f2 = 0.1, 0.2
    f3, f4 = 0.3, 0.6
    f5, f6 = 0.5, 0.5
    tau1, tau2 = n/2, n/2
    return np.exp(2j*pi*(f1*t1+f2*t2)-t1/tau1) + 0.5*np.exp(2j*pi*(f3*t1+f4*t2)-t1/tau2) + 0.25*np.exp(2j*pi*(f5*t1+f6*t2)-t1/tau2)


sig0 = sig.waveform2(n,non_stationary_frequency,range(n))
signal0 = sig.Signal(sig0)
signal0.add(sig.whitenoise_complex(0.2,(n,n)))
signal0.dt1 = signal0.dt2 = 1/n

signal0.plot("freq",cmap=newcmp)
plt.savefig("2d_rec_orig.png",dpi=300)
plt.show()

sampling_rows = np.random.choice([0,1],n,replace=True,p=(1-sampling_ratio, sampling_ratio))
sampling_mask = np.array([np.ones(n) if ifrow else np.zeros(n) for ifrow in sampling_rows])

sampling_mat = sig.sampling_matrix(sig.vectorization(sampling_mask))

sig_sam = sig0[sampling_mask.astype(bool)].reshape((np.count_nonzero(sampling_rows),n))
signal_sampled = sig.Signal(sig_sam)

x = sig.cs_reconstruct_2d(signal_sampled,sampling_mat,0.1)

signal_rec = sig.Signal(np.fft.ifft2(x.reshape((n,n)).T))
signal_rec.dt1 = signal_rec.dt2 = 1/n
signal_rec.plot("freq",cmap=newcmp)
plt.savefig("2d_rec_rec.png",dpi=300)
plt.show()
