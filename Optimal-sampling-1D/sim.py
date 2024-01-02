import signals as sig
import numpy as np
import matplotlib.pyplot as plt

def exp_sig(amplitude,frequency,inverse_decay,n):
    return np.array([amplitude*np.exp(2j*np.pi*frequency(t)*t - inverse_decay*t) for t in range(n)])

def exp_sig_sampled(amplitude,frequency,inverse_decay,n,sampling):
    return np.array([amplitude*np.exp(2j*np.pi*frequency(tau)*t - inverse_decay*t) for t, tau in zip(range(n),sampling)])


n = 200


### Brute force check
max = 0.0
sampling_max = []
spectrum_max = []

for i in range(10000):
    sampling = np.random.choice(np.arange(n),n,replace=False)

    wave = exp_sig_sampled(1,lambda t: 0.2 + 0.08*t/n,0,n,sampling)
    spectrum = np.fft.fft(wave)
    max_current = np.max(spectrum.real)
    if max_current > max:
        max = max_current
        sampling_max = sampling
        spectrum_max = spectrum
    if i % 100 == 0:
        print(i)

print(sampling_max)
print(max)
# plt.plot(wave)
# plt.show()
plt.plot(spectrum_max)
plt.show()

### Linear
sampling = np.arange(n)
wave = exp_sig_sampled(1,lambda t: 0.2 + 0.08*t/n,0,n,sampling)
spectrum = np.fft.fft(wave)
print(np.max(spectrum.real))
plt.plot(spectrum)
plt.show()

np.save("arrmax2.npy",sampling_max)

### Check brute force result

sampling = np.arange(n)
wave = exp_sig_sampled(1,lambda t: 0.5 + 0.081*t/n,0,n,sampling)
spectrum = np.fft.fft(wave)
print(np.max(spectrum.real))
plt.plot(spectrum)
plt.show()

sampling_max = np.load("arrmax2.npy")

wave = exp_sig_sampled(1,lambda t: 0.5 + 0.081*t/n,0,n,sampling_max)
spectrum = np.fft.fft(wave)
print(np.max(spectrum.real))
plt.plot(spectrum)
plt.show()