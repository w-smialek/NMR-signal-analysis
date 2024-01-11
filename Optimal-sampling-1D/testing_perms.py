import numpy as np
import matplotlib.pyplot as plt

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def exp_sig_sampled(amplitude,frequency,inverse_decay,t_sampling,tau_sampling):
    try:
        result = np.zeros(len(t_sampling))
        for a, f, dec in zip(amplitude,frequency, inverse_decay):
            result += np.array([a*np.cos(2*np.pi*f(tau)*t - dec*t) for t, tau in zip(t_sampling,tau_sampling)])
    except:
        result = np.array([amplitude*np.cos(2*np.pi*frequency(tau)*t - inverse_decay*t) for t, tau in zip(t_sampling,tau_sampling)])
    return result

def whitenoise(stdev,n):
    rng = np.random.default_rng()
    return rng.normal(0,stdev,n)

n = 500
sigma = 0.5

freq1 = lambda tau: 0.4-0.01*tau/n
# freq2 = lambda tau: 0.43-0.06*tau/n
# freq3 = lambda tau: 0.37-0.06*tau/n


# form = exp_sig_sampled((1,1,1), (freq1,freq2,freq3), (0,0,0), range(-n//2,n//2), range(-n//2,n//2)) + whitenoise(sigma,n)
form = exp_sig_sampled(1, freq1, 0, range(-n//2,n//2), range(-n//2,n//2)) + whitenoise(sigma,n)
formft = np.fft.fft(form).real

max_control = max(abs(formft))
maxarg_control = min(np.argmax(abs(formft))/n, 1- np.argmax(abs(formft))/n)
plt.plot(abs(formft))
plt.show()

###
###
###

tau_sampling = []

n_pieces = 8
    
pieces_forw = [list(i) for i in list(split(range(n), n_pieces))]

sampling_forw = []
for p in [7,5,6,3,4,1,2,0]:
    sampling_forw += pieces_forw[p]

plt.plot(sampling_forw)
plt.show()

# form = exp_sig_sampled((1,1,1), (freq1,freq2,freq3), (0,0,0), range(-n//2,n//2), sampling_forw) + whitenoise(sigma,n)
form = exp_sig_sampled(1, freq1, 0, range(-n//2,n//2), sampling_forw) + whitenoise(sigma,n)
formft = np.fft.fft(form).real

max_sampling = max(abs(formft))
maxarg_sampling = min(np.argmax(abs(formft))/n, 1- np.argmax(abs(formft))/n)
plt.plot(abs(formft))
plt.show()

print(max_control, max_sampling)
print(maxarg_control, maxarg_sampling)