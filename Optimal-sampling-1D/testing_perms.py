import numpy as np
import matplotlib.pyplot as plt
import os

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def exp_sig_sampled(amplitude,frequency,inverse_decay,t_sampling,tau_sampling):
    try:
        result = np.zeros(len(t_sampling))
        for a, f, dec in zip(amplitude,frequency, inverse_decay):
            result += np.array([a*np.cos(2*np.pi*f(tau)*t)*np.exp(-dec*t) for t, tau in zip(t_sampling,tau_sampling)])
    except:
        result = np.array([amplitude*np.cos(2*np.pi*frequency(tau)*t)*np.exp(-inverse_decay*t) for t, tau in zip(t_sampling,tau_sampling)])
    return result

def whitenoise(stdev,n):
    rng = np.random.default_rng()
    return rng.normal(0,stdev,n)

n = 500
sigma = 0.5

freq1 = lambda tau: 0.40-0*tau/n
freq2 = lambda tau: 0.30-0.01*tau/n
freq3 = lambda tau: 0.34-0.006*tau/n

amplitude_used = (1,1)
freq_used = (freq1,freq2)
dec_used = (0,0)

amplitude_used = 1
freq_used = freq1
dec_used = 3/n

# form = exp_sig_sampled(amplitude_used, freq_used, dec_used, range(-n//2,n//2), range(-n//2,n//2)) + whitenoise(sigma,n)
form = exp_sig_sampled(amplitude_used, freq_used, dec_used, range(-n//2,n//2), range(0,n))# + whitenoise(sigma,n)
formft = np.fft.fft(form).real

max_control = max(abs(formft))
maxarg_control = min(np.argmax(abs(formft))/n, 1- np.argmax(abs(formft))/n)
plt.plot(form)
plt.show()
plt.plot(abs(formft))
# plt.ylim(0,300)
# plt.savefig("./NMR-signal-analysis/optimal-sampling-1d/control.png",dpi=300)
plt.show()

###
###
###

tau_sampling = []


def forw_sampling(n_pieces, n_cycle): 
    pieces_forw = [list(i) for i in list(split(range(n), n_pieces))]
    sampling_forw = []
    perm = np.array([j for j in range(n_pieces)])
    cycle = np.array([(j+1)%n_pieces for j in range(n_pieces)])
    for a in range(n_cycle):
        perm = perm[cycle]
    for p in perm:
        sampling_forw += pieces_forw[p]
    return sampling_forw

def backw_sampling(n_pieces, n_cycle): 
    pieces_backw = [list(i) for i in list(split(range(n-1,-1,-1), n_pieces))]
    sampling_backw = []
    # perm = np.array([(j+1)%n_pieces for j in range(n_pieces)])
    # cycle = np.arange(n_pieces-1,-1,-1)
    perm = np.array([j for j in range(n_pieces)])
    cycle = np.array([(j-1)%n_pieces for j in range(n_pieces)])
    for a in range(n_cycle):
        perm = perm[cycle]
    for p in perm:
        sampling_backw += pieces_backw[p]
    return sampling_backw

n_min = 1
n_tot = 100

maxes = np.zeros((n_tot,n_tot*2))
maxargs = np.zeros((n_tot,n_tot*2))

for ni in range(n_min,n_tot):
    for nc in range(0,ni):
        sampling_forw = forw_sampling(ni,nc)
        # form = exp_sig_sampled(amplitude_used, freq_used, dec_used, range(0,n), sampling_forw) + whitenoise(sigma,n)
        form = exp_sig_sampled(amplitude_used, freq_used, dec_used, range(0,n), sampling_forw)# + whitenoise(sigma,n)
        formft = np.fft.fft(form).real
        maxes[ni,n_tot+1+nc] = max(abs(formft))
        maxargs[ni,n_tot+1+nc] = min(np.argmax(abs(formft))/n, 1- np.argmax(abs(formft))/n)
# for ni in range(n_min,n_tot):
#     for nc in range(0,ni):
#         sampling_backw = backw_sampling(ni,nc)
#         # form = exp_sig_sampled(amplitude_used, freq_used, dec_used, range(0,n), sampling_forw) + whitenoise(sigma,n)
#         form = exp_sig_sampled(amplitude_used, freq_used, dec_used, range(0,n), sampling_backw)# + whitenoise(sigma,n)
#         formft = np.fft.fft(form).real
#         maxes[ni,n_tot-1-nc] = max(abs(formft))
#         maxargs[ni,n_tot-1-nc] = min(np.argmax(abs(formft))/n, 1- np.argmax(abs(formft))/n)

plt.matshow(maxes)
plt.show()

n_p = 80
plt.plot(maxes[n_p,n_tot+1:n_tot+n_p+1])
plt.show()

# plt.matshow(maxargs)
# plt.show()

ni_max = np.unravel_index(np.argmax(maxes),(n_tot,n_tot*2)) - np.array([0,100])
print(ni_max)

# ni_max = np.array([6,-1])
if ni_max[1] > 0:
    sampling_forw = forw_sampling(ni_max[0],ni_max[1]-1)
else:
    sampling_forw = backw_sampling(ni_max[0],-ni_max[1]-1)
form = exp_sig_sampled(amplitude_used, freq_used, dec_used, range(0,n), sampling_forw)# + whitenoise(sigma,n)
formft = np.fft.fft(form).real
print(max_control,max(abs(formft)))

plt.plot(sampling_forw)
plt.show()
plt.plot(abs(formft))
plt.ylim(0,300)
# plt.savefig("./NMR-signal-analysis/optimal-sampling-1d/sampling.png",dpi=300)
plt.show()

###
### bruteforce
###

'''n_trials = 50000

max_all = 0
maxarg_all = 0
maxsampling_all = np.random.choice(np.arange(n),n,replace=False)

for i in range(n_trials):
    sampling_current = np.random.choice(np.arange(n),n,replace=False)
    form = exp_sig_sampled(amplitude_used, freq_used, dec_used, range(0,n), sampling_current)# + whitenoise(sigma,n)
    formft = np.fft.fft(form).real
    max_current = max(abs(formft))
    if max_current > max_all:
        max_all = max_current
        maxarg_all = min(np.argmax(abs(formft))/n, 1- np.argmax(abs(formft))/n)
        maxsampling_all = sampling_current
    if i % 10000 == 0:
        print(i)

plt.plot(maxsampling_all)
plt.show()
print(max_control, max_all, maxarg_all)
form = exp_sig_sampled(amplitude_used, freq_used, dec_used, range(0,n), maxsampling_all)# + whitenoise(sigma,n)
formft = np.fft.fft(form).real
plt.plot(formft)
plt.show()'''

###
### bruteforce with forw pieces
###

'''n_pieces = 10
n_trials = 100000

def random_forw_sampling(n_pieces): 
    pieces_forw = [list(i) for i in list(split(range(n), n_pieces))]
    sampling_forw = []
    perm = np.random.choice(n_pieces,n_pieces,replace=False)# np.arange(n_pieces-1,-1,-1)
    for p in perm:
        sampling_forw += pieces_forw[p]
    return sampling_forw

max_all = 0
maxarg_all = 0
maxsampling_all = np.random.choice(np.arange(n),n,replace=False)

for i in range(n_trials):
    sampling_current = random_forw_sampling(n_pieces)
    form = exp_sig_sampled(amplitude_used, freq_used, dec_used, range(0,n), sampling_current)# + whitenoise(sigma,n)
    formft = np.fft.fft(form).real
    max_current = max(abs(formft))
    if max_current > max_all:
        max_all = max_current
        maxarg_all = min(np.argmax(abs(formft))/n, 1- np.argmax(abs(formft))/n)
        maxsampling_all = sampling_current
    if i % 1000 == 0:
        print(i)

plt.plot(maxsampling_all)
plt.show()
form = exp_sig_sampled(amplitude_used, freq_used, dec_used, range(0,n), maxsampling_all)# + whitenoise(sigma,n)
formft = np.fft.fft(form).real
print(max_control, max_all, maxarg_all)
plt.plot(abs(formft))
plt.show()'''

###
### cycles
###

# maxes = []
# maxargs = []

# for nc in range(n):
#     sampling_forw = forw_sampling(n,nc)
#     # form = exp_sig_sampled(amplitude_used, freq_used, dec_used, range(0,n), sampling_forw) + whitenoise(sigma,n)
#     form = exp_sig_sampled(amplitude_used, freq_used, dec_used, range(0,n), sampling_forw)# + whitenoise(sigma,n)
#     formft = np.fft.fft(form).real
#     maxes.append(max(abs(formft)))
#     maxargs.append(min(np.argmax(abs(formft))/n, 1- np.argmax(abs(formft))/n))
#     # max_sampling = max(abs(formft))
#     # maxarg_sampling = min(np.argmax(abs(formft))/n, 1- np.argmax(abs(formft))/n)

# plt.plot(maxes)
# plt.plot(np.array(maxargs)*500)
# plt.show()

# ni_max = np.argmax(maxes)

# sampling_forw = forw_sampling(n,ni_max)
# form = exp_sig_sampled(amplitude_used, freq_used, dec_used, range(0,n), sampling_forw)# + whitenoise(sigma,n)
# formft = np.fft.fft(form).real
# print(max_control,max(abs(formft)))

# plt.plot(sampling_forw)
# plt.show()
# plt.plot(abs(formft))
# plt.show()