import numpy as np
import matplotlib.pyplot as plt

def exp_sig_sampled(amplitude,frequency,inverse_decay,t_sampling,tau_sampling):
    return np.array([amplitude*np.cos(2*np.pi*frequency(tau)*t - inverse_decay*t) for t, tau in zip(t_sampling,tau_sampling)])

def whitenoise(stdev,n):
    rng = np.random.default_rng()
    return rng.normal(0,stdev,n)

n = 500
sigma = 0.3

freq = lambda tau: 0.4-0.01*tau/n

form = exp_sig_sampled(1, freq, 0, range(-n//2,n//2), range(-n//2,n//2)) + whitenoise(sigma,n)
formft = np.fft.fft(form).real

max_control = max(abs(formft))
maxarg_control = min(np.argmax(abs(formft))/n, 1- np.argmax(abs(formft))/n)
plt.plot(formft)
plt.show()

###
###
###
tau_sampling = []

h = 50
for i in range(n):
    if (i//h)%2 == 0:
        tau_sampling.append((i//h)/2*h+i%h)
    else:
        tau_sampling.append(n//2 + ((i//h)-1)/2*h+i%h)
tau_sampling = np.array(tau_sampling) - n//2

form = exp_sig_sampled(1, freq, 0, range(-n//2,n//2), tau_sampling) + whitenoise(sigma,n)
formft = np.fft.fft(form).real

plt.plot(formft)
plt.show()

maxes_maxes = []

for fbar in [0.05]:#np.linspace(0,0.2,400):

    freq1 = lambda tau: 0.2-fbar*tau/n
    freq2 = lambda tau: 0.25-fbar*tau/n


    maxes = []
    for ind in range(1,n):
        tau_sampling = []
        h = 140
        for i in range(n):
            if (i//h)%2 == 0:
                tau_sampling.append((i//h)/2*h+i%h)
            else:
                tau_sampling.append(n//2 + ((i//h)-1)/2*h+i%h)
        tau_sampling = np.array(tau_sampling) - n//2
        # tau_sampling = np.random.choice(np.arange(n),n,replace=False)

        form = exp_sig_sampled(1, freq1, 0, range(-n//2,n//2), tau_sampling) + exp_sig_sampled(1, freq2, 0, range(-n//2,n//2), tau_sampling) + whitenoise(sigma,n)
        form2 = exp_sig_sampled(1, freq1, 0, range(-n//2,n//2), range(-n//2,n//2)) + exp_sig_sampled(1, freq2, 0, range(-n//2,n//2), range(-n//2,n//2)) + whitenoise(sigma,n)
        formft = np.fft.fft(form).real
        formft2 = np.fft.fft(form2).real

        plt.plot(abs(formft))
        plt.show()
        plt.plot(abs(formft2))
        plt.show()

        maxes.append((max(abs(formft)), np.argmax(abs(formft))/n))

    maxes1 = [m[0] for m in maxes]
    maxes2 = [min(m[1],1-m[1]) for m in maxes]

    maxes_maxes.append(np.argmax(maxes1))

    # plt.ylim(0, n)
    plt.plot(maxes1)
    plt.plot([max_control for ind in range(1,n)])
    plt.show()
    plt.close()
    # plt.ylim(0, 1)
    plt.plot(maxes2)
    plt.plot([maxarg_control for ind in range(1,n)])
    plt.show()

plt.plot(maxes_maxes)
plt.show()