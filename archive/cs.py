import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib as mplt

# w = 10
# k = 20

# A = np.random.randn(w,k)
# b = np.random.random(w)

# x = cp.Variable(k)
# objective = cp.Minimize(cp.sum_squares(A @ x - b))
# constraints = [0 <= x, x <= 1, x[0] == x[-1]]
# prob = cp.Problem(objective,constraints)

# result = prob.solve()

# print(x.value)

# plt.matshow([x.value])
# plt.show()


def sinexp(x,ampl,freq,tau):

    return ampl*np.exp(x*(2*np.pi*freq*1j - 1/tau))

n = 100
sampling = 0.06

signal = np.zeros(n).astype(complex)
ampls = (1,3,2)
freqs = (0.1,0.3,0.2)
taus = (45,30,56)
for (a,f,t) in zip(ampls,freqs,taus):
    signal += np.array([sinexp(x,a,f,t) for x in range(n)])

# plt.plot(signal.real)
# plt.show()

sampling = np.array([[1 if i==int(j**1.47) else 0 for i in range(n)] for j in range(int(sampling*n))])
# plt.matshow(sampling)
# plt.show()

signal_sampled = np.matmul(sampling,signal)
# plt.plot(signal_sampled.real)
# plt.show()

signal_sampled_padded = np.array([item if np.sum(sampling,0)[i] == 1 else 0 for i, item in enumerate(signal)])
# plt.plot(signal_sampled_padded.real)
# plt.show()

# plt.plot(np.fft.fft(signal).real)
# plt.plot(np.fft.fft(signal_sampled).real)
# plt.plot(np.fft.fft(signal_sampled_padded).real)
# plt.show()

ift_matrix = np.fromfunction(lambda w, k: 1/n*np.exp(2*np.pi*1j/n*w*k),(n,n))
# plt.matshow(ift_matrix.real,cmap=mplt.colormaps['hsv'])
# plt.show()

x = cp.Variable(n, complex=True)
objective = cp.Minimize(cp.norm(x,1))
constraints = [cp.abs(sampling@ift_matrix@x - signal_sampled) <= 0.1]
prob = cp.Problem(objective, constraints)
result = prob.solve(verbose=True)

plt.plot(np.fft.fft(signal).real)
plt.plot(np.fft.fft(signal_sampled_padded).real)
plt.plot(x.value.real)
plt.show()


plt.plot(signal.real)
plt.plot(signal_sampled_padded.real)
plt.plot(np.fft.ifft(x.value).real)
plt.show()