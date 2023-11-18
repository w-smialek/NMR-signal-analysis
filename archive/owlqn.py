from pylbfgs import owlqn
import numpy as np
import matplotlib.pyplot as plt

n = 20
s = 0.2

sampling = [np.random.choice(n,int(n*s),replace=False) for i in range(n)]
sampling_mask = np.zeros((n,n)).astype(bool)
for i in range(n):
    sampling_mask[i,sampling[i]] = 1

sig = np.random.randint(0,10,(n,n))

sig_s = np.reshape(sig[sampling_mask],(n,int(n*s)))

plt.matshow(sig)
plt.matshow(sampling_mask)
plt.matshow(sig_s)
plt.show()

def evaluate(x, g, step):

    Ax = np.reshape(np.fft.ifft2(x)[sampling_mask],(n,int(n*s)))

    return np.sum(np.power(Ax-sig,2))
    
print(evaluate(np.random.randint(0,10,(10,10)),1,1))