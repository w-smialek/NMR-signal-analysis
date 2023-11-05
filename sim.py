import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import matplotlib
import scipy.fftpack as fpack
import signals as sig

###
### 1D
###

sampling_rate = 0.0001
sampling_ratio = 0.4
delta = 0.2


# n = 100
# ampls = (1,3,2)
# freqs = (0.1,0.04,0.3)
# taus = (35,60,20)

# s = Signal(sampling_rate)
# for (a,f,t) in zip(ampls,freqs,taus):
#     s.add_signal([sinexp(x,a,f,t) for x in range(n)])

# noise = whitenoise_complex(stdev=0.2,n=n)

# s.add_signal(noise)

# sampling_mask = np.random.choice([0,1],n,p=[1-sampling_ratio,sampling_ratio])
# s_m = sampling_matrix(sampling_mask)

# print(sampling_mask)
# print(s_m)

# s_s, s_s_p = s.sparse_sample(sampling_mask)

# s_rec = Signal(sampling_rate)
# s_rec.set_signal(np.fft.ifft(cs_reconstruct_1d(s_s,s_m,0.1)))


# s.plot("time","real")
# s_rec.plot("time","real")
# plt.show()

# s.plot("freq","real")
# s_rec.plot("freq","real")
# plt.show()


###
### 2D
###

n = 10

form2d = [[sig.sinexp(t1,1,0.3+t2*0.3/100,30)*sig.sinexp(t2,1,0.15,60) for t1 in range(n)] for t2 in range(n)]

s2 = sig.Signal2D(sampling_rate,sampling_rate)

s2.add_signal(sig.sinexp_2d_list((0.6,0.5),(0.3,0.7),(40,50),(n,n)))
s2.add_signal(sig.sinexp_2d_list((0.7,0.6),(0.7,0.8),(60,60),(n,n)))
s2.add_signal(sig.sinexp_2d_list((0.6,0.3),(0.7,0.2),(25,45),(n,n)))
s2.add_signal(form2d)
s2.add_signal(sig.whitenoise_complex(0.15,(n,n)))
s2.plot2D()
s2.freqplot2D()

row_mask = np.random.choice([0,1],n,p=[1-sampling_ratio,sampling_ratio])
sampling_mask = np.array([np.ones(n) * r for r in row_mask])

plt.matshow(sampling_mask)
plt.show()

s2_s, s2_p = s2.sparse_sample(sampling_mask)

s2_s.plot2D()
s2_s.freqplot2D()

s2_p.plot2D()
s2_p.freqplot2D()

sampl_mat = np.array([row for row in np.diag(sig.vectorization(sampling_mask)) if np.sum(row) == True])

s2_rec = sig.Signal2D(sampling_rate,sampling_rate)

sig_s_vector = np.array([sig.vectorization(s2_s.timedom.form)])
sig_s = s2_s.timedom.form.real

ift_tensor = np.fromfunction(lambda t1, t2, k1, k2: 1/(n**2)*np.exp(2*np.pi*1j/n*(t1*k1+t2*k2)),(n,n,n,n))
# ift_tensor_matricized = np.fromfunction(lambda t, k: 1/(l**2)*np.exp(2*np.pi*1j/l*((t%l)*(k//l)+(t//l)*(k%l))),(l,l))
ift_tensor_matricized = sig.matricization(ift_tensor)

# print(np.shape(sampl_mat))
# print(np.shape(ift_tensor_matricized))

A_mat = np.matmul(sampl_mat,ift_tensor_matricized)

# print(np.shape(A_mat))

def dct2(x):
    return fpack.dct(fpack.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct2(x):
    return fpack.idct(fpack.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

# def evaluate(x, g, step):
#     x2 = x.reshape((n,n)).T
#     # m = np.shape(sampl_mat)[0]
#     # print(m)
#     Ax2 = idct2(x2)
#     Ax = Ax2.T.flat[sampling_mask.astype(bool).reshape(n**2)].reshape(sig_s_vector.shape)
#     # Ax = Ax[sampling_mask.astype(bool)]
#     Axb = Ax - sig_s_vector
#     print(Axb)

#     # Axb = np.matmul(A_mat,x) - sig_s_vector.T
#     # Axb = Axb.real

#     Axb2 = np.zeros(x2.shape)
#     Axb2.T.flat[sampling_mask.astype(bool).reshape(n**2)] = Axb

#     AtAxb2 = 2 * dct2(Axb2)
#     AtAxb = AtAxb2.T.reshape(x.shape)

#     # grad = 2*np.matmul(A_mat.T.real,Axb.T.real)
#     np.copyto(g, AtAxb)
#     # s = np.sum(np.power(Axb,2))
#     # print(s)
#     return np.sum(np.power(abs(Axb),2))

# def evaluate(x,g,step)
    


from pylbfgs import owlqn

Xat2 = owlqn(n**2, evaluate, None, 100)
Xat = Xat2.reshape(n, n) # stack columns
Xa = np.fft.ifft2(Xat)

f = open("rec_sig", "w")
f.write(np.array_str(Xa))
f.close()

s2_rec.set_signal(Xa)
s2_rec.plot2D()
s2_rec.freqplot2D()

# reconstructed_vectorized = cs_reconstruct_2d(s2_s,sampl_mat, delta)
# reconstructed = np.array([[reconstructed_vectorized[i+j*n] for i in range(n)] for j in range(n)])

# s2_rec.set_signal(np.fft.ifft2(reconstructed))
# s2_rec.plot2D()
# s2_rec.freqplot2D()