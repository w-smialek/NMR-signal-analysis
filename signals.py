import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import matplotlib
import scipy.fftpack as fpack

###
### Functions
###

def sinexp(x,ampl,freq,tau):
    return ampl*np.exp(x*(2*np.pi*freq*1j - 1/tau))

def whitenoise_complex(stdev,n):
    rng = np.random.default_rng()
    return rng.normal(0,stdev/2,n) + 1j* rng.normal(0,stdev/2,n)

def sinexp_2d_list(ampls,freqs,taus,ns):
    return [[sinexp(t1,ampls[0],freqs[0],taus[0])*sinexp(t2,ampls[1],freqs[1],taus[1]) for t1 in range(ns[0])] for t2 in range(ns[1])]

def cs_reconstruct_1d(sig_sampled,sampling_matrix,delta):
    l = np.shape(sampling_matrix)[1]
    print(l)
    sig_sampled = sig_sampled.timedom.form
    ift_matrix = np.fromfunction(lambda w, k: 1/l*np.exp(2*np.pi*1j/l*w*k),(l,l))
    x = cp.Variable(l, complex=True)
    objective = cp.Minimize(cp.norm(x,1))
    constraints = [cp.abs(sampling_matrix@ift_matrix@x - sig_sampled) <= delta]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(verbose=True)
    return x.value

def cs_reconstruct_2d(sig_sampled,sampling_matricized,delta):
    l = int(np.sqrt(np.shape(sampling_matricized)[1]))
    print(l)
    sig_sampled = sig_sampled.timedom.form
    sig_sampled_vectorized = vectorization(sig_sampled)
    ift_tensor = np.fromfunction(lambda t1, t2, k1, k2: 1/(l**2)*np.exp(2*np.pi*1j/l*(t1*k1+t2*k2)),(l,l,l,l))
    # ift_tensor_matricized = np.fromfunction(lambda t, k: 1/(l**2)*np.exp(2*np.pi*1j/l*((t%l)*(k//l)+(t//l)*(k%l))),(l,l))
    ift_tensor_matricized = matricization(ift_tensor)
    # sampling_matricized = matricization(sampling)
    x = cp.Variable(l**2, complex=True)
    objective = cp.Minimize(cp.norm(x,1))
    constraints = [cp.abs(sampling_matricized@ift_tensor_matricized@x - sig_sampled_vectorized) <= delta]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(verbose=True)
    return x.value

def sampling_matrix(sampling_mask):
    '''rectangular sampling matrix from a vector 0 or 1 sampling mask'''
    sampling_mat = np.array([row for row in np.diag(sampling_mask) if np.sum(row) == True])
    return sampling_mat

def vectorization(matrix):
    n,m = np.shape(matrix)
    return np.array([matrix[i//m,i%m] for i in range(n*m)])

def matricization(tensor):
    a,b,c,d = np.shape(tensor)
    return np.array([[tensor[j%a,j//a,i//d,i%d] for i in range(c*d)] for j in range(a*b)])

###
### Classes
###

class Waveform:
    def __init__(self,l):
        self.form = np.array(l)
        self.size = self.form.size

    def real(self):
        return np.real(self.form)
    
    def imag(self):
        return np.imag(self.form)
    
    def ft(self):
        return Waveform(np.fft.fft(self.form))
    
class Waveform2D(Waveform):

    def ft(self):
        return Waveform2D(np.fft.fft2(self.form))
    
    def n_dir(self):
        return np.shape(self.form)[0]
    
    def n_indir(self):
        return np.shape(self.form)[1]
    
    
class Signal:
    def __init__(self,delta1):
        self.dt1 = delta1
        self.timedom = None
        self.len = 0

    def update(self):
        self.freqdom = self.timedom.ft()
        self.len = np.shape(self.timedom.form)[0]
    
    def set_signal(self, signal):
        self.timedom = Waveform(signal)
        self.update()

    def add_signal(self, signal):
        if self.timedom == None:
            self.set_signal(signal)
        else:
            self.timedom.form += np.array(signal)
            self.update()

    def sparse_sample(self, sampling_mask):
        sig_s = Signal(self.dt1)
        sig_p = Signal(self.dt1)
        sig_s.set_signal(self.timedom.form[sampling_mask>0])
        sig_p.set_signal(self.timedom.form*sampling_mask)
        return sig_s, sig_p

    def plot(self, type="time", part = "complex"):
        if type == "time":
            if not part == "imag":
                plt.plot([i*self.dt1 for i in range(self.timedom.size)], self.timedom.real())
            if not part == "real":
                plt.plot([i*self.dt1 for i in range(self.timedom.size)], self.timedom.imag())
            plt.xlabel("[s]")
            # plt.show()
        if type == "freq":
            if not part == "imag":
                plt.plot([i/self.freqdom.size/self.dt1 for i in range(self.freqdom.size)], self.freqdom.real())
            if not part == "real":
                plt.plot([i/self.freqdom.size/self.dt1 for i in range(self.freqdom.size)], self.freqdom.imag())
            plt.xlabel("[Hz]")
            # plt.show()

class Signal2D(Signal):
    def __init__(self, delta1, delta2):
        super().__init__(delta1)
        self.dt2 = delta2

    def set_signal(self, signal):
        self.timedom = Waveform2D(signal)
        self.update()

    def direct_slice(self,indirect_t):
        dir_sl = Signal(self.dt1)
        dir_sl.set_signal(self.timedom.form[indirect_t,:])
        return dir_sl
    
    def set_direct_slice(self,l,indirect_t):
        self.timedom.form[indirect_t,:] = np.array(l)
        self.update()
    
    def add_direct_slice(self,l,indirect_t):
        self.timedom.form[indirect_t,:] += np.array(l)
        self.update()

    def sparse_sample(self, sampling_mask):
        l = np.shape(self.timedom.form)[0]
        sig_s = Signal2D(self.dt1,self.dt2)
        sig_p = Signal2D(self.dt1,self.dt2)

        sampl_mat = np.array([row for row in np.diag(vectorization(sampling_mask)) if np.sum(row) == True])

        sig_s_v = np.matmul(sampl_mat,vectorization(self.timedom.form))
        sig_s_2d = [[sig_s_v[i+l*j] for i in range(l)] for j in range(len(sig_s_v)//l)]

        sig_s.set_signal(sig_s_2d)
        sig_p.set_signal(self.timedom.form*sampling_mask)
        return sig_s, sig_p

    def plot_direct(self, indirect_t ,type="time", part = "complex"):
        print(self.direct_slice(indirect_t).timedom.form)
        self.direct_slice(indirect_t).plot(type,part)

    def plot2D(self):
        plt.imshow(self.timedom.real(), interpolation='none')
        plt.xlabel("direct dimension [%s s]"%self.dt1)
        plt.ylabel("indirect dimension [%s s]"%self.dt2)
        # plt.show()

    def freqplot2D(self):
        fig = plt.figure(figsize=(6, 6))
        ax1 = fig.add_subplot(projection='3d')

        x = np.arange(np.shape(self.timedom.form)[1])/self.len/self.dt1
        y = np.arange(self.len)/self.len/self.dt1
        x, y = np.meshgrid(x, y)

        # x = [i%self.len for i in range(self.len**2)]
        # y = [i//self.len for i in range(self.len**2)]
        # dz = [self.freqdom.form[i//self.len,i%self.len].real for i in range(self.len**2)]

        # offset = dz + np.abs(min(dz))
        # fracs = (offset.astype(float)/max(offset))**(1/2)
        # norm = matplotlib.colors.Normalize(fracs.min(), fracs.max())
        # colors = matplotlib.cm.jet(norm(fracs))

        # ax1.bar3d(x, y, 0, 1, 1, dz, shade=True, color=colors)

        ax1.plot_surface(x,y,self.freqdom.form,cmap=matplotlib.cm.seismic,linewidth=0,antialiased=True)
        plt.xlabel("direct dimension [Hz]")
        plt.ylabel("indirect dimension [Hz]")


        # plt.imshow(self.freqdom.real(), interpolation='none')
        # plt.show()