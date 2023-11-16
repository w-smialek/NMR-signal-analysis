import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from colorsys import hls_to_rgb
import cvxpy as cp
import matplotlib as mpl
import imageio

def average_over_neighbours(array):
    ars = [np.roll(array,t,(0,1)) for t in [(-1,0),(0,-1),(0,0),(0,1),(1,0),(-1,-1),(1,1),(-1,1),(1,-1)]]
    return sum(ars)

def colorize(z):
    r = np.abs(z)
    arg = np.angle(z) 

    h = (arg + pi)  / (2 * pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2) 
    return c

newcolors = np.array([[1,0,0,1-np.exp(-i/150)] for i in range(256)])
newcmp = mpl.colors.ListedColormap(newcolors)

def whitenoise_complex(stdev,shape):
    rng = np.random.default_rng()
    return rng.normal(0,stdev/2,shape) + 1j* rng.normal(0,stdev/2,shape)

def waveform2(n, F, t_indir):
    '''t_indir - indirect time values in terms of real time'''
    form = np.zeros((n,n)).astype(complex)
    for t2, tr in zip(t_indir, range(n)):
        form[t2,:] = np.array([F(t1,t2,tr) for t1 in range(n)])
    return form

class Signal():
    def __init__(self, form=np.array([0])):
        self.timedom = np.array(form)
        self.shape = self.timedom.shape
        self.len = self.shape[0]
        self.dim = len(self.shape)
        self.freqdom = np.fft.fft2(self.timedom,None,[i for i in range(self.dim)])
        self.dt1 = 1/self.len
        self.dt2 = 1/self.len

    def update(self):
        self.freqdom = np.fft.fft2(self.timedom,None,[i for i in range(self.dim)])

    def add(self,form):
        if not self.timedom.any():
            self.timedom = form
            self.update()
            return
        self.timedom += form
        self.update()

    def deshuffle(self,t_indir):
        self.timedom = self.timedom[np.argsort(t_indir)]
        # inverse_perm = np.empty_like(t_indir)
        # inverse_perm[t_indir] = np.arange(t_indir.size)
        # self.timedom = self.timedom[inverse_perm]
        self.update()

    def plot(self, type="time", **kwargs):
        fig = plt.figure(figsize=(6, 6))
        ax1 = fig.add_subplot()
        times_axis1 = np.array([self.dt1*i for i in range(self.len)])
        try:
            times_axis2 = np.array([self.dt2*i for i in range(self.timedom.shape[1])])
        except:
            pass
        freq_axis1 = np.array([1/(n*self.dt1)*i for i in range(self.len)])
        try:
            freq_axis2 = np.array([1/(n*self.dt2)*i for i in range(self.timedom.shape[1])])
        except:
            pass
        if type == "time":
            if self.dim == 1:
                plt.xlabel("time [s]")
                plt.plot(times_axis1, self.timedom.real, **kwargs)
            else:
                plt.imshow(self.timedom.real, **kwargs)
            # plt.show()
        if type == "freq":
            if self.dim == 1:
                ax1.set_ylim([-40, 270])
                plt.xlabel("frequency [Hz]")
                ax1.plot(freq_axis1,self.freqdom.real, **kwargs)
            else:
                # ax1.matshow(np.flip(average_over_neighbours(self.freqdom[int(1024*0.10):int(1024*0.30),int(1024*0.0):int(1024*0.2)].real),0), extent=[freq_axis1[-1]*0.00,freq_axis1[-1]*0.2,freq_axis2[-1]*0.10,freq_axis2[-1]*0.30], **kwargs)
                # ax1.tick_params(labeltop=False,top=False,bottom=True,labelbottom=True)
                # plt.xlabel("frequency [Hz]")
                # plt.ylabel("frequency [Hz]")

                # ax1.matshow(np.flip(self.freqdom.real,0), extent=[freq_axis1[0],freq_axis1[-1],freq_axis2[0],freq_axis2[-1]], **kwargs)
                # ax1.tick_params(labeltop=False,top=False,bottom=True,labelbottom=True)
                # plt.xlabel("direct frequency [Hz]")
                # plt.ylabel("indirect frequency [Hz]")

                fig = plt.figure(figsize=(6, 6))
                ax1 = fig.add_subplot(projection='3d')
                x = np.arange(np.shape(self.timedom)[1])/self.len/self.dt1
                y = np.arange(self.len)/self.len/self.dt2
                x, y = np.meshgrid(x, y)
                ax1.set_zlim([-200, 2000])
                plt.xlabel("direct frequency [Hz]")
                plt.ylabel("indirect frequency [Hz]")
                ax1.plot_surface(x,y,abs(self.freqdom),cmap=newcmp,linewidth=0,antialiased=True)

            # plt.show()

n = 40

###
### Effects of non-stationary frequency
###

### 1D

# form  = [1*np.exp(2j*pi*(0.1+0.05*t**2/n**2)*t - t/(n/2)) for t in range(n)]
# noise = whitenoise_complex(0.1,n)

# sig_1D_nonstat = Signal(form)
# sig_1D_nonstat.add(noise)
# sig_1D_nonstat.dt = 1/n

# plt.style.use('ggplot')

# sig_1D_nonstat.plot("freq", linewidth = 0.8)
# plt.savefig("1d_nonstat_example.png",dpi=300)

### 2D

# def non_stationary_frequency(t1,t2,treal):
#     f1, f2 = 0.1, 0.2 + 0.02*treal/n
#     tau1 = n/2
#     return np.exp(2j*pi*(f1*t1+f2*t2) - t1/tau1)

# t_indir = np.random.choice(n,n,replace=False)

# signal_lin = Signal(waveform2(n,non_stationary_frequency,range(n)))

# signal_perp = Signal(waveform2(n,non_stationary_frequency,t_indir))

# signal_lin.dt1 = signal_lin.dt2 = signal_perp.dt1 = signal_perp.dt2 = 1/(n/2)

# plt.style.use('classic')

# signal_lin.plot("freq",cmap="PiYG")
# # plt.savefig("indirect_lin.png",dpi=300)
# plt.show()
# signal_perp.plot("freq",cmap="PiYG")
# plt.savefig("indirect_perp_big.png",dpi=300)
# plt.show()

###
### Time-resolved NUS
###

# def snapshot_2d(n, F, t_real, t_indir):
#     '''n - number of direct time datapoints \n
#     treal - range of real time values \n
#     t_indir - corresponding indirect time values'''
#     form = np.zeros((n,n)).astype(complex)
#     for t2, tr in zip(t_indir, t_real):
#         form[t2,:] = np.array([F(t1,t2,tr) for t1 in range(n)])
#     return form

# def average_over_neighbours(array):
#     ars = [np.roll(array,t,(0,1)) for t in [(-1,0),(0,-1),(0,0),(0,1),(1,0),(-1,-1),(1,1),(-1,1),(1,-1)]]
#     return sum(ars)

# n_snapshots = 80
# treal_interval = 20

# snaphot_sampling_ratio = treal_interval/n

# snapshots = []
# for i in range(n_snapshots):
#     t_real_interval = (int(treal_interval*i),int(treal_interval*(i+1)))
#     t_indir = np.random.choice(n,int(n*snaphot_sampling_ratio),replace=False)
#     ss = Signal(snapshot_2d(n,non_stationary_frequency,range(*t_real_interval),t_indir))
#     ss.freqdom = average_over_neighbours(ss.freqdom)
#     snapshots.append(ss)

# maxes = []

# for snap in snapshots:
#     # snap.plot(type="time")
#     # snap.plot(type="freq")
#     maxes.append(np.argmax(snap.freqdom))

# mm = np.zeros((n,n))
# for m in maxes:
#     mm[(m//n,m%n)] = 1

# plt.imshow(mm)
# plt.show()


###
### CS reconstruction
###

### 1D

def sampling_matrix(sampling_mask):
    '''rectangular sampling matrix from a vector 0 or 1 sampling mask'''
    sampling_mat = np.array([row for row in np.diag(sampling_mask) if np.sum(row) == True])
    return sampling_mat

def vectorization(matrix):
    '''flattening of a 2d array'''
    n,m = np.shape(matrix)
    return np.array([matrix[i//m,i%m] for i in range(n*m)])

def matricization(tensor):
    '''2d matrix from a 4d tensor'''
    a,b,c,d = np.shape(tensor)
    return np.array([[tensor[j%a,j//a,i//d,i%d] for i in range(c*d)] for j in range(a*b)])

def cs_reconstruct_1d(sig_sampled,sampling_matrix,delta):
    l = np.shape(sampling_matrix)[1]
    sig_sampled = sig_sampled.timedom
    ift_matrix = np.fromfunction(lambda w, k: 1/l*np.exp(2*np.pi*1j/l*w*k),(l,l))
    x = cp.Variable(l, complex=True)
    objective = cp.Minimize(cp.norm(x,1))
    print(sig_sampled.shape, sampling_matrix.shape, ift_matrix.shape)
    constraints = [cp.abs(sampling_matrix@ift_matrix@x - sig_sampled) <= delta]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(verbose=True)
    return x.value

def cs_reconstruct_2d(sig_sampled,sampling_matricized,delta):
    l = int(np.sqrt(np.shape(sampling_matricized)[1]))
    print(l)
    sig_sampled = sig_sampled.timedom
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

# sampling_ratio = 0.2

# mask = np.random.choice([0,1],n,replace=True,p=(1-sampling_ratio,sampling_ratio))
# sampl_matrix = sampling_matrix(mask)

# test_sig = np.array([np.exp(2j*pi*0.4*t1-t1/(n/2))+np.exp(2j*pi*0.7*t1-t1/(n/2)) for t1 in range(n)])
# test_sig += whitenoise_complex(0.3,n)
# test_sig_full = Signal(test_sig)
# test_sig = test_sig[mask.astype(bool)]
# test_sig = Signal(test_sig)

# plt.style.use("ggplot")
# # test_sig_full.plot(type="time")
# test_sig_full.plot(type="freq", linewidth = 0.8)
# plt.savefig("1d_rec_orig.png",dpi=300)
# plt.show()
# # test_sig.plot(type="time")
# test_sig.plot(type="freq")
# plt.show()


# x = cs_reconstruct_1d(test_sig,sampl_matrix,0.1)

# rec_sig = Signal(np.fft.ifft(x))
# # rec_sig.plot(type="time")
# rec_sig.plot(type="freq", linewidth = 0.8)
# plt.savefig("1d_rec_rec.png",dpi=300)
# plt.show()


### 2D

# sampling_ratio = 0.3

# # def non_stationary_frequency(t1,t2,treal):
# #     f1, f2 = 0.1+0.02*treal/n , 0.2 + 0.02*treal/n
# #     tau1, tau2 = n/2, n/2
# #     return np.exp(2j*pi*(f1*t1+f2*t2)-t1/(tau1)-t2/(tau2))

# def non_stationary_frequency(t1,t2,treal):
#     f1, f2 = 0.1, 0.2
#     f3, f4 = 0.3, 0.6
#     tau1, tau2 = n/2, n/2
#     return np.exp(2j*pi*(f1*t1+f2*t2)-t1/tau1) + 0.5*np.exp(2j*pi*(f3*t1+f4*t2)-t1/tau2)


# sig = waveform2(n,non_stationary_frequency,range(n))
# signal0 = Signal(sig)
# signal0.add(whitenoise_complex(0.2,(n,n)))
# signal0.dt1 = signal0.dt2 = 1/n

# signal0.plot("freq",cmap=newcmp)
# plt.savefig("2d_rec_orig.png",dpi=300)
# plt.show()

# sampling_rows = np.random.choice([0,1],n,replace=True,p=(1-sampling_ratio, sampling_ratio))
# sampling_mask = np.array([np.ones(n) if ifrow else np.zeros(n) for ifrow in sampling_rows])

# sampling_mat = sampling_matrix(vectorization(sampling_mask))

# sig_sam = sig[sampling_mask.astype(bool)].reshape((np.count_nonzero(sampling_rows),n))
# signal_sampled = Signal(sig_sam)

# x = cs_reconstruct_2d(signal_sampled,sampling_mat,0.1)

# signal_rec = Signal(np.fft.ifft2(x.reshape((n,n)).T))
# signal_rec.dt1 = signal_rec.dt2 = 1/n
# signal_rec.plot("freq",cmap=newcmp)
# plt.savefig("2d_rec_rec.png",dpi=300)
# plt.show()

###
### NUS - non-stationarity compromise
###

### 1D

# frate = 0.01

# def waveform(F, t_real, t_sampled):
#     return [F(t_r,t_s) for t_r, t_s in zip(t_real, t_sampled)]

# def non_stat_freq(t_r, t_s):
#     a1, a2 = 2, 1
#     f1, f2 = 0.2 + frate*t_r/n, 0.5 + frate*t_r/n
#     tau1, tau2 = n/3, n/3
#     return a1*np.exp(2j*pi*t_s*f1 -t_s/tau1) + a2*np.exp(2j*pi*t_s*f2 -t_s/tau2)

# plt.style.use("ggplot")

# sig = Signal(waveform(non_stat_freq,range(n),range(n)))
# sig.add(whitenoise_complex(0.3,n))
# sig.dt1 = 1/n
# # sig.plot(type="time")
# sig.plot(type="freq", linewidth = 0.8)
# plt.savefig("1d_compromise_%s_orig.png"%int(100*frate),dpi=300)
# plt.close()

# for i in np.linspace(0.1,1,10):
#     sampling_ratio = i
#     mask = np.zeros(n)
#     filter = np.random.choice(np.arange(n),int(sampling_ratio*n),replace=False)
#     mask[filter] = 1

#     form = waveform(non_stat_freq, range(int(n*sampling_ratio)), filter)

#     sig_sampled = Signal(form)
#     sig_sampled.deshuffle(filter)
#     # sig_sampled.plot(type="time")
#     # sig_sampled.plot(type="freq")
#     # plt.show()
    
#     a = np.fft.ifft(cs_reconstruct_1d(sig_sampled, sampling_matrix(mask), 0.5))

#     if i ==0.99:
#         sig_rec = Signal(a)
#     else:
#         sig_rec = Signal(a)
#     # sig_rec.plot(type="time")
#     plt.style.use("ggplot")
#     sig_rec.plot(type="freq", linewidth = 0.8)
#     # plt.plot(sig_rec.freqdom)
#     plt.savefig("1d_compromise_%s_%s.png"%(int(100*frate),(int(i*100))),dpi=300)
#     plt.close()

# filenames = ["1d_compromise_%s_%s.png"%(int(100*frate),(int(i*100))) for i in np.linspace(0.1,1,10)]

# with imageio.get_writer("1d_compromise_%s.gif"%int(100*frate), mode='I',duration=1) as writer:
#     for filename in filenames:
#         image = imageio.imread(filename)
#         writer.append_data(image)


### 2D

frate = 0.02

def non_stationary_frequency(t1,t2,treal):
    f1, f2, f3, f4 = 0.2+frate*treal/n, 0.4+frate*treal/n, 0.6+frate*treal/n, 0.3+frate*treal/n
    tau1, tau2 = n, n
    return 2*np.exp(2j*pi*(f1*t1+f2*t2)-t1/tau1) + np.exp(2j*pi*(f3*t1+f4*t2)-t1/tau2)#-t1/(tau1)-t2/(tau2))

def snapshot_2d(n, F, t_real, t_indir):
    '''n - number of direct time datapoints \n
    treal - range of real time values \n
    t_indir - corresponding indirect time values'''
    form = np.zeros((n,n)).astype(complex)
    for t2, tr in zip(t_indir, t_real):
        form[t2,:] = np.array([F(t1,t2,tr) for t1 in range(n)])
    return form[t_indir]

sig_original = Signal(snapshot_2d(n,non_stationary_frequency,range(n),range(n)))
sig_original.add(whitenoise_complex(0.3,(n,n)))
# sig_original.plot()
sig_original.plot("freq")
# plt.matshow(sig_original.freqdom.real)
plt.savefig("2d_compromise_%s_orig.png"%int(frate*100),dpi=300)
plt.close()

for i in np.linspace(0.1,1,10):
    t_r_schedule = np.random.choice(np.arange(int(n*i)),int(n*i),replace=False)

    filter = np.random.choice(np.arange(n),int(n*i),replace=False)
    mask = np.zeros((n,n))
    mask[filter] = np.ones(n)
    sampling_mat = sampling_matrix(vectorization(mask))

    sig_padded = np.zeros((n,n)).astype(complex)
    for t2, treal in zip(filter, t_r_schedule):
        sig_padded[t2,:] = [non_stationary_frequency(t1,t2,treal) for t1 in range(n)]

    sig_subsampled = sig_padded[mask.astype(bool)].reshape((filter.size,n))

    sig_padded = Signal(sig_padded)
    sig_subsampled = Signal(sig_subsampled)
    # sig_padded.plot()
    # sig_subsampled.plot()

    sig_reconstructed = cs_reconstruct_2d(sig_subsampled,sampling_mat,0.1*i)
    sig_reconstructed = Signal(np.fft.ifft2(sig_reconstructed.reshape((n,n)).T))

    sig_reconstructed.plot("freq")
    # plt.matshow(sig_reconstructed.freqdom.real)
    plt.savefig("2d_compromise_%s_%s.png"%(int(100*frate),int(i*100)),dpi=300)

filenames = ["2d_compromise_%s_%s.png"%(int(100*frate),(int(i*100))) for i in np.linspace(0.1,1,10)]

with imageio.get_writer("2d_compromise_%s.gif"%int(100*frate), mode='I',duration=1) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)


# n_snapshots = 10
# t_r_length = 5

# sampling_ratio = t_r_length/n

# for i in range(n_snapshots):
#     t_r_schedule = np.random.choice(np.arange(i*t_r_length,(i+1)*t_r_length),t_r_length,replace=False)

#     filter = np.random.choice(np.arange(n),int(n*sampling_ratio),replace=False)
#     mask = np.zeros((n,n))
#     mask[filter] = np.ones(n)
#     sampling_mat = sampling_matrix(vectorization(mask))

#     sig_padded = np.zeros((n,n)).astype(complex)
#     for t2, treal in zip(filter, t_r_schedule):
#         sig_padded[t2,:] = [non_stationary_frequency(t1,t2,treal) for t1 in range(n)]

#     sig_subsampled = sig_padded[mask.astype(bool)].reshape((filter.size,n))

#     sig_padded = Signal(sig_padded)
#     sig_subsampled = Signal(sig_subsampled)
#     sig_padded.plot()
#     sig_subsampled.plot()

#     sig_reconstructed = cs_reconstruct_2d(sig_subsampled,sampling_mat,0.1)
#     sig_reconstructed = Signal(np.fft.ifft2(sig_reconstructed.reshape((n,n)).T))

#     sig_reconstructed.plot("freq")

###
### 2D time-resolved NUS with CS reconstruction
###

def non_stationary_frequency(t1,t2,treal):
    f1, f2 = 0.2+0.015*treal/n, 0.4+0.015*treal/n
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

sampling_ratio = 0.3
t_r_length = int(sampling_ratio*n)

n_samples = 10

# for i in range(n_samples):
#     t_r_schedule = np.random.choice(np.arange(int(t_r_length*i),int(t_r_length*(i+1))),t_r_length,replace=False)
#     filter = np.random.choice(np.arange(n),t_r_length,replace=False)
#     mask = np.zeros((n,n))
#     mask[filter] = np.ones(n)
#     sampling_mat = sampling_matrix(vectorization(mask))

#     sig_padded = np.zeros((n,n)).astype(complex)
#     for t2, treal in zip(filter, t_r_schedule):
#         sig_padded[t2,:] = [non_stationary_frequency(t1,t2,treal) for t1 in range(n)]

#     sig_subsampled = sig_padded[mask.astype(bool)].reshape((filter.size,n))

#     sig_padded = Signal(sig_padded)
#     sig_subsampled = Signal(sig_subsampled)
#     # sig_padded.plot()
#     # sig_subsampled.plot()

#     sig_reconstructed = cs_reconstruct_2d(sig_subsampled,sampling_mat,0.6)
#     sig_reconstructed = Signal(np.fft.ifft2(sig_reconstructed.reshape((n,n)).T))

#     # sig_reconstructed.plot("freq")
#     plt.matshow(sig_reconstructed.freqdom.real)
#     plt.savefig("2d_TS1_%s.png"%(int(i*100)),dpi=300)
#     plt.close()


# def average_over_neighbours(array):
#     ars = [np.roll(array,t,(0,1)) for t in [(-1,0),(0,-1),(0,0),(0,1),(1,0),(-1,-1),(1,1),(-1,1),(1,-1)]]
#     return sum(ars)

# n_snapshots = 10
# treal_interval = 10

# snaphot_sampling_ratio = treal_interval/n

# snapshots = []
# masks = []
# for i in range(n_snapshots):
#     t_real_interval = (int(treal_interval*i),int(treal_interval*(i+1)))
#     t_indir = np.random.choice(n,int(n*snaphot_sampling_ratio),replace=False)
#     ss = Signal(snapshot_2d(n,non_stationary_frequency,range(*t_real_interval),t_indir))
#     ss.deshuffle(t_indir)
#     mask = np.array([np.ones(n) if (i in t_indir) else np.zeros(n) for i in range(n)])
#     masks.append(mask)
#     snapshots.append(ss)

# maxes = []
# recs = []

# for snap, mask in zip(snapshots,masks):
#     snap.plot(type="freq")
#     x = cs_reconstruct_2d(snap,sampling_matrix(vectorization(mask)),1)
#     rec = Signal(np.fft.ifft2(x.reshape((n,n))).T)
#     rec.plot("freq")
#     recs.append(rec)
#     maxes.append(np.argmax(rec.freqdom))

# mm = np.zeros((n,n))
# for m in maxes:
#     mm[(m//n,m%n)] = 1

# plt.imshow(mm)
# plt.show()