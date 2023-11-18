import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import matplotlib as mpl

def average_over_neighbours(array):
    ars = [np.roll(array,t,(0,1)) for t in [(-1,0),(0,-1),(0,0),(0,1),(1,0),(-1,-1),(1,1),(-1,1),(1,-1)]]
    return sum(ars)

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
        freq_axis1 = np.array([1/(self.len*self.dt1)*i for i in range(self.len)])
        try:
            freq_axis2 = np.array([1/(self.len*self.dt2)*i for i in range(self.timedom.shape[1])])
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
                ax1.set_zlim([-10, 1000])
                plt.xlabel("direct frequency [Hz]")
                plt.ylabel("indirect frequency [Hz]")
                ax1.plot_surface(x,y,abs(self.freqdom),cmap=newcmp,linewidth=0,antialiased=True)

            # plt.show()

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