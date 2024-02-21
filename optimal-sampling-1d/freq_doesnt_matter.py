import numpy as np
import matplotlib.pyplot as plt
import itertools
import math
from random import randrange

### Functions

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

n = 500

### real time passes, indirect time sampled
### or reshuffled - sampling of real time values

# def signal(f, fbar, dec, sampling):
#     return [np.exp(2j*np.pi*(f + fbar * tau)*t)*np.exp(-t*dec) for t, tau in zip(range(n),sampling)]

def signal(f0, f0bar, f, fbar, dec, sampling):
    result = np.zeros(n).astype(complex)
    for t, tau in zip(sampling,range(n)):
        result[t] = np.exp(2j*np.pi*((f + fbar * tau)*t + f0 + f0bar*tau))*np.exp(-t/dec)
    return result

def block_perm_sampling(n_pieces, perm, signs):
    pieces_forw = [list(i) for i in list(split(range(n), n_pieces))]
    pieces_backw = [list(i) for i in list(split(range(n-1,-1,-1), n_pieces))]
    pieces_backw.reverse()
    sampling = []
    for p,s in zip(perm,signs):
        if s:
            sampling += pieces_forw[p]
        else:
            sampling += pieces_backw[p]
    return sampling

def fitness_function(sampling, f, fbar, d_fbar, dec):
    fit = 0
    for k in np.linspace(-5,5,20):
        sig = signal(f0, f0bar, f,fbar+k*d_fbar,dec,sampling)
        sig_ft = abs(np.fft.fft(sig))
        fit += max(abs(sig_ft))*np.exp(-alpha*abs(k))
    return fit

_invo_cnts = [1, 1, 2, 4, 10, 26, 76, 232, 764, 2620, 9496, 35696, 140152]
def invo_count(n):
    """Return the number of involutions of size n and cache the result."""
    for i in range(len(_invo_cnts), n+1):
        _invo_cnts.append(_invo_cnts[i-1] + (i-1) * _invo_cnts[i-2])
    return _invo_cnts[n]

def randinvolution(n):
    """Return a random (uniform) involution of size n."""
    involution = list(range(n))
    unseen = list(range(n))
    cntunseen = n
    while cntunseen > 1:
        if randrange(invo_count(cntunseen)) < invo_count(cntunseen - 1):
            cntunseen -= 1
        else:
            idxother = randrange(cntunseen - 1)
            other = unseen[idxother]
            current = unseen[cntunseen - 1]
            involution[current], involution[other] = (involution[other], involution[current])
            unseen[idxother] = unseen[cntunseen - 2]
            cntunseen -= 2

    return involution

def InvolutionChanges(n):
    """Generate change sequence for involutions on n items.
    Uses a variation of the Steinhaus-Johnson-Trotter idea,
    in which we first recurse for n-1, generating involutions
    in which the last item is fixed, and then we the match
    for the last item back and forth over a recursively
    generated sequence for n-2."""
    if n <= 3:
        for c in [[],[],[0],[0,1,0]][n]:
            yield c
        return
    for c in InvolutionChanges(n-1):
        yield c
    yield n-2
    for i in range(n-4,-1,-1):
        yield i
    ic = InvolutionChanges(n-2)
    up = range(0,n-2)
    down = range(n-3,-1,-1)
    try:
        while True:
            yield next(ic) + 1
            for i in up:
                yield i
            yield next(ic)
            for i in down:
                yield i
    except StopIteration:
        yield n-4

def Involutions(n):
    """Generate involutions on n items.
    The first involution is always the one in which all items
    are mapped to themselves, and the last involution is the one
    in which only the final two items are swapped.
    Each two involutions differ by a change that either adds or
    removes an adjacent pair of swapped items, moves a swap target
    by one, or swaps two adjacent swap targets."""
    p = list(range(n))
    yield p
    for c in InvolutionChanges(n):
        x,y = p[c],p[c+1]   # current partners of c and c+1
        if x == c and y != c+1: x = c+1
        if x != c and y == c+1: y = c
        p[x],p[y],p[c],p[c+1] = c+1, c, y, x    # swap partners
        yield p

### Parameters

f = 0.8
fbar = 0.005/n
f0 = 0.2
f0bar = 0.0/n
dec = n*10

n_pieces = 6
delta_f = fbar/20#0.003/n
alpha = 0

### Loop

perms = itertools.permutations(range(n_pieces))

max_all = 0
sampling_max = []
perm_max = []
indx = 0

maxes = []

print_interval = math.factorial(n_pieces)//10
for p in perms:
    for s in [n_pieces*[True],n_pieces*[False]]:#itertools.product([True,False], repeat=n_pieces):
        sampling = block_perm_sampling(n_pieces,p,s)
        fit_current = fitness_function(sampling,f,fbar,delta_f,dec)
        maxes.append(fit_current)
        if fit_current > max_all:
            print(fit_current)
            max_all = fit_current
            sampling_max = sampling
            perm_max = p
    indx+=1
    if indx%print_interval == 0:
        print(10*indx//print_interval)

indx = 0
max_all_inv = 0
sampling_max_inv = None
perm_max_inv = None

maxes_inv = []

print_interval = invo_count(n_pieces)//10
for p in Involutions(n_pieces):
    for s in [n_pieces*[True],n_pieces*[False]]:#itertools.product([True,False], repeat=n_pieces):
        sampling = block_perm_sampling(n_pieces,p,s)
        fit_current = fitness_function(sampling,f,fbar,delta_f,dec)
        maxes_inv.append(fit_current)
        if fit_current > max_all_inv:
            print(fit_current)
            max_all_inv = fit_current
            sampling_max_inv = sampling
            perm_max_inv = p
    indx+=1
    if indx%print_interval == 0:
        print(10*indx//print_interval)


# ### Plots

# Best sampling
print(perm_max)
print(perm_max_inv)
plt.plot(sampling_max)
plt.show()
plt.plot(sampling_max_inv)
plt.show()

# Trivial sampling fitness
max_control = fitness_function(range(n),f,fbar,delta_f,dec)
print(max_control)

# Fitness of all permutations and reference trivial sampling
plt.plot(maxes)
plt.plot(maxes_inv)
# plt.plot([max_control for m in maxes])
plt.show()

# Trivial and optimal sampling spectra
sig = signal(f0, f0bar, f,fbar,dec,range(n))
sig_ft = abs(np.fft.fft(sig))
plt.plot(sig_ft)

sig = signal(f0, f0bar, f,fbar,dec,sampling_max_inv)
sig_ft = abs(np.fft.fft(sig))
plt.plot(sig_ft)
plt.show()

###

# for fbar in [0.001/n,0.005/n,0.01/n,0.02/n,0.04/n,0.08/n,0.12/n]:
#     maxes_down = []
#     maxes_up = []
#     for l in range(1,n):
#         perm_down = list(range(l-1,-1,-1))
#         perm_up = list(range(l))
#         s_down = l*[True]
#         s_up = l*[False]
#         sampling_down = block_perm_sampling(l,perm_down,s_down)
#         sampling_up = block_perm_sampling(l,perm_up,s_up)
#         fit_current_down = fitness_function(sampling_down,f,fbar,delta_f,dec)
#         fit_current_up = fitness_function(sampling_up,f,fbar,delta_f,dec)
#         maxes_down.append(fit_current_down)
#         maxes_up.append(fit_current_up)

#     sampling_jump_down = block_perm_sampling(2,[1,0],[True,True])
#     sampling_jump_up = block_perm_sampling(2,[0,1],[False,False])

#     fit_down = fitness_function(sampling_jump_down,f,fbar,delta_f,dec)
#     fit_up = fitness_function(sampling_jump_up,f,fbar,delta_f,dec)

#     fit_control = fitness_function(range(n),f,fbar,delta_f,dec)
#     fit_backw = fitness_function(range(n-1,-1,-1),f,fbar,delta_f,dec)

#     plt.plot(maxes_down, label="downw. saw")
#     plt.plot(maxes_up, label="upw. saw")
#     plt.plot([fit_down for i in range(n)], label="1 jump down")
#     plt.plot([fit_up for i in range(n)], label="1 jump up")
#     plt.plot([fit_control for i in range(n)], label="all forw.")
#     plt.plot([fit_backw for i in range(n)], label=" all backw.")
#     plt.legend()
#     plt.savefig("patterns_compar_%f.png"% (fbar*n), dpi=300)
#     plt.close()
#     # plt.show()

### Next - take a few random samples from each n-block schedules and see how it performs on average
### for different n_pieces

for fbar in [0.001/n,0.005/n,0.01/n,0.02/n,0.04/n,0.08/n,0.12/n]:
    continue
    maxes = []
    for l in range(1,n):
        fit_current = 0
        for i in range(5):
            perm = np.random.choice(l,l,replace=False)#list(range(l-1,-1,-1))
            s = l*[True]#np.random.choice([True,False],l)
            sampling = block_perm_sampling(l,perm,s)
            fit_current += fitness_function(sampling,f,fbar,delta_f,dec)/5
        maxes.append(fit_current)

    sampling_jump_down = block_perm_sampling(2,[1,0],[True,True])
    sampling_jump_up = block_perm_sampling(2,[0,1],[False,False])

    fit_down = fitness_function(sampling_jump_down,f,fbar,delta_f,dec)
    fit_up = fitness_function(sampling_jump_up,f,fbar,delta_f,dec)

    fit_control = fitness_function(range(n),f,fbar,delta_f,dec)
    fit_backw = fitness_function(range(n-1,-1,-1),f,fbar,delta_f,dec)

    plt.plot(maxes, label="random n-block schedls")
    plt.plot([fit_down for i in range(n)], label="1 jump down")
    plt.plot([fit_up for i in range(n)], label="1 jump up")
    plt.plot([fit_control for i in range(n)], label="all forw.")
    plt.plot([fit_backw for i in range(n)], label=" all backw.")
    plt.legend()
    plt.savefig("random_nblocks_compar_allforw_%f.png"% (fbar*n), dpi=300)
    plt.close()
    # plt.show()

    ### Next - compare involutions to just random permutations


for fbar in [0.001/n,0.005/n,0.01/n,0.02/n,0.04/n,0.08/n,0.12/n]:
    continue
    maxes_inv = []
    maxes_norm = []
    for l in range(1,n):
        print(l)
        # if l%(n/10) == 0:
        #     print(100*l//n)
        fit_inv = 0
        fit_norm = 0
        for i in range(5):
            perm_inv = randinvolution(l)
            perm_norm = np.random.choice(l,l,replace=False)
            s = l*[True]
            sampling_inv = block_perm_sampling(l,perm_inv,s)
            sampling_norm = block_perm_sampling(l,perm_norm,s)
            fit_inv += fitness_function(sampling_inv,f,fbar,delta_f,dec)/5
            fit_norm += fitness_function(sampling_norm,f,fbar,delta_f,dec)/5
        maxes_inv.append(fit_inv)
        maxes_norm.append(fit_norm)

    sampling_jump_down = block_perm_sampling(2,[1,0],[True,True])
    sampling_jump_up = block_perm_sampling(2,[0,1],[False,False])

    fit_down = fitness_function(sampling_jump_down,f,fbar,delta_f,dec)
    fit_up = fitness_function(sampling_jump_up,f,fbar,delta_f,dec)

    fit_control = fitness_function(range(n),f,fbar,delta_f,dec)
    fit_backw = fitness_function(range(n-1,-1,-1),f,fbar,delta_f,dec)

    plt.plot(maxes_inv, label="random involutions")
    plt.plot(maxes_norm, label="random ordinary perm.")
    plt.plot([fit_down for i in range(n)], label="1 jump down")
    # plt.plot([fit_up for i in range(n)], label="1 jump up")
    plt.plot([fit_control for i in range(n)], label="all forw.")
    # plt.plot([fit_backw for i in range(n)], label=" all backw.")
    plt.legend()
    plt.savefig("random_nblocks_compar_invol_%f.png"% (fbar*n), dpi=300)
    plt.close()
    # plt.show()
