import numpy as np
import matplotlib.pyplot as plt
from random import randrange

N = 1000

def signal(a,f,fb,dec_inv,sampling):
    try:
        sig = np.zeros(N).astype(complex)
        for a0,f0,fb0,dec_inv0 in zip(a,f,fb,dec_inv):
            sig += np.array([a0*np.exp(1j*2*np.pi*(f0+fb0*tau)*t-t*dec_inv0) for t,tau in zip(range(N),sampling)])
    except:
        sig = np.array([a*np.exp(1j*2*np.pi*(f+fb*tau)*t-t*dec_inv) for tau,t in zip(range(N),sampling)])
    return sig

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def block_perm_sampling(n_pieces, perm, signs = None):
    if signs == None:
        signs = [True]*n_pieces
    pieces_forw = [list(i) for i in list(split(range(N), n_pieces))]
    pieces_backw = [list(i) for i in list(split(range(N-1,-1,-1), n_pieces))]
    pieces_backw.reverse()
    sampling = []
    for p,s in zip(perm,signs):
        if s:
            sampling += pieces_forw[p]
        else:
            sampling += pieces_backw[p]
    return sampling

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

def randpermutation(n):
    return np.random.choice(n,n,replace=False)

# a = (1,1)
# f = (0.01,0.3)
# fb = (0.03/N,0.05/N)
# dec = (10*N,10*N)
a = 1
f = 0.3
fb = 0.05/N
dec_inv = 0

sampling = block_perm_sampling(6,[3,4,5,0,1,2])

sig0 = signal(a,f,fb,dec_inv,sampling)

for fb in np.array([0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1])/N:
    maxes = []
    for r in range(1,200):
        sampling = block_perm_sampling(r,range(r-1,-1,-1))
        sig_sampled = signal(a,f,fb,dec_inv,sampling)
        sig_sampledft = np.fft.fft(sig_sampled)
        # plt.plot(abs(sig_sampledft))
        # plt.show()
        now_max = np.max(abs(sig_sampledft))
        maxes.append(now_max)
    plt.plot(maxes)
    plt.show()

###
###
###

# perm_maxes = []

# for r in range(1,14):
#     perm_max = (0,0)
#     for p in Involutions(r):
#         sampling_perm = block_perm_sampling(r,p)

#         sig_perm = signal(a,f,fb,dec_inv,sampling_perm)

#         sig_permft = np.fft.fft(sig_perm)

#         now_max = np.max(abs(sig_permft))
#         if now_max > perm_max[0]:
#             perm_max = (now_max,sampling_perm)

#         # if i%1000 == 0:
#         #     print(i)
#     print(invo_count(r))
#     print(r)
#     perm_maxes.append(perm_max)

# vals = [i[0] for i in perm_maxes]
# perms = [i[1] for i in perm_maxes]

# plt.plot(vals)
# plt.show()
# for p in perms:
#     plt.plot(p)
#     plt.show()

###
### Comparing involutions on average
###
'''
perm_maxes = []
invol_maxes = []

ratios = []

for r in range(1,50):
    for i in range(1000):
        sampling_perm = block_perm_sampling(r,randpermutation(r))
        sampling_invol = block_perm_sampling(r,randinvolution(r))

        sig_perm = signal(a,f,fb,dec_inv,sampling_perm)
        sig_invol = signal(a,f,fb,dec_inv,sampling_invol)

        sig_permft = np.fft.fft(sig_perm)
        sig_involft = np.fft.fft(sig_invol)
        
        perm_maxes.append(np.max(abs(sig_permft)))
        invol_maxes.append(np.max(abs(sig_involft)))

        # if i%100 == 0:
        #     print(i)
    print(r)
    ratios.append(np.average(invol_maxes)/np.average(perm_maxes))
plt.plot(ratios)
plt.show()
'''