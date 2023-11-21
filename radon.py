import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import signals as sig

n = 100
f0 = 0.5
frate = 0.3

form = np.array([[np.exp(2j*pi*(f0 + frate*tr/n)*t) for t in range(n)] for tr in range(n)])
sig1 =  sig.Signal(form)
# sig1.plot("time")
# plt.show()

fts = np.array([np.fft.fft(form[i,:]) for i in range(n)])
# plt.matshow(abs(fts))
# plt.show()

rt = np.array([[np.sum([fts[s,int((om1+om2*s))%n] for s in range(n)]) for om1 in range(n)]for om2 in np.linspace(0,1,100)])

# plt.matshow(abs(rt))
# plt.show()

###
### 2D
###

n = 40

# def non_stationary_frequency(t1,t2,treal):
#     f1, f2 = 0.1 + 0.1*treal/n, 0.2
#     f3, f4 = 0.3 + 0.2*treal/n, 0.6
#     f5, f6 = 0.5 + 0.3*treal/n, 0.5
#     tau1, tau2 = n/2, n/2
#     return np.exp(2j*pi*(f1*t1+f2*t2)-t1/tau1) + 0.5*np.exp(2j*pi*(f3*t1+f4*t2)-t1/tau2) + 0.25*np.exp(2j*pi*(f5*t1+f6*t2)-t1/tau2)

def non_stationary_frequency(t1,t2,treal):
    f1, f2 = 0.25 + 0.005*treal/n, 0.5
    return np.exp(2j*pi*(f1*t1+f2*t2))


def snapshot_2d(n, F, t_real, t_indir):
    '''n - number of direct time datapoints \n
    treal - range of real time values \n
    t_indir - corresponding indirect time values'''
    form = np.zeros((n,n)).astype(complex)
    for t2, tr in zip(t_indir, t_real):
        form[t2,:] = np.array([F(t1,t2,tr) for t1 in range(n)])
    return form[t_indir]

n_samples = 40
sig_3d = []

for i in range(n_samples):
    snapshot_ft = np.fft.fft2(snapshot_2d(n,non_stationary_frequency,range(i*n,(i+1)*n),range(n)))
    sig_3d.append(snapshot_ft)

sig_3d = np.array(sig_3d).swapaxes(0,2)
print(sig_3d.shape)


# for a in range(n):
#     plt.matshow(sig_3d[:,:,a].real)
#     plt.show()
#     plt.plot(sig_3d[a,19,:].real)
#     plt.show()
#     plt.plot(sig_3d[a,20,:].real)
#     plt.show()
#     plt.plot(sig_3d[a,21,:].real)
#     plt.show()

# print(np.sum([sig_3d[s,20,3] for s in range(n)]))
# for i in range(n):
    # print(i)
    # plt.plot(sig_3d[i,10,:])
    # plt.show()

rt_3d = np.array([[[np.sum([sig_3d[int((f1+n*om*s))%n,f2,s] for s in range(n_samples)]) for om in np.linspace(0,0.01,40)]for f2 in range(n)] for f1 in range(n)])
# rt_3d = np.array([[[np.exp(-(f1-(10-0.5*om))**2-(f2-20)**2) for om in range(n_samples)]for f2 in range(n)] for f1 in range(n)])
print(np.max(rt_3d))

import plotly.graph_objects as go
import numpy as np

# #X, Y, Z = np.mgrid[-5:5:40j, -5:5:40j, -5:5:40j]
# #X, Y, Z = np.mgrid[ 0.1:5:40j, 0.1:5:40j, 0.1:5:40j]
# X, Y, Z = np.mgrid[ 0.01:2:40j, 0.01:2:40j, 0.01:2:40j]
# #
# # pascal dagsi surfaces
# #values=(X**(Y*Z)) + (Y**(X*Z)) + (Z**(X*Y))
# #values=(X**(Y+Z)) + (Y**(X+Z)) + (Z**(X+Y))
# #values=(X**(Y+Z)) * (Y**(X+Z)) * (Z**(X+Y))
# values=(X**(Y*Z)) * (Y**(X*Z)) * (Z**(X*Y))

# fig = go.Figure(data=go.Isosurface(
#     x=X.flatten(),
#     y=Y.flatten(),
#     z=Z.flatten(),
#     value=values.flatten(),
#     opacity=0.2,
#     isomin=0.5,
#     isomax=25,
#     surface_count=10,
#     caps=dict(x_show=False, y_show=False)
#     ))
# fig.write_html("density.html")
# fig.show()

X, Y, Z = np.mgrid[ 0:n, 0:n, 0:n]


values=abs(rt_3d)

fig = go.Figure(data=go.Isosurface(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=values.flatten(),
    opacity=0.2,
    isomin=0.5,
    isomax=35000,
    surface_count=10,
    caps=dict(x_show=False, y_show=False)
    ))
#
fig.write_html("density.html")
fig.show()