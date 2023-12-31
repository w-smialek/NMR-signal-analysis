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

fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot()
ax1.matshow(abs(fts))
plt.xlabel("frequency")
plt.ylabel("real time")
ax1.tick_params(labeltop=True,top=True)
plt.savefig("ft_series_1d.png",dpi=300)
plt.show()

rt = np.array([[np.sum([fts[s,int((om1+om2*s))%n] for s in range(n)]) for om1 in range(n)]for om2 in np.linspace(0,1,100)])

fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot()
ax1.matshow(abs(rt))
plt.xlabel("frequency")
plt.ylabel("frequency change rate")
ax1.tick_params(labeltop=True,top=True)
plt.yticks([0,20,40,60,80],[0,0.2,0.4,0.6,0.8])
plt.savefig("rt_1d.png",dpi=300)
plt.show()

###
### 2D
###

n = 40

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

rt_3d = np.array([[[np.sum([sig_3d[int((f1+n*om*s))%n,f2,s] for s in range(n_samples)]) for om in np.linspace(0,0.01,40)]for f2 in range(n)] for f1 in range(n)])
print(np.max(rt_3d))

import plotly.graph_objects as go
import numpy as np

values=abs(rt_3d)

X, Y, Z = np.mgrid[ 0:n, 0:n, 0:0.4:40j]

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

fig.write_html("rt_density.html")
fig.show()

values=abs(sig_3d)

X, Y, Z = np.mgrid[ 0:n, 0:n, 0:n]

fig = go.Figure(data=go.Isosurface(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=values.flatten(),
    opacity=0.2,
    isomin=0.5,
    isomax=1700,
    surface_count=10,
    caps=dict(x_show=False, y_show=False)
    ))

fig.write_html("ft_series_2d.html")
fig.show()