import numpy as np
import signals as sig
import matplotlib.pyplot as plt

###
###
###

n = 512

dt1 = 0.001
dt2 = 0.001

shuffle = np.random.choice(n,n,replace=False)

def form(t1,t2,treal):
    f1,f2 = 0.05*treal/n, 0.05*treal/n
    # return sig.sinexp(t1,1,f1,n/2)*sig.sinexp(t2,1,f2,n/2)
    return np.sin(2*np.pi*t1*(f1))*np.sin(2*np.pi*t2*(f2))

signal_form_lin = [[form(t1,t2,treal) for t1 in range(n)] for t2, treal in zip(range(n),range(n))]
signal_form_perp = [[form(t1,t2,treal) for t1 in range(n)] for t2, treal in zip(range(n),shuffle)]
signal_lin = sig.Signal2D(signal_form_lin,dt1,dt2)
signal_perp = sig.Signal2D(signal_form_perp,dt1,dt2)

signal_lin.add_signal(sig.whitenoise_complex(0.1,(n,n)).real)
signal_perp.add_signal(sig.whitenoise_complex(0.1,(n,n)).real)

# signal_lin.plot2D()
signal_lin.freqplot2D()
signal_perp.freqplot2D()

###
### Interleaved acquisition
###