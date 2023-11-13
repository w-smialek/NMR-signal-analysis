import numpy as np
import matplotlib.pyplot as plt

n = 2000
iters = 20
tresh = 50

plane = np.fromfunction(lambda a, b: ((a-n/2)*1j+(b-3*n/2))/n,(n,n))
values = np.zeros((n,n))
count = np.zeros((n,n))

for i in range(iters):
    values = values*values + plane
    count[abs(values)>tresh] = i
    values[abs(values)>tresh] = 0j

plt.imshow(count)
plt.savefig("mandelbrot.png",dpi=500)
plt.show()

###
### Julia
###

n = 2000
iters = 20
tresh = 5
c = 1j

plane = np.fromfunction(lambda a, b: ((a-n/2)*1j+(b-n/2))/(n/4),(n,n))
values = plane*plane + c
count = np.zeros((n,n))

for i in range(iters):
    values = values*values + c
    # plt.imshow(abs(values))
    # plt.show()
    count[abs(values)>tresh] = i
    values[abs(values)>tresh] = 0j

plt.imshow(count)
plt.savefig("julia.png",dpi=500)
plt.show()