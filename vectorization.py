import numpy as np

n=2

x = np.random.randint(0,10,(n,n))
print(x)

# xv = np.array([x[(i-i%n)//n,i%n] for i in range(n**2)])
# print(xv)

Tns = np.random.randint(0,10,(3,2,2,3))
print(Tns)

# Tnsv = np.array([[Tns[j%n,(j-j%n)//n,(i-i%n)//n,i%n] for i in range(n**2)] for j in range(n**2)])
# print(Tnsv)

# print(Tns[0,0,:,:])
# print(Tns[0,0,:,:]*x)
# print(np.sum(Tns[0,0,:,:]*x))
# s = np.array([[np.sum(Tns[t1,t2,:,:]*x) for t1 in range(n)] for t2 in range(n)])
# print(s)

# sv = np.matmul(Tnsv,xv)
# print(sv)

def Vectorization(matrix):
    n,m = np.shape(matrix)
    return np.array([[matrix[i//m,i%m] for i in range(n*m)]])

print(Vectorization(x))

def Matricization(tensor):
    a,b,c,d = np.shape(tensor)
    return np.array([[tensor[j%a,j//a,i//d,i%d] for i in range(c*d)] for j in range(a*b)])

print(Matricization(Tns))