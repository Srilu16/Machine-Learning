import numpy as np

def dot_product(A,B):
    s=0
    for i in range(len(A)):
        s=s+A[i]*B[i]
    return s
def e_n(A):
    s=0
    for i in range(len(A)):
        s=s+A[i]**2
    return s**0.5
A=np.array([1,2,3])
B=np.array([4,5,6])

print("Dot Product using Formula:", dot_product(A, B))
print("Dot Product using Numpy:", np.dot(A, B))

print("Euclidean Norm using Formula:",e_n(A))
print("Euclidean Norm using Numpy:", np.linalg.norm(A))
