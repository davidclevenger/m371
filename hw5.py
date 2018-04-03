import numpy as np




def jacobi(A,b):
    max_iter = 10

    D = np.array(np.zeros(len(A) * len(A)).reshape([len(A), len(A)])) #diagonal array
    R = np.array(np.zeros(len(A) * len(A)).reshape([len(A), len(A)])) #remainder matrix
    x = np.array([0,0,0,0])

    #set D and R
    for i in range(len(A)):
        D[i][i] = 1./A[i][i] #prepare for inverse

    for i in range(len(A)):
        for j in range(len(A)):
            if i != j:
                R[i][j] = A[i][j]

    for i in range(max_iter):
        x = np.matmul(D,b - np.matmul(R,x))
        print x

def gauss_seidel(A,b):
    max_iter = 10
    L = np.array(np.zeros(len(A) * len(A)).reshape([len(A), len(A)]))  # diagonal array
    U = np.array(np.zeros(len(A) * len(A)).reshape([len(A), len(A)]))  # remainder matrix
    x = np.array(np.zeros(4))

    # set L and U
    for i in range(len(A)):
        for j in range(len(A)):
            if i >= j:
                L[i][j] = A[i][j]
            else:
                U[i][j] = A[i][j]

    for i in range(max_iter):
        x = np.matmul(np.linalg.inv(L), b - np.matmul(U,x))
        print x


A = np.array([[7.,1.,-1.,2.],
              [1.,8.,0.,-2.],
              [-1.,0.,4.,-1.],
              [2.,-2.,-1.,6.]])

b = np.array([3.,-5.,4.,-3.])

if __name__ == "__main__":
    print 'i)\nJacobi\n'
    jacobi(A,b)
    print '\nGauss-Seidel\n'
    gauss_seidel(A,b)

    #print np.matmul(,b)



