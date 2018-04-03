import numpy as np

#question 9.A (jacobi() and gauss_seidel())
def jacobi(A,b):
    max_iter = 10

    D = np.array(np.zeros(len(A) * len(A)).reshape([len(A), len(A)])) #diagonal array
    R = np.array(np.zeros(len(A) * len(A)).reshape([len(A), len(A)])) #remainder matrix
    x = np.array(np.zeros(len(A)))

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
    L = np.array(np.zeros(len(A) * len(A)).reshape([len(A), len(A)]))  # lower matrix
    U = np.array(np.zeros(len(A) * len(A)).reshape([len(A), len(A)]))  # strict upper matrix
    x = np.array(np.zeros(len(A)))

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

#question 10.A (least_squares(), error())
def least_squares(x,y,n):
    A = np.array(np.zeros(len(x)*(n+1)).reshape([len(x), (n+1)]))

    for i in range(len(A)):
        for j in range(n+1):
            A[i][j] = x[i]**(n-j)

    coefs = np.matmul(np.linalg.inv( np.matmul(np.transpose(A),A) ),np.matmul(np.transpose(A),y))
    return coefs

def error(x,y,coefs):
    predicted = np.zeros(len(y))
    n = len(coefs)

    for i in range(len(x)):
        pred = 0 #predicted value
        for j in range(len(coefs)):
            pred += coefs[j]*(x[i]**(n-j-1))
        predicted[i] = pred

    delta = y - predicted

    sse = 0
    for i in range(len(delta)):
        sse += delta[i]**2

    print 'error=', delta
    print 'SSE=', sse



if __name__ == "__main__":

    #set up  1st system
    A = np.array([[7., 1., -1., 2.],
                  [1., 8., 0., -2.],
                  [-1., 0., 4., -1.],
                  [2., -2., -1., 6.]])

    b = np.array([3., -5., 4., -3.])

    print 'Q 9.b\n\ni)\nJacobi\n'
    jacobi(A,b)
    print '\nGauss-Seidel\n'
    gauss_seidel(A,b)

    #set up 2nd system
    A = np.zeros(50*50).reshape([50,50])
    b = np.arange(1,51,1)

    for i in range(len(A)):
        for j in range(len(A)):
            if i == j:
                A[i][j] = 5
            elif i == j+1:
                A[i][j] = -1
            elif j == i+1:
                A[i][j] = -1
            else:
                A[i][j] = 0

    print '\nii)\nJacobi\n'
    jacobi(A, b)
    print '\nGauss-Seidel\n'
    gauss_seidel(A, b)

    x = [0.,0.2,0.8,1.,1.2,1.9,2.0,2.1,2.95,3.0]
    y = [0.01,0.22,0.76,1.03,1.18,1.94,2.01,2.08,2.9,2.95]

    coefs = least_squares(x,y,1)
    print '\n\n\nQ 10.b\n\ncoefficients', coefs
    error(x,y,coefs)

    x = [1,2,3,4,5,6,7]
    y = [2.31,2.01,1.8,1.66,1.55,1.47,1.41]

    coefs = least_squares(x,y,2)
    print '\ncoefficients', coefs
    error(x,y,coefs)