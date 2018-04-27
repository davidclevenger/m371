import numpy as np
from scipy.integrate import quadrature

def f1(x):
    return np.log(x)

def f2(x):
    return 1 / (1+(x*x))

def midpoint(f,a,b,n):
    dx = float(b-a)/n
    sum = 0
    for i in range(n):
        sum += f((a + dx/2.0) + i*dx)
    sum *= dx

    return sum

def trapezoid(f,a,b,n):
    dx = float(b-a)/n
    sum = 0.5*f(a) + 0.5*f(b)
    for i in range(1,n):
        sum += f(a + i*dx)
    sum *= dx

    return sum

def simpson(f,a,b,n):
    dx = float(b-a)/n
    k = 0.0
    x = a + dx
    for i in range(1, n/2 + 1):
        k += 4*f(x)
        x += 2*dx

    x = a + 2*dx
    for i in range(1, n/2):
        k += 2*f(x)
        x += 2*dx

    return (dx/3)*(f(a) + f(b) + k)

def gaussquad(f,a,b,n):
    out = quadrature(f,a,b,maxiter=n)
    return out[0]

def rk4(f,t,x,h,n):
    ta = t
    for j in range(1,n):
        k1 = h*f(t, x)
        k2 = h*f(t+(1./2.)*h, x+(1./2.)*k1)
        k3 = h*f(t+(1./2.)*h, x+(1./2.)*k2)
        k4 = h*f(t+h, x+k3)
        x = x + (1./6.)*(k1 + 2*k2 + 2*k3 + k4)
        t = ta + j*h
        print 'j:', j
        print 't:', t
        print 'x:', x



N = 4

#b.i)
EXACT1 = 0.386294361119891
EXACT2 = 0.785398163397448
a = midpoint(f1,1,2,N)
b = trapezoid(f1,1,2,N)
c = simpson(f1,1,2,N)
d = gaussquad(f1,1,2,N)

print 'err: f1'
print 'mid ', EXACT1 - a
print 'trap ', EXACT1 - b
print 'simp ', EXACT1 - c
print '2pGauss ', EXACT1 - d
print '\n'

a = midpoint(f2,0,1,N)
b = trapezoid(f2,0,1,N)
c = simpson(f2,0,1,N)
d = gaussquad(f2,0,1,N)
print 'err: f2'
print 'mid ', EXACT2 - a
print 'trap ', EXACT2 - b
print 'simp ', EXACT2 - c
print '2pGauss ', EXACT2 - d
print'\n\n'

#c
a = midpoint(f1,1,1.5,N) + midpoint(f1,1.5,2,N)
b = trapezoid(f1,1,1.5,N) + trapezoid(f1,1.5,2,N)
c = simpson(f1,1,1.5,N) + simpson(f1,1.5,2,N)
d = gaussquad(f1,1,1.5,N) + gaussquad(f1,1.5,2,N)

print'\n splitting integrals into 2 integrals\n'
print 'err: f1'
print 'mid ', EXACT1 - a
print 'trap ', EXACT1 - b
print 'simp ', EXACT1 - c
print '2pGauss ', EXACT1 - d
print '\n'

a = midpoint(f2,0,0.5,N) + midpoint(f2,0.5,1,N)
b = trapezoid(f2,0,0.5,N) + trapezoid(f2,0.5,1,N)
c = simpson(f2,0,0.5,N) + simpson(f2,0.5,1,N)
d = gaussquad(f2,0,0.5,N) + gaussquad(f2,0.5,1,N)
print 'err: f2'
print 'mid ', EXACT2 - a
print 'trap ', EXACT2 - b
print 'simp ', EXACT2 - c
print '2pGauss ', EXACT2 - d