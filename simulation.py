#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt

vlen = 150
Wlen = 300
i = 0
args = sys.argv[1:]
while i < len(args):
  flag = args.pop(0)
  if flag == '-v':
    arg = args.pop(0)
    vlen = int(arg)
  elif flag == '-W':
    arg = args.pop(0)
    Wlen = int(arg)
  i += 1

vstd = .5
def relu(x): return np.maximum(x, 0)

Wvstd = np.arctanh(np.sqrt(1./3))
v = np.random.normal(0, vstd, size=vlen)
W = np.random.normal(0, Wvstd/vstd*np.sqrt(1./vlen), size=(Wlen,vlen))
z = np.dot(W, v)
f = np.tanh(z)
zstd = np.std(z)
fstd = np.std(f)
print 'tanh(x)'
print 'z)\texpected: %.3f; observed: %.3f; ratio: %.3f' % (Wvstd, zstd, zstd/Wvstd)
print 'f)\texpected: %.3f; observed: %.3f; ratio: %.3f' % (vstd, fstd, fstd/vstd)

Wvstd = np.sqrt(2*(1-2/np.pi))
v = relu(np.random.normal(0, vstd, size=vlen))
W = np.random.normal(0, Wvstd/vstd*np.sqrt(2./vlen), size=(Wlen,vlen))
z = np.dot(W, v)
f = relu(z)
zstd = np.std(z)
fstd = np.std(f)
print 'relu(x)'
print 'z)\texpected: %.3f; observed: %.3f; ratio: %.3f' % (Wvstd, zstd, zstd/Wvstd)
print 'f)\texpected: %.3f; observed: %.3f; ratio: %.3f' % (vstd, fstd, fstd/vstd)

Wvstd = .5*np.arctanh(np.sqrt(1./3)) + .5*np.sqrt(2*(1-2/np.pi))
v = np.random.normal(0, vstd, size=vlen)
W = np.random.normal(0, Wvstd/vstd*np.sqrt(1./vlen), size=(Wlen,vlen))
z = np.dot(W, v)
f = .5*np.tanh(z) + .5*relu(z)
zstd = np.std(z)
fstd = np.std(f)
print 'both(x)'
print 'z)\texpected: %.3f; observed: %.3f; ratio: %.3f' % (Wvstd, zstd, zstd/Wvstd)
print 'f)\texpected: %.3f; observed: %.3f; ratio: %.3f' % (vstd, fstd, fstd/vstd)

#xs = np.array(np.arange(-4,4,.1))
#def n(x): return 1./(Wvstd*np.sqrt(np.pi))*np.exp(-x**2 / (2*Wvstd**2))
#
#plt.figure()
#def f_(x): return 1/(1+np.exp(-2*x))
#def g_(x): return 2*(f_(x)*(1-f_(x)))
#def h_(x): return 2*(g_(x)*(1-f_(x)) - f_(x)*g_(x))
#def f(x): return np.tanh(x)
#def g(x): return 1-f(x)**2
#def h(x): return -2*f(x)*g(x)
#plt.plot(xs, f_(xs), color='b')
#plt.plot(xs, g_(xs), color='r')
#plt.plot(xs, h_(xs), color='g')
#plt.plot(xs, n(xs), color='k', lw=2)
#plt.plot(xs, f(xs), color='b', ls='--')
#plt.plot(xs, g(xs), color='r', ls='--')
#plt.plot(xs, h(xs), color='g', ls='--')
#plt.grid()
#plt.show()
#