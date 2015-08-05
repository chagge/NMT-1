#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt

vlen = 1000
Wlen = 1000
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

def relu(x): return np.maximum(x, 0)
def sigmoid(x): return 1./(1+np.exp(-2*x))

##  RELU
#a = 1.
a = 1./(np.arctanh(np.sqrt(1./3))*np.sqrt(1-2/np.pi))
#Wtarget = 1./np.sqrt(1-2/np.pi)
Wtarget = np.arctanh(np.sqrt(1./3))
v = relu(np.random.normal(0, 1, size=vlen))
W = np.random.normal(0, Wtarget/np.sqrt(vlen/2.), size=(Wlen,vlen))
z = np.dot(W, v)
f = a*relu(z)
zstd = np.std(z)
fstd = np.std(f)
print 'relu(x)\texpected z: %.3f; observed z: %.3f; ratio: %.3f; std: %.3f' % (Wtarget, zstd, zstd/Wtarget, fstd)

##  TANH
a = 2.
Wtarget = np.arctanh(np.sqrt(1./3))
v = np.random.normal(0, 1, size=vlen)
W = np.random.normal(0, Wtarget/np.sqrt(vlen/1.), size=(Wlen,vlen))
z = np.dot(W, v)
f = a*np.tanh(z)
zstd = np.std(z)
fstd = np.std(f)
print 'tanh(x)\texpected z: %.3f; observed z: %.3f; ratio: %.3f; std: %.3f' % (Wtarget, zstd, zstd/Wtarget, fstd)

##  SIGMOID
a = 4.
Wtarget = np.arctanh(np.sqrt(1./3))
v = np.random.normal(0, 1, size=vlen)
W = np.random.normal(0, Wtarget/np.sqrt(vlen/1.), size=(Wlen,vlen))
z = np.dot(W, v)
f = a*sigmoid(z)
zstd = np.std(z)
fstd = np.std(f)
print 'sig(x)\texpected z: %.3f; observed z: %.3f; ratio: %.3f; std: %.3f' % (Wtarget, zstd, zstd/Wtarget, fstd)

##  GATE
a = 4*np.sqrt(5)/5
Wtarget = np.arctanh(np.sqrt(1./3))
v = np.random.normal(0, 1, size=Wlen)
v_i = np.random.normal(0, 1, size=vlen)
W_i = np.random.normal(0, Wtarget/np.sqrt(vlen/1.), size=(Wlen,vlen))
z = np.dot(W_i, v_i)
f = a * v * sigmoid(z)
zstd = np.std(z)
fstd = np.std(f)
print 'gate(x)\texpected z: %.3f; observed z: %.3f; ratio: %.3f; std: %.3f' % (Wtarget, zstd, zstd/Wtarget, fstd)

raw_input()
##  RNN
outs = [np.zeros(Wlen)]
for i in xrange(20):
  x = np.random.randn(vlen)
  y = outs[-1]
  W = np.random.normal(0, np.arctanh(np.sqrt(1./3))/np.sqrt(vlen), size=(Wlen,vlen))
  U = np.random.normal(0, np.arctanh(np.sqrt(1./3))/np.sqrt(Wlen), size=(Wlen,Wlen))
  outs.append(2*np.tanh((np.dot(W, x) + np.dot(U, y))/np.sqrt(2)))
  print np.std(outs[-1])
  
##  GRU
outs = [np.random.randn(Wlen)]
for i in xrange(20):
  x = np.random.randn(vlen)
  y = outs[-1]
  W_h = np.random.normal(0, np.arctanh(np.sqrt(1./3))/np.sqrt(vlen), size=(Wlen,vlen))
  U_h = np.random.normal(0, np.arctanh(np.sqrt(1./3))/np.sqrt(Wlen), size=(Wlen,Wlen))
  W_r = np.random.normal(0, np.arctanh(np.sqrt(1./3))/np.sqrt(vlen), size=(Wlen,vlen))
  U_r = np.random.normal(0, np.arctanh(np.sqrt(1./3))/np.sqrt(Wlen), size=(Wlen,Wlen))
  W_z = np.random.normal(0, np.arctanh(np.sqrt(1./3))/np.sqrt(vlen), size=(Wlen,vlen))
  U_z = np.random.normal(0, np.arctanh(np.sqrt(1./3))/np.sqrt(Wlen), size=(Wlen,Wlen))
  r = 4/np.sqrt(5)*sigmoid((np.dot(W_r, x) + np.dot(U_r, y))/np.sqrt(2))
  z = sigmoid((np.dot(W_z, x) + np.dot(U_z, y))/np.sqrt(2))
  h = 2*np.tanh((np.dot(W_h, x) + np.dot(U_h, r*y))/np.sqrt(2))
  outs.append(((1-z)*4/np.sqrt(5)*y + z*4/np.sqrt(5)*h)/np.sqrt(2))
  print np.std(r*y), np.std(z*4/np.sqrt(5)*h), np.std((1-z)*4/np.sqrt(5)*y), np.std(outs[-1])

##  LSTM
cells = [np.zeros(Wlen)]
outs = [np.zeros(Wlen)]
for i in xrange(20):
  x = np.random.randn(vlen)
  y = outs[-1]
  W_z = np.random.normal(0, np.arctanh(np.sqrt(1./3))/np.sqrt(vlen), size=(Wlen,vlen))
  U_z = np.random.normal(0, np.arctanh(np.sqrt(1./3))/np.sqrt(Wlen), size=(Wlen,Wlen))
  W_i = np.random.normal(0, np.arctanh(np.sqrt(1./3))/np.sqrt(vlen), size=(Wlen,vlen))
  U_i = np.random.normal(0, np.arctanh(np.sqrt(1./3))/np.sqrt(Wlen), size=(Wlen,Wlen))
  W_f = np.random.normal(0, np.arctanh(np.sqrt(1./3))/np.sqrt(vlen), size=(Wlen,vlen))
  U_f = np.random.normal(0, np.arctanh(np.sqrt(1./3))/np.sqrt(Wlen), size=(Wlen,Wlen))
  W_o = np.random.normal(0, np.arctanh(np.sqrt(1./3))/np.sqrt(vlen), size=(Wlen,vlen))
  U_o = np.random.normal(0, np.arctanh(np.sqrt(1./3))/np.sqrt(Wlen), size=(Wlen,Wlen))
  z = 2*np.tanh((np.dot(W_z, x) + np.dot(U_z, y))/np.sqrt(2))
  i = 4/np.sqrt(5)*sigmoid((np.dot(W_i, x) + np.dot(U_i, y))/np.sqrt(2))
  f = 4/np.sqrt(5)*sigmoid((np.dot(W_f, x) + np.dot(U_f, y))/np.sqrt(2))
  o = 4/np.sqrt(5)*sigmoid((np.dot(W_o, x) + np.dot(U_o, y))/np.sqrt(2))
  cells.append((i*z + f*cells[-1])/np.sqrt(2))
  h = 2*np.tanh(np.arctanh(np.sqrt(1./3))*cells[-1])
  outs.append(o*h)
  print np.std(z), np.std(i*z), np.std(f*cells[-2]), np.std(cells[-1]), np.std(h), np.std(outs[-1])
