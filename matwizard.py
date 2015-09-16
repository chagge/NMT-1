 #!/usr/bin/env python
import numpy as np
import scipy.special as ss
from scipy import integrate
import sys

#=======================================================================
# The gaussian probability density function
def gauss(x, s):
  """"""
  
  return 1./np.sqrt(2*np.pi*s**2) * np.exp(-x**2/(2*s**2))

#=======================================================================
# Integral computation of the scaling factor of a soft function
def integral(func):
  """"""
  
  return np.sqrt(integrate.quad(lambda x: func(x)**2*gauss(x, np.arctanh(np.sqrt(1./3))), -100,100)[0])

#=======================================================================
# Creates the diagonal of a rectangular matrix
def diagonal(dims, dtype=np.float32):
  """ Creates the diagonal of a rectangular matrix """

  W = np.zeros(dims, dtype=dtype)
  m = (float(W.shape[1]-1) / float(W.shape[0]-1)) if float(W.shape[0]-1) > 0 else 1.
  
  for x in xrange(W.shape[0]):
    y = m*x
    W[x, np.floor(y)] = 1#(int(y)+1)-y
    W[x, min(np.ceil(y), W.shape[1]-1)]  = 1#y-int(y)

  return W

#=======================================================================
# Creates the tridiagonal of a rectangular matrix
def tridiagonal(dims, dtype=np.float32):
  """ Creates the tridiagonal of a rectangular matrix """

  W = np.zeros(dims, dtype=dtype)
  m = float(W.shape[0]-1) / float(W.shape[1]-1)

  for x in xrange(W.shape[1]):
    y = m*x
    if np.floor(y)-1 >= 0:
      W[np.floor(y)-1,x] = 1#(int(y)+1)-y
    if np.ceil(y)-1 >= 0:
      W[np.ceil(y)-1,x]  = 1#y-int(y)
    W[np.floor(y),x] = 1#(int(y)+1)-y
    W[np.ceil(y),x]  = 1#y-int(y)
    if np.floor(y)+1 < W.shape[0]:
      W[np.floor(y)+1,x] = 1#(int(y)+1)-y
    if np.ceil(y)+1 < W.shape[0]:
      W[np.ceil(y)+1,x]  = 1#y-int(y)

  return W

#=======================================================================
# Creates the upper triangle of a rectangular matrix
def triangle(dims, dtype=np.float32):
  """ Creates the upper triangle of a rectangular matrix """

  W = np.zeros(dims, dtype=dtype)
  m = float(W.shape[0]-1) / float(W.shape[1]-1)

  for x in xrange(W.shape[1]):
    y = m*x
    W[:np.floor(y),x] += 1
    W[np.floor(y),x] += 1
    W[np.ceil(y),x]  += y-int(y)

  return W

#***********************************************************************
# Creates a weight matrix optimized for a tanh/relu neural network
def matwizard(*dims, **kwargs):
  """ Creates a weight matrix optimized for a tanh/relu neural network """

  #---------------------------------------------------------------------
  # Keyword arguments
  if 'shape' in kwargs:
    shape = kwargs['shape']
  else:
    shape = 'rect'

  if 'spar' in kwargs:
    spar = kwargs['spar']
  else:
    spar = 1.
  
  if 'recur' in kwargs:
    recur = True
  else:
    recur = False
  
  if 'imput' in kwargs:
    imput = kwargs['imput']
  else:
    imput = ''

  if 'output' in kwargs:
    output = kwargs['output']
  else:
    output = ''

  #---------------------------------------------------------------------
  # Error checking/warnings
  #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  # Value Warning
  if shape != 'rect' and spar != 1.:
    sys.stderr.write('Warning: Sparsity is currently only defined for rectangular matrices\n')
  #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  # Value Error
  if shape not in ('rect', 'triang', 'tridiag', 'diag'):
    raise ValueError('Got bad input (%s) for keyword argument "shape".\nAcceptable values are "rect", "diag", "tridiag", and "triang".' % shape)
  #---------------------------------------------------------------------
  # Shape the matrix
  #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  # Rectangular matrix
  if shape == 'rect':
    mat = np.ones(dims)
    mat *= np.random.randn(*dims)
    if spar < 1:
      mask = np.random.binomial(1, spar, dims)
      mat *= mask
    elif spar > 1:
      mask = np.less(np.array([np.random.permutation(dims[1]) for i in xrange(dims[0])]), spar)
      mat *= mask
  #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  # Upper triangular matrix
  elif shape == 'triang':
    mat = triangle(dims)
    mat *= np.random.randn(*dims)
  #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  # Tridiagonal matrix
  elif shape == 'tridiag':
    mat = tridiagonal(dims)
  #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  # Diagonal matrix
  elif shape == 'diag':
    mat = diagonal(dims)
    
  #---------------------------------------------------------------------
  # Normalize by row
  if len(dims) > 1:
    mat /= np.sqrt(np.sum(np.not_equal(mat, 0), axis=1, keepdims=True))

  #---------------------------------------------------------------------
  # Set the standard deviation
  if output in ('softsign', 'tanh'):
    mat *= np.arctanh(np.sqrt(1./3))
    if imput == output:
      mat /= integral(lambda x: np.tanh(x))
  elif output in ('softstep', 'sig', 'sigmoid'):
    mat *= np.arctanh(np.sqrt(1./3))
    if imput == output:
      mat /= integral(lambda x: (np.tanh(x)+1)/2)
  elif output in ('softabs',):
    mat *= np.arctanh(np.sqrt(1./3))
    if imput == output:
      mat /= integral(lambda x: np.log(2*np.cosh(x)))
  elif output in ('softpo', 'softplus'):
    mat *= np.arctanh(np.sqrt(1./3))
    if imput == output:
      mat /= integral(lambda x: (np.log(2*np.cosh(x))+x)/2)
  elif output in ('softbool', 'tub'):
    mat *= np.arctanh(np.sqrt(1./3))
    if imput == output:
      mat /= integral(lambda x: np.tanh(x)**2)
  elif output in ('softpos',):
    mat *= np.arctanh(np.sqrt(1./3))
    if imput == output:
      mat /= integral(lambda x: ((np.tanh(x)+1)/2)**2)
      
  elif output in ('sharpsign', 'hardtanh'):
    if imput == output:
      mat *= 1/ss.erf(2/np.sqrt(2))
    else: mat *= .5
  elif output in ('sharpstep',):
    if imput == output:
      mat *= .875/ss.erf(2/np.sqrt(2))
    else: mat *= .5
  elif output in ('sharpabs', 'abs'):
    if imput == output:
      mat *= 1.
    else: mat *= .5
  elif output in ('sharppo', 'relu'):
    if imput == output:
      mat *= np.sqrt(2)
    else: mat *= .5
  elif output in ('sharpbool',):
    if imput == output:
      mat *= 1/ss.erf(2/np.sqrt(2))
    else: mat *= .5
  elif output in ('sharppos',):
    if imput == output:
      mat *= 1/ss.erf(2/np.sqrt(2))
    else: mat *=.5
    
  elif output in ('softmax',):
    mat *= 0

  if recur:
    mat /= np.sqrt(2)
  #---------------------------------------------------------------------
  # Return the matrix
  return mat.astype('float32')

#***********************************************************************
# Test it out
if __name__ == '__main__':
  """"""

  import matplotlib.pyplot as plt
  def relu(x): return np.maximum(x, 0)
  #print matwizard(4,5)
  #print matwizard(4,5, shape='diag')
  #print matwizard(4,5, shape='tridiag')
  #print matwizard(4,5, shape='triang')

  vlen = 2000.
  Wlen = 5000.
  print np.std(np.random.randn(vlen) + np.random.randn(vlen))

  h = np.zeros(Wlen)
  W = matwizard(Wlen, vlen, output='tanh', recur=True)
  U = matwizard(Wlen, Wlen, output='tanh', imput='tanh', recur=True)
  for t in xrange(10):
    v = np.random.randn(vlen)
    w = np.dot(W, v)
    u = np.dot(U, h)
    a = w + u
    h = np.tanh(a)
    print '==='
    print np.std(w), np.std(u), np.std(a)
    print np.std(np.tanh(w)), np.std(np.tanh(u)), np.std(h)
    print '---'
    print np.mean(w), np.mean(u), np.mean(a)
    print np.mean(np.tanh(w)), np.mean(np.tanh(u)), np.mean(h)