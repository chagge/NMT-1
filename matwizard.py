 #!/usr/bin/env python
import numpy as np
import sys

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

  if 'relu_input' in kwargs:
    relu_input = kwargs['relu_input']
  else:
    relu_input = False

  if 'relu_output' in kwargs:
    relu_output = kwargs['relu_output']
  else:
    relu_output = False

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
    mat *= np.random.normal(0,1, size=dims)
    if spar < 1:
      mask = np.random.binomial(1, spar, dims)
      mat *= mask
    elif spar > 1:
      mask = np.array([np.random.permutation(dims[1]) for i in xrange(dims[0])]).less(spar)
      mat *= mask
  #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  # Upper triangular matrix
  elif shape == 'triang':
    mat = triangle(dims)
    mat *= np.random.normal(0,1, size=dims)
  #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  # Tridiagonal matrix
  elif shape == 'tridiag':
    mat = tridiagonal(dims)
  #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  # Diagonal matrix
  elif shape == 'diag':
    mat = diagonal(dims)
  nonzero_elts = np.sum(np.not_equal(mat, 0), axis=1, keepdims=True)

  #---------------------------------------------------------------------
  # Generate random values
  if relu_output:
    mat *= (1./np.sqrt(1-2/np.pi))
  else:
    mat *= (np.arctanh(np.sqrt(1./3)))

  #---------------------------------------------------------------------
  # Account for the size of the input vector
  mat /= np.sqrt(nonzero_elts)
  if relu_input:
    mat *= np.sqrt(2.)

  #---------------------------------------------------------------------
  # Return the matrix
  return mat.astype(np.float32)

#***********************************************************************
# Test it out
if __name__ == '__main__':
  """"""

  import matplotlib.pyplot as plt
  def relu(x): return np.maximum(x, 0)
  def gauss(x, m, s): return 1./(s*np.sqrt(2*np.pi))*np.exp(-(x-m)**2/(2*s**2))
  print matwizard(4,5)
  print matwizard(4,5, shape='diag')
  print matwizard(4,5, shape='tridiag')
  print matwizard(4,5, shape='triang')
  print matwizard(4,5, relu_input=True)
  print matwizard(4,5, relu_output=True)

  vlen = 3000.
  Wlen = 3000.

  print 'Testing rect:'
  W = matwizard(Wlen, vlen)
  r = np.random.randn(vlen)
  dot1 = W.dot(r)
  print np.mean(dot1), np.std(dot1)
  print np.std(2*np.tanh(dot1))
  
  print 'Testing diag:'
  W = matwizard(Wlen, vlen, shape='diag')
  r = np.random.randn(vlen)
  dot2 = W.dot(r)
  print np.mean(dot2), np.std(dot2)
  print np.std(2*np.tanh(dot2))

  print 'Testing tridiag:'
  W = matwizard(Wlen, vlen, shape='tridiag')
  r = np.random.randn(vlen)
  dot3 = W.dot(r)
  print np.mean(dot3), np.std(dot3)
  print np.std(2*np.tanh(dot3))

  print 'Testing triang:'
  W = matwizard(Wlen, vlen, shape='triang')
  r = np.random.randn(vlen)
  dot4 = W.dot(r)
  print np.mean(dot4), np.std(dot4)
  print np.std(2*np.tanh(dot4))

  plt.figure()
  plt.hist(dot1, 50, normed=1, alpha=1, label='rect')
  plt.hist(dot4, 50, normed=1, alpha=.75, label='triang')
  plt.hist(dot3, 50, normed=1, alpha=.5, label='tridiag')
  plt.hist(dot2, 50, normed=1, alpha=.25, label='diag')
  x = np.arange(-3,3,.01)
  plt.plot(x, gauss(x,0,np.arctanh(np.sqrt(1./3))), color='k', lw=1.5, ls='--')
  plt.legend()
  plt.grid()
  plt.show()
  #---------------------------------------------------------------------
  print 'Testing relu:'
  W = matwizard(Wlen, vlen)
  r = np.random.randn(vlen)
  dot5 = W.dot(r)
  print np.mean(dot5), np.std(dot5)
  func5 = relu(dot5)/(np.arctanh(np.sqrt(1./3))*np.sqrt(1-2/np.pi))
  print np.std(func5)

  print 'Testing relu_input:'
  W = matwizard(Wlen, vlen, relu_input=True)
  r = relu(np.random.randn(vlen))
  dot6 = W.dot(r)
  print np.mean(dot6), np.std(dot6)
  func6 = 2*np.tanh(dot6)
  print np.std(func6)

  print 'Testing relu_output:'
  W = matwizard(Wlen, vlen, relu_output=True)
  r = np.random.randn(vlen)
  dot7 = W.dot(r)
  print np.mean(dot7), np.std(dot7)
  func7 = relu(dot7)
  print np.std(func7)

  plt.figure()
  plt.hist(dot5, 50, normed=1, alpha=1, label='relu')
  plt.hist(dot6, 50, normed=1, alpha=.67, label='relu_input')
  plt.hist(dot7, 50, normed=1, alpha=.33, label='relu_output')
  x = np.arange(-6,6,.01)
  plt.plot(x, gauss(x,0,np.arctanh(np.sqrt(1./3))), color='k', lw=1.5, ls='--')
  plt.plot(x, gauss(x,0,1/(np.arctanh(np.sqrt(1./3)))), color='k', ls='--')
  plt.legend()
  plt.grid()
  plt.show()
