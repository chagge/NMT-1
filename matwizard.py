 #!/usr/bin/env python
import numpy as np

#=======================================================================
# Creates the diagonal of a rectangular matrix
def diagonal(dims, dtype=np.float32):

  W = np.zeros(dims, dtype=dtype)
  m = (float(W.shape[0]-1) / float(W.shape[1]-1)) if float(W.shape[1]-1) > 0 else 1.
  for x in xrange(W.shape[1]):
    y = m*x
    W[np.floor(y),x] += (int(y)+1)-y
    W[min(np.ceil(y), W.shape[0]-1),x]  += y-int(y)
  return W

#=======================================================================
# Creates the tridiagonal of a rectangular matrix
def tridiagonal(dims, dtype=np.float32):

  W = np.zeros(dims, dtype=dtype)
  m = float(W.shape[0]-1) / float(W.shape[1]-1)
  for x in xrange(W.shape[1]):
    y = m*x
    if np.floor(y)-1 >= 0:
      W[np.floor(y)-1,x] += (int(y)+1)-y
    if np.ceil(y)-1 >= 0:
      W[np.ceil(y)-1,x]  += y-int(y)
    W[np.floor(y),x] += (int(y)+1)-y
    W[np.ceil(y),x]  += y-int(y)
    if np.floor(y)+1 < W.shape[0]:
      W[np.floor(y)+1,x] += (int(y)+1)-y
    if np.ceil(y)+1 < W.shape[0]:
      W[np.ceil(y)+1,x]  += y-int(y)
  return W

#=======================================================================
# Creates the upper triangle of a rectangular matrix
def triangle(dims, dtype=np.float32):

  W = np.zeros(dims, dtype=dtype)
  m = float(W.shape[0]-1) / float(W.shape[1]-1)
  for x in xrange(W.shape[1]):
    y = m*x
    W[:np.floor(y),x] += 1
    W[np.floor(y),x] += 1
    W[np.ceil(y),x]  += y-int(y)
  return W

#=======================================================================
# Randomly initializes a weight matrix
def matwizard(*dims, **kwargs):
  
  #---------------------------------------------------------------------
  # Arguments
  if len(dims) < 1:
    raise ValueError('Matrix must have at least one dimension')
  if isinstance(dims[0], (tuple, list)):
    dims = dims[0]

  #---------------------------------------------------------------------
  # Keyword arguments
  if 'dtype' in kwargs:
    dtype = kwargs['dtype']
  else:
    dtype = 'float32'

  if 'shape' in kwargs:
    shape=kwargs['shape']
  else:
    shape='1'

  if 'dist' in kwargs:
    dist = kwargs['dist']
  else:
    dist = 'normal'

  if 'sign' in kwargs:
    sign = kwargs['sign']
  else:
    sign = 0

  if 'scale' in kwargs:
    scale = kwargs['scale']
  else:
    scale = 1.

  if 'sparse' in kwargs:
    sparse = kwargs['sparse']
  else:
    sparse = 1.

  if 'normalize' in kwargs:
    normalize = kwargs['normalize']
  else:
    normalize = False

  if 'center' in kwargs:
    center = kwargs['center']
  else:
    center = False

  #---------------------------------------------------------------------
  # Initialize the matrix
  
  # Shape the matrix
  if shape in ('1'):
    W = np.ones(dims, dtype=dtype)
  elif shape in ('0'):
    W = np.zeros(dims, dtype=dtype)
  elif shape in ('I'):
    W = diagonal(dims, dtype=dtype)
  elif shape in ('3I'):
    W = tridiagonal(dims, dtype=dtype)
  elif shape in ('^'):
    W = triangle(dims, dtype=dtype)
  else:
    raise ValueError('%s is not currently a valid keyword for shape' % shape)

  # Value the matrix
  if dist == 'normal':
    W *= np.random.standard_normal(dims)
  elif dist == 'uniform':
    W *= np.random.uniform(-1,1,dims)
  elif dist == 'uninomial':
    pass
  else:
    raise ValueError('%s is not currently a valid keyword for dist' % dist)

  # Sign the matrix
  if sign > 0:
    W = np.abs(W)
  elif sign < 0:
    W = -np.abs(W)

  # Scale the matrix
  if isinstance(scale, basestring):
    if scale in ('in'):
      scale = np.sum(dims[:-1])**-(1./2)
    elif scale in ('out'):
      scale = dims[-1]**-(1./2)
    elif scale in ('in+out'):
      scale = np.sum(dims)**-2
  W *= scale

  # Sparse the matrix
  W *= np.random.binomial(1, sparse, W.shape)

  # Center the matrix
  if center:
    total = np.sum(W, axis=0)
    length = np.sum(W.astype(bool), axis=0)
    mean = total / length
    mean[np.where(length <= 1)] = 0
    W -= mean

  # Normalize the matrix
  if normalize:
    W /= np.sqrt(np.sum(np.abs(W**2), axis=1, keepdims=True))
  
  return W

#=======================================================================
# Randomly initializes a rectangular matrix
def rect_mat(*dims, dist='normal', spar=1., var=1., normalize=False, dtype=np.float32):

  if dist == 'normal':
    mat = np.random.normal(0,.5,*dims).astype(dtype)
  elif dist == 'uniform':
    mat = np.random.uniform(-1,1,dims).astype(dtype)
  else:
    mat = np.ones(*dims).astype(dtype)

  if spar < 1:
    mask = np.random.binomial(1, spar, dims).astype(dtype)
    mat *= mask * np.sqrt(float(var)/np.sum(mask, axis=1, keepdims=True)).astype(dtype)
  elif spar > 1:
    mask = np.array([np.random.permutation(spar) for i in xrange(len(mat))]) .astype(dtype)< spar
    mat *= mask * np.sqrt(float(var)/spar).astype(dtype)
  else:
    mat *= sqrt(var/np.prod(dims[:-1])).astype(dtype)

  if normalize:
    mat /= np.sqrt(np.sum(np.abs(mat**2), axis=1, keepdims=True))

  return mat

#=======================================================================
# Randomly initializes a tridiagonal matrix
def tridiag_mat(*dims, dist=None, scale=1., dtype=np.float32):

  if dist == 'normal':
    mat = np.random.randn(*dims).astype(dtype)
  elif dist == 'uniform':
    mat = np.random.uniform(-1,1,dims).astype(dtype)
  else:
    mat = np.ones(*dims).astype(dtype)

  mat *= tridiagonal(dims, dtype=dtype) * scale

  return mat

#=======================================================================
# Randomly initializes a diagonal matrix
def diag_mat(*dims, dist=None, scale=1., dtype=np.float32):

  if dist == 'normal':
    mat = np.random.randn(*dims).astype(dtype)
  elif dist == 'uniform':
    mat = np.random.uniform(-1,1,dims).astype(dtype)
  else:
    mat = np.ones(*dims).astype(dtype)

  mat *= diagonal(dims).astype(dtype) * scale

  return mat

#=======================================================================
# Randomly initializes an upper-triangular matrix
def tri_mat(*dims, dist='normal', spar=1., var=1., normalize=False, dtype=np.float32):

  if dist == 'normal':
    mat = np.random.randn(*dims)
  elif dist == 'uniform':
    mat = np.random.uniform(-1,1,dims)
  else:
    mat = np.ones(*dims)

  mat *= triangle(dims)
  if spar < 1:
    mask = np.random.binomial(1, spar, dims)
    mat *= mask
    mat *= np.sqrt(float(var)/np.sum(mat.astype(np.bool), axis=1, keepdims=True))
  else:
    mat *= sqrt(var/np.prod(dims[:-1]))

  if normalize:
    mat /= np.sqrt(np.sum(np.abs(mat**2), axis=1, keepdims=True))
  return mat


