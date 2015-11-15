#!/usr/bin/env python
import numpy as np
import theano
import theano.tensor as T
from scipy.optimize import fsolve

#***********************************************************************
# Error functions
#=======================================================================
# Squared difference
def squared_difference(output, target):
  """"""

  return T.sqr(output-target)/np.float32(2)

#=======================================================================
# Absolute difference
def absolute_difference(output, target):
  """"""

  return T.abs_(output-target)

#=======================================================================
# Sum squared error
def sum_squared_error(output, target):
  """"""

  return T.sum(squared_difference(output, target), axis=1)

#=======================================================================
# Sum absolute error
def sum_absolute_error(output, target):
  """"""

  return T.sum(absolute_difference(output, target), axis=1)

#=======================================================================
# Cosine similarity
def cosine_similarity(output, target):
  """"""

  return T.dot(output, target) / (T.sum(T.sqr(output), axis=0, keepdims=True) * T.sum(T.sqr(target), axis=1, keepdims=True))

error_funx = {
  'sum squared error': sum_squared_error,
  'sum absolute error': sum_absolute_error,
  'SSE': sum_squared_error,
  'SAE': sum_absolute_error
  }

#***********************************************************************
# Clipping functions
#=======================================================================
# Soft step function / Logistic sigmoid
def softstep(x):
  """"""
  
  return T.nnet.sigmoid(np.float32(2)*x)

#=======================================================================
# Sharp step function 
def sharpstep(x):
  """"""
  
  return T.clip((x+1)/2, 1, 0)

#=======================================================================
# Soft sign function / Hyperbolic tangent
def softsign(x):
  """"""
  
  return T.tanh(x)

#=======================================================================
# Sharp sign function 
def sharpsign(x):
  """"""
  
  return T.clip(x,1,-1)

#=======================================================================
# Soft absolute value
def softabs(x):
  """"""
  
  return T.log(np.float32(2)*T.cosh(x)) - T.log(np.float32(2))

#=======================================================================
# Sharp absolute value
def sharpabs(x):
  """"""
  
  return T.abs_(x)

#=======================================================================
# Soft rectifier function / Softplus function
def softpo(x):
  """"""
  
  return T.nnet.softplus(np.float32(2)*x)/np.float32(2)
  
#=======================================================================
# Sharp rectifier function / ReLU
def sharppo(x):
  """"""
  
  return T.maximum(x, np.float32(0))

#=======================================================================
# Soft truth function / bathtub function
def softbool(x):
  """"""
  
  return T.sqr(T.tanh(x))

#=======================================================================
# Sharp truth function
def sharpbool(x):
  """"""
  
  return T.clip(T.abs_(x), 1, 0)

#=======================================================================
# Soft positive function
def softpos(x):
  """"""
  
  return T.sqr(T.nnet.sigmoid(np.float32(2)*x))

#=======================================================================
# Sharp positive function
def sharppos(x):
  """"""
  
  return T.clip(x, 1, 0)

#=======================================================================
# Square root function
def sqrt(x):
  """"""
  
  return T.sign(x)*T.sqrt(T.abs_(x))

#=======================================================================
# Square function
def sqr(x):
  """"""
  
  return T.sign(x)*T.sqr(x)

#=======================================================================
# Soft max function
def softmax(x):
  """"""
  
  #return T.nnet.softmax(np.float32(2)*T.log(x.shape[0])*x) #Fastmax
  return T.nnet.softmax(2*x) #Slowmax

#=======================================================================
# Shortcuts
tanh = softsign
hardtanh = sharpsign
sig = softstep
sigmoid = softstep
abs = sharpabs
softplus = softpo
relu = sharppo
tub = softbool

clip_funx = {
        'softsign': softsign,
        'sharpsign': sharpsign,
        'softstep': softstep,
        'sharpstep': sharpstep,
        'softabs': softabs,
        'sharpabs': sharpabs,
        'softpo': softpo,
        'sharppo': sharppo,
        'softbool': softbool,
        'sharpbool': sharpbool,
        'softpos': softpos,
        'sharppos': sharppos,
        'sqrt': sqrt,
        'sqr': sqr,
        
        'tanh': softsign,
        'hardtanh': sharpsign,
        'sig': softstep,
        'sigmoid': softstep,
        'abs': sharpabs,
        'softplus': softpo,
        'relu': sharppo,
        'tub': softbool,
        
        'softmax': softmax,
  }
  
#***********************************************************************
# Splicing functions
#=======================================================================
# Sum
def sum(w):
  """"""
  
  return T.sum(w, axis=0)

#=======================================================================
# Product
def prod(w):
  """"""
  
  return T.prod(w, axis=0)

#=======================================================================
# Concatenate
def cat(w):
  """"""
  
  return T.concatenate(w, axis=0)

splice_funx = {
  'sum': sum,
  'prod': prod,
  'cat': cat,
  
  'product': prod,
  'concatenate': cat}

#***********************************************************************
# Glove reweighting function
#=======================================================================
# Fit the parameters to transition between the power and tanh functions smoothly
def splice(pair):
  xi, alpha = pair
  return (np.power(xi, alpha)-np.tanh(xi/alpha), alpha*np.power(xi, alpha-1) - 1/alpha*(1-np.tanh(xi/alpha)**2))
xi, alpha = fsolve(splice, (.533, .693))

#=======================================================================
# The reweighting function for GloVe
def reweight_glove(x, xmax):
  """"""
  
  return T.switch(x > xi*xmax,
                  T.tanh(x/(np.float32(alpha)*xmax)),
                  T.power(x/xmax, np.float32(alpha)))
 