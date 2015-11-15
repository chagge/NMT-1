#!/usr/bin/env python
# -*- coding utf-8 -*-

# Matrix operations
import numpy as np
np.random.seed(9412)
import theano
#theano.config.optimizer='None'
#theano.config.exception_verbosity='High'
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams 
srng = RandomStreams(9412)
from funx import ?
# Data structures
import codecs
# Pickling
import cPickle as pkl
import sys
sys.setrecursionlimit(50000)
# Utilities
import time
import warnings
warnings.simplefilter('error')

CLIPSIZE = 32

#***********************************************************************
# An optimizable sequence to sequence class
class Opt():
  pass

#***********************************************************************
# Sequence to sequence neural network
class S2SNN(Opt):
  """"""
  
  #=====================================================================
  # Initialize the model
  def __init__(self, hsizes, gloves, **kwargs):
    """"""
    
    #-------------------------------------------------------------------
    # Keyword arguments
    if 'model' in kwargs:
      self.model = kwargs['model']
    else:
      self.model = 'RNN'
    
    if 'feedback' in kwargs:
      self.feedback = kwargs['feedback']
    else:
      self.feedback = True
    
    if 'feed_enc_from_top' in kwargs:
      self.feed_enc_from_top = kwargs['feed_enc_from_top']
    else:
      self.feed_enc_from_top = True
    
    if 'feed_dec_from_start' in kwargs:
      self.feed_dec_from_start = kwargs['feed_dec_from_start']
    else:
      self.feed_dec_from_start = True
    
    if 'hfunc' in kwargs:
      self.hfunc = kwargs['hfunc']
    else:
      self.hfunc = 'tanh' # change this to sharpsign?
    
    if 'window' in kwargs:
      self.window = kwargs['window']
    else:
      self.window = 1
    
    if 'reverse' in kwargs:
      self.reverse = kwargs['reverse']
    else:
      self.reverse = True
    
    if 'L1reg' in kwargs:
      self.L1reg = np.float32(kwargs['L1reg'])
    else:
      self.L1reg = np.float32(0)
    
    if 'L2reg' in kwargs:
      self.L2reg = np.float32(kwargs['L2reg'])
    else:
      self.L2reg = np.float32(0)
    
    #-------------------------------------------------------------------
    # Build model params
    if self.model in ('RNN',):
      gates = 0
    elif self.model in ('FastGRU',): #FastGRU only has a coupled input/forget gate
      gates = 1
    elif self.model in ('GRU', 'FastLSTM'): #FastLSTM has a coupled input/forget gate
      gates = 2
    elif self.model in ('LSTM'):
      gates =3
    
    self.theta_enc_W  = []
    self.theta_enc_Wb = []
    self.hmasks_enc   = []
    
    self.theta_dec_W  = []
    self.theta_dec_Wb = []
    self.hmasks_dec   = []
    
    self.theta_h0     = []
    self.c0           = []
    
    self._hsizes = [sum(gloves[0].wsizes())] + hsizes
    for i in xrange(1, len(self._hsizes)-1):
      W = matwizard(self.hsizes(i), self.hsizes(i-1), shape='diag', output=self.hfunc, imput=('' if i == 1 else self.hfunc), recur=2+self.feed_enc_from_top*(i==1))
      U = matwizard(self.hsizes(i), self.hsizes(i), shape='diag', output=self.hfunc, imput=self.hfunc, recur=2+self.feed_enc_from_top*(i==1))
      if i == 1 and self.feed_enc_from_top:
        V = matwizard(self.hsizes(i), self.hsizes(-1), shape='diag', output=self.hfunc, imput=self.hfunc, recur=3)
      if gates > 0:
        W = np.concatenate([W, matwizard(self.hsizes(i)*gates, self.hsizes(i-1), output='sigmoid', imput=('' if i == 1 else self.hfunc), recur=2+self.feed_enc_from_top*(i==1))], axis=0)
        U = np.concatenate([U, matwizard(self.hsizes(i)*gates, self.hsizes(i), output='sigmoid', imput=self.hfunc, recur=2+self.feed_enc_from_top*(i==1))], axis=0)
        if i == 1 and self.feed_enc_from_top:
          V = np.concatenate([V, matwizard(self.hsizes(i)*gates, self.hsizes(-1), output='sigmoid', imput=self.hfunc, recur=3)], axis=0)
      if i == 1 and self.feed_enc_from_top:
        self.theta_enc_W.append(theano.shared(np.concatenate([W, U, V], axis=1), name='W-%d' % i))
        self.theta_dec_W.append(theano.shared(np.concatenate([W, U, V], axis=1), name='W-%d' % i))
      else:
        self.theta_enc_W.append(theano.shared(np.concatenate([W, U], axis=1), name='W-%d' % i))
        self.theta_dec_W.append(theano.shared(np.concatenate([W, U, V], axis=1), name='W-%d' % i))
      self.theta_enc_Wb.append(theano.shared(np.zeros(self.hsizes(i)*(1+gates), dtype='float32'), name='Wb-%d' % i))
      self.theta_dec_Wb.append(theano.shared(np.zeros(self.hsizes(i)*(1+gates), dtype='float32'), name='Wb-%d' % i))
      self.theta_h0.append(theano.shared(np.zeros(self.hsizes(i), dtype='float32'), name='h_0-%d' % i))
      self.c0.append(theano.shared(np.zeros(self.hsizes(i), dtype='float32'), name='c_0-%d' % i))
      self.hmasks_enc.append(theano.shared(np.ones(self.hsizes(i), dtype='float32'), name='hmask-%d' % i))
      self.hmasks_dec.append(theano.shared(np.ones(self.hsizes(i), dtype='float32'), name='hmask-%d' % i))
  
  #=====================================================================
  # Return the hidden sizes of the model
  def hsizes(self, layer=None):
    """"""
    
    if layer is not None:
      return self._hsizes
    else:
      return self._hsizes[layer]