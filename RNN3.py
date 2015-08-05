#!/usr/bin/env python

import numpy as np
import matwizard as mw
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import cPickle as pkl
import multiprocessing as mp
import sys
sys.setrecursionlimit(50000)

#***********************************************************************
# Helper functions

#=======================================================================
# Squared difference
def squared_difference(output, target):
  """"""

  return T.pow(output-target, 2)/2

#=======================================================================
# Absolute difference
def absolute_difference(output, target):
  """"""

  return T.abs_(output-target)

#=======================================================================
# Worker function
def func_worker(dataQueue, outQueue, func=None):
  """"""

  store = None
  for datum in iter(dataQueue.get, 'STOP'):
    if store is None:
      store = func(*datum) if func is not None else datum
    elif isinstance(store, (tuple, list)):
      for s, d in zip(store, func(*datum) if func is not None else datum):
        s += d
    else:
      store += func(*datum) if func is not None else datum
  if store is not None:
    outQueue.put(store)
  outQueue.put('STOP')
  #print 'Worker is done with %s; size: %d' % (str(func), dataQueue.qsize())
  return True

#***********************************************************************
# A library
class Library:
  """"""

  #=====================================================================
  # Initialize the model
  def __init__(self, keys, mat, **kwargs):
    """"""

    #-------------------------------------------------------------------
    # Keyword Arguments
    if 'start' in kwargs:
      self.start = kwargs['start']
    else:
      self.start = '<S>'

    if 'stop' in kwargs:
      self.stop = kwargs['stop']
    else:
      self.stop = '</S>'

    if 'unk' in kwargs:
      self.unk = kwargs['unk']
    else:
      self.unk = '<UNK>'

    if 'mutable' in kwargs:
      self.mutable = kwargs['mutable']
    else:
      self.mutable = True

    if isinstance(mat, int):
      mat = mw.rect_mat(len(keys), mat, normalize=True, dtype='float32')
      self.mutable = True
    else:
      assert self.start in keys
      assert self.stop in keys
      assert self.unk in keys
    self.wsize = mat.shape[1]

    if isinstance(keys, (tuple, list, set)):
      keys = set(keys)
      keys.add(self.start)
      keys.add(self.unk)
      keys.add(self.stop)
      keys = list(keys)
      self.idxs = {}
      self.strs = {}
      for i, key in enumerate(keys):
        self.idxs[key] = np.int32(i)
        self.strs[i] = key
    elif isinstance(keys, dict):
      if 0 in keys:
        self.strs = keys
        self.idxs = {v:k for k, v in keys.iteritems()}
      else:
        self.idxs = keys
        self.strs = {v:k for k, v in keys.iteritems()}

    self.L = theano.shared(mat)

    #=====================================================================
    # Convert idxs to vectors
    x = T.ivector(name='x')
    self.idx_to_vec = theano.function(
        inputs=[x],
        outputs=self.L[x],
        allow_input_downcast=True)

    #=====================================================================
    # Convert vectors to idxs 
    v = T.fmatrix(name='v')
    self.vec_to_idx = theano.function(
        inputs=[v],
        outputs=T.argmax(T.sum(T.sqr(self.L[None,:,:] - v[:,None,:]), axis=2), axis=1)
        allow_input_downcast=True)

  #=====================================================================
  # Get mutability 
  def mutable(self):
    """"""

    return self.mutable

  #=====================================================================
  # Get word size
  def wsize(self):
    """"""

    return self.wsize

  #=====================================================================
  # Get start string
  def start_str(self):
    """"""

    return self.start

  #=====================================================================
  # Get stop string 
  def stop_str(self):
    """"""

    return self.stop

  #=====================================================================
  # Get unk string
  def unk_str(self):
    """"""

    return self.unk

  #=====================================================================
  # Get start index
  def start_idx(self):
    """"""

    return self.str_to_idx(self.starts())

  #=====================================================================
  # Get stop index 
  def stop_idx(self):
    """"""

    return self.str_to_idx(self.stops())

  #=====================================================================
  # Get unk index
  def unk_idx(self):
    """"""

    return self.str_to_idx[self.stops()]

  #=====================================================================
  # Get start vector
  def start_vec(self):
    """"""

    return self.L[self.starti()]

  #=====================================================================
  # Get stop vector
  def stop_vec(self):
    """"""

    return self.L[self.stopi()]

  #=====================================================================
  # Get unk vector
  def unk_vec(self):
    """"""

    return self.L[self.unki()]

  #=====================================================================
  # Convert strs to idxs
  def str_to_idx(self, strs):
    """"""

    if not hasattr(strs, '__iter__'):
      strs = [strs]
    return np.array([[self.idxs[s] for s in strs]], dtype='int32')

  #=====================================================================
  # Convert idxs to strs 
  def idx_to_str(self, idxs):
    """"""

    if not hasattr(idxs, '__iter__'):
      idxs = [idxs]
    return [self.strs[i] for i in idxs]

  #=====================================================================
  # Convert strs to vectors 
  def str_to_vec(self, strs):
    """"""

    return self.idx_to_vec(np.array(self.str_to_idx(strs)))

  #=====================================================================
  # Convert vectors to strs
  def vec_to_str(self, vectors):
    """"""

    return self.idx_to_str(self.vec_to_idx(np.array(vectors)))

  #=====================================================================
  # Get tensor variable
  def get_subtensor(self, idxs):
    """"""

    return self.L[idxs]
  
#***********************************************************************
# An interface for optimization functions
class Opt:
  """"""
      
  #=====================================================================
  # Run SGD (with NAG)
  def SGD(self, eta_0=.01, T_eta=1, mu_max=.95, T_mu=1, dropout=1., anneal=0, accel=0):
    """"""

    #-------------------------------------------------------------------
    # Set up the updates & givens
    grad_norm  = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), self.gparams)))
    not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
    updates = []
    givens = []

    #-------------------------------------------------------------------
    # Set up a variable to keep track of the iteration
    tau = theano.shared(np.cast[self.dtype](0), name='tau')
    updates.extend([(tau, tau+1)])

    #-------------------------------------------------------------------
    # Set the annealing/acceleration schedule
    eta = eta_0*T.pow(T_eta/(tau+T_eta), anneal)
    mu  = mu_max*(1-T.pow(T_mu/(tau+T_mu), accel))

    #-------------------------------------------------------------------
    # Initialize stored vectors
    vparams  = [theano.shared(np.zeros_like(param.get_value()), name='v%s' % param.name) for param in self.params]

    #-------------------------------------------------------------------
    # Build up the updates
    updates.extend([(param, T.switch(not_finite, .1*param, param - eta*gparam)) for param, gparam in zip(self.params, self.gparams)])
    updates.extend([(vparam, T.switch(not_finite, .1*vparam, mu*vparam - eta*gparam)) for vparam, gparam in zip(vparams, self.gparams)])
    givens.extend([(param, param + mu*vparam) for param, vparam in zip(self.params, vparams)])

    #-------------------------------------------------------------------
    # Set up the dropout
    srng = RandomStreams()
    givens.extend([(hmask, srng.binomial(hmask.shape, 1, dropout, dtype=self.dtype)) for hmask in self.hmasks[:-1]] if dropout < 1 else [])

    #-------------------------------------------------------------------
    # Compile the sgd function
    sgd = theano.function(
        inputs=[],
        outputs=[],
        givens=givens,
        updates=updates,
        allow_input_downcast=True)

    #-------------------------------------------------------------------
    # Return the compiled function
    print 'SGD function compiled'
    return sgd

  #=====================================================================
  # Run AdaGrad (with NAG)
  def AdaGrad(self, eta_0=.01, T_eta=1, mu_max=.95, T_mu=1, epsilon=1e-6, dropout=1., anneal=0, accel=0):
    """"""

    #-------------------------------------------------------------------
    # Set up the updates & givens
    grad_norm  = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), self.gparams)))
    not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
    updates = []
    givens = []

    #-------------------------------------------------------------------
    # Set up a variable to keep track of the iteration
    tau = theano.shared(np.cast[self.dtype](0), name='tau')
    updates.extend([(tau, tau+1)])

    #-------------------------------------------------------------------
    # Set the annealing/acceleration schedule
    eta = eta_0*T.pow(T_eta/(tau+T_eta), anneal)
    mu  = mu_max*(1-T.pow(T_mu/(tau+T_mu), accel))

    #-------------------------------------------------------------------
    # Initialize stored vectors
    g2params_tm1   = [theano.shared(np.zeros_like(param.get_value()), name='g2%s' % param.name) for param in self.params]
    vparams = [theano.shared(np.zeros_like(param.get_value()), name='v%s' % param.name) for param in self.params]

    #-------------------------------------------------------------------
    # Build up the updates
    g2params_t    = [g2param_tm1 + (gparam_t**2) for g2param_tm1, gparam_t in zip(g2params_tm1, self.gparams)]
    deltaparams_t = [1/T.sqrt(g2param_t+epsilon)*gparam for g2param_t, gparam in zip(g2params_t, self.gparams)]

    updates.extend([(param, T.switch(not_finite, .1*param, param - eta*deltaparam_t)) for param, deltaparam_t in zip(self.params, deltaparams_t)])
    updates.extend([(g2param_tm1, T.switch(not_finite, .1*g2param_tm1, g2param_t)) for g2param_tm1, g2param_t in zip(g2params_tm1, g2params_t)])
    updates.extend([(vparam, T.switch(not_finite, .1*vparam, mu*vparam - eta*deltaparam_t)) for vparam, deltaparam_t in zip(vparams, deltaparams_t)])
    givens.extend([(param, param + mu*vparam) for param, vparam in zip(self.params, vparams)])

    #-------------------------------------------------------------------
    # Set up the dropout
    srng = RandomStreams()
    givens.extend([(hmask, srng.binomial(hmask.shape, 1, dropout, dtype=self.dtype)) for hmask in self.hmasks[:-1]] if dropout < 1 else [])

    #-------------------------------------------------------------------
    # Compile the sgd function
    adanag = theano.function(
        inputs=[],
        outputs=[],
        givens=givens,
        updates=updates,
        allow_input_downcast=True)

    #-------------------------------------------------------------------
    # Return the compiled function
    print 'AdaNAG function compiled'
    return adanag 

  #=====================================================================
  # Run RMSProp (with NAG)
  def RMSProp(self, eta_0=.01, T_eta=1, rho_0=.9, T_rho=1, mu_max=.95, T_mu=1, epsilon=1e-6, dropout=1., anneal=0, expand=0, accel=0):
    """"""

    #-------------------------------------------------------------------
    # Set up the updates & givens
    grad_norm  = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), self.gparams)))
    not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
    updates = []
    givens = []

    #-------------------------------------------------------------------
    # Set up a variable to keep track of the iteration
    tau = theano.shared(np.cast[self.dtype](0.), name='tau')
    updates.extend([(tau, tau+1)])

    #-------------------------------------------------------------------
    # Set the annealing/expansion schedule
    eta = eta_0*T.pow(T_eta/(tau+T_eta), anneal)
    rho = rho_0*(1-T.pow(T_rho/(tau+T_rho), expand))
    mu  = mu_max*(1-T.pow(T_mu/(tau+T_mu), accel))

    #-------------------------------------------------------------------
    # Initialize stored vectors
    g2params_tm1      = [theano.shared(np.zeros_like(param.get_value()), name='g2%s' % param.name) for param in self.params]
    vparams = [theano.shared(np.zeros_like(param.get_value()), name='v%s' % param.name) for param in self.params]

    #-------------------------------------------------------------------
    # Build up the updates
    g2params_t     = [rho*g2param_tm1 + (1-rho)*(gparam_t**2) for g2param_tm1, gparam_t in zip(g2params_tm1, self.gparams)]
    deltaparams_t  = [1/T.sqrt(g2param_t+epsilon)*gparam for g2param_t, gparam in zip(g2params_t, self.gparams)]

    updates.extend([(param, T.switch(not_finite, .1*param, param - eta*deltaparam_t)) for param, deltaparam_t in zip(self.params, deltaparams_t)])
    updates.extend([(g2param_tm1, T.switch(not_finite, .1*g2param_tm1, g2param_t)) for g2param_tm1, g2param_t in zip(g2params_tm1, g2params_t)])
    updates.extend([(vparam, T.switch(not_finite, .1*vparam, mu*vparam - eta*deltaparam_t)) for vparam, deltaparam_t in zip(vparams, deltaparams_t)])
    givens.extend([(param, param + mu*vparam) for param, vparam in zip(self.params, vparams)])

    #-------------------------------------------------------------------
    # Set up the dropout
    srng = RandomStreams()
    givens.extend([(hmask, srng.binomial(hmask.shape, 1, dropout, dtype=self.dtype)) for hmask in self.hmasks[:-1]] if dropout < 1 else [])

    #-------------------------------------------------------------------
    # Compile the rmsprop function
    rmsprop = theano.function(
        inputs=[],
        outputs=[],
        givens=givens,
        updates=updates,
        allow_input_downcast=True)

    print 'RMSProp function compiled'
    return rmsprop 
  
  #=====================================================================
  # Run AdaDelta
  def AdaDelta(self, eta_0=1., T_eta=1, rho_0=.9, T_rho=1, epsilon=1e-6, dropout=1., anneal=0, expand=0):
    """"""

    #-------------------------------------------------------------------
    # Set up the updates & givens
    grad_norm  = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), self.gparams)))
    not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
    updates = []
    givens = []

    #-------------------------------------------------------------------
    # Set up a variable to keep track of the iteration
    tau = theano.shared(np.cast[self.dtype](0.), name='tau')
    updates.extend([(tau, tau+1)])

    #-------------------------------------------------------------------
    # Set the annealing/expansion schedule
    eta = eta_0*T.pow(T_eta/(tau+T_eta), anneal)
    rho = rho_0*(1-T.pow(T_rho/(tau+T_rho), expand))

    #-------------------------------------------------------------------
    # Initialize stored vectors
    g2params_tm1      = [theano.shared(np.zeros_like(param.get_value()), name='g2%s' % param.name) for param in self.params]
    delta2params_tm1  = [theano.shared(np.zeros_like(param.get_value()), name='delta%s' % param.name) for param in self.params]

    #-------------------------------------------------------------------
    # Build up the updates
    g2params_t     = [rho*g2param_tm1 + (1-rho)*(gparam_t**2) for g2param_tm1, gparam_t in zip(g2params_tm1, self.gparams)]
    deltaparams_t  = [T.sqrt(delta2param_tm1+epsilon)/T.sqrt(g2param_t+epsilon)*gparam for delta2param_tm1, g2param_t, gparam in zip(delta2params_tm1, g2params_t, self.gparams)]
    delta2params_t = [rho*delta2param_tm1 + (1-rho)*(deltaparam_t**2) for delta2param_tm1, deltaparam_t in zip(delta2params_tm1, deltaparams_t)]

    updates.extend([(param, T.switch(not_finite, .1*param, param - eta*deltaparam_t)) for param, deltaparam_t in zip(self.params, deltaparams_t)])
    updates.extend([(delta2param_tm1, T.switch(not_finite, .1*delta2param_tm1, delta2param_t)) for delta2param_tm1, delta2param_t in zip(delta2params_tm1, delta2params_t)])
    updates.extend([(g2param_tm1, T.switch(not_finite, .1*g2param_tm1, g2param_t)) for g2param_tm1, g2param_t in zip(g2params_tm1, g2params_t)])

    #-------------------------------------------------------------------
    # Set up the dropout
    srng = RandomStreams()
    givens.extend([(hmask, srng.binomial(hmask.shape, 1, dropout, dtype=self.dtype)) for hmask in self.hmasks[:-1]] if dropout < 1 else [])

    #-------------------------------------------------------------------
    # Compile the adadelta function
    adadelta = theano.function(
        inputs=[],
        outputs=[],
        givens=givens,
        updates=updates,
        allow_input_downcast=True)

    print 'AdaDelta function compiled'
    return adadelta

  #=====================================================================
  # Run Adam
  def Adam(self, eta_0=.05, T_eta=1, rho1_0=.9, rho2_0=.99, T_rho=1, epsilon=1e-6, dropout=1., anneal=0, expand=0):
    """"""

    #-------------------------------------------------------------------
    # Set up the updates & givens
    grad_norm  = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), self.gparams)))
    not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
    updates = []
    givens = []

    #-------------------------------------------------------------------
    # Set up a variable to keep track of the iteration
    tau = theano.shared(np.cast[self.dtype](0.), name='tau')
    updates.extend([(tau, tau+1)])

    #-------------------------------------------------------------------
    # Set the annealing schedule
    eta = eta_0*T.pow(T_eta/(tau+T_eta), anneal)
    rho1 = rho1_0*(1-T.pow(T_rho/(tau+T_rho), expand))
    rho2 = rho2_0*(1-T.pow(T_rho/(tau+T_rho), expand))

    #-------------------------------------------------------------------
    # Initialize stored vectors
    mparams = [theano.shared(np.zeros_like(param.get_value()), name='m%s' % param.name) for param in self.params]
    vparams = [theano.shared(np.zeros_like(param.get_value()), name='v%s' % param.name) for param in self.params]

    #-------------------------------------------------------------------
    # Set up the updates 
    mparams_t = [(rho1*mparam + (1-rho1)*gparam) / (1-rho1) for mparam, gparam in zip(mparams, self.gparams)]
    vparams_t = [(rho2*vparam + (1-rho2)*gparam**2) / (1-rho2) for vparam, gparam in zip(vparams, self.gparams)]
    deltaparams_t = [mparam_t/(T.sqrt(vparam_t)+epsilon) for mparam_t, vparam_t in zip(mparams_t, vparams_t)]

    updates.extend([(param, T.switch(not_finite, .1*param, param - eta * deltaparam_t)) for param, deltaparam_t in zip(self.params, deltaparams_t)])
    updates.extend([(mparam, T.switch(not_finite, .1*mparam, mparam_t)) for mparam, mparam_t in zip(mparams, mparams_t)])
    updates.extend([(vparam, T.switch(not_finite, .1*vparam, vparam_t)) for vparam, vparam_t in zip(vparams, vparams_t)])

    #-------------------------------------------------------------------
    # Set up the dropout
    srng = RandomStreams()
    givens.extend([(hmask, srng.binomial(hmask.shape, 1, dropout, dtype=self.dtype)) for hmask in self.hmasks[:-1]] if dropout < 1 else [])

    #-------------------------------------------------------------------
    # Compile the adam function
    adam = theano.function(
        inputs=[],
        outputs=[],
        givens=givens,
        updates=updates,
        allow_input_downcast=True)
    
    print 'Adam function compiled'
    return adam

  #=====================================================================
  # Run SMORMS3 (Simon Funk)
  def SMORMS3(self, eta_0=.05, T_eta=1, rho1_0=.9, rho2_0=.99, T_rho=1, epsilon=1e-6, dropout=1., anneal=0, expand=0):
    """"""

    #-------------------------------------------------------------------
    # Set up the updates & givens
    grad_norm  = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), self.gparams)))
    not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
    updates = []
    givens = []

    #-------------------------------------------------------------------
    # Set up a variable to keep track of the iteration
    tau = theano.shared(np.cast[self.dtype](0.), name='tau')
    updates.extend([(tau, tau+1)])

    #-------------------------------------------------------------------
    # Set the annealing schedule
    eta = eta_0*T.pow(T_eta/(tau+T_eta), anneal)
    rho1 = rho1_0*(1-T.pow(T_rho/(tau+T_rho), expand))
    rho2 = rho2_0*(1-T.pow(T_rho/(tau+T_rho), expand))

    #-------------------------------------------------------------------
    # Initialize stored vectors
    mparams_tm1 = [theano.shared(np.ones_like(param.get_value()), name='m%s' % param.name) for param in self.params]
    vparams_tm1 = [theano.shared(np.zeros_like(param.get_value()), name='v%s' % param.name) for param in self.params]
    g2params_tm1 = [theano.shared(np.zeros_like(param.get_value()), name='g2%s' % param.name) for param in self.params]
    
    #-------------------------------------------------------------------
    # Set up the updates 
    rparams_t = [1/(mparam_tm1+1) for mparam_tm1 in mparams_tm1]
    mparams_t = [(1-rparam_t)*mparam_tm1 + rparam_t*gparam for rparam_t, mparam_tm1, gparam in zip(rparams_t, mparams_tm1, self.gparams)]
    vparams_t = [(1-rparam_t)*vparam_tm1 + rparam_t*gparam**2 for rparam_t, vparam_tm1, gparam in zip(rparams_t, vparams_tm1, self.gparams)]
    deltaparams_t = [vparam_t**2/(g2param_t+epsilon) for vparam_t, g2param_t in zip(vparams_t, g2params_t)]

    updates.extend([(param, T.switch(not_finite, .1*param, param - T.min(eta, deltaparam_t)/(T.sqrt(g2param_t)+epsilon))) for param, deltaparam_t, g2param_t in zip(self.params, deltaparams_t, g2params_t)])
    updates.extend([(mparam_tm1, T.switch(not_finite, .1*mparam_tm1, 1+mparam_tm1*(1-deltaparam_t))) for mparam_tm1, deltaparam_t in zip(mparams_tm1, deltaparams_t)])
    updates.extend([(g2param_tm1, T.switch(not_finite, .1*g2param_tm1, g2param_t)) for g2param_tm1, g2param_t in zip(g2params_tm1, g2params_t)])
    updates.extend([(vparam_tm1, T.switch(not_finite, .1*vparam_tm1, vparam_t)) for vparam_tm1, vparam_t in zip(vparams_tm1, vparams_t)])

    #-------------------------------------------------------------------
    # Set up the dropout
    srng = RandomStreams()
    givens.extend([(hmask, srng.binomial(hmask.shape, 1, dropout, dtype=self.dtype)) for hmask in self.hmasks[:-1]] if dropout < 1 else [])

    #-------------------------------------------------------------------
    # Compile the adam function
    adam = theano.function(
        inputs=[],
        outputs=[],
        givens=givens,
        updates=updates,
        allow_input_downcast=True)
    
    print 'Adam function compiled'
    return adam

#***********************************************************************
# A multilayer basic recurrent neural encoder
class Encoder(Opt):
  """"""

  def __init__(self, libs, dims, **kwargs):
    """"""

    #-------------------------------------------------------------------
    # Keywork arguments
    if 'model' in kwargs:
      self.model = kwargs['model']
    else:
      self.model = 'RNN'
    
    if 'window' in kwargs:
      self.window = kwargs['window']
    else:
      self.window = 1

    if 'reverse' in kwargs:
      self.reverse = kwargs['reverse']
    else:
      self.reverse = 'RNN'

    if 'L1reg' in kwargs:
      self.L1reg = kwargs['L1reg']
    else:
      self.L1reg = 0.

    if 'L2reg' in kwargs:
      self.L2reg = kwargs['L1reg']
    else:
      self.L2reg = 0.


    #-------------------------------------------------------------------
    # Process the libraries
    # TODO still need Lparams?
    self.libs = []
    ldims = []
    l_0 = []
    l_max = []
    for lib in libs:
      if not isinstance(lib, Library):
        lib = Library(*lib)
      self.libs.append(lib)
      ldims.append(lib.wsize())
      if reverse:
        l_0.append(lib.stop_vec())
        l_max.append(lib.start_vec())
      else:
        l_0.append(lib.start_vec())
        l_max.append(lib.stop_vec())
    ldims = np.sum(ldims)*self.window
    l_0 = T.concatenate(l_0, axis=1)
    l_max = T.concatenate(l_max, axis=1)
    dims.insert(0, ldims)

    #-------------------------------------------------------------------
    # Bundle the model params
    self.model = model
    if self.model in ('RNN',):
      gates = 1
    elif self.model in ('GRU', 'FastGRU', 'FastLSTM'): # FastLSTM couples the input/forget gate
      gates = 3
    elif self.model in ('LSTM',):
      gates = 4

    self.Wparams = []
    self.bparams = []
    self.hparams = []
    self.h_0     = []
    self.hmasks  = []
    if self.model.endswith('LSTM'):
      self.c_0   = []

    for i in xrange(1, len(dims)):
      W = matwizard(dims[i-1], dims[i]*gates)
      U = matwizard(dims[i-1], dims[i], shape='diag')
      if gates > 1:
        U = np.concatenate([U, matwizard(dims[i-1], dims[i]*(gates-1))], axis=1)
      self.Wparams.append(theano.shared(np.concatenate([W, U]), name='W-%d' % (i+1)))
      self.bparams.append(theano.shared(np.zeros(dims[i]*gates), name='b-%d' % (i+1)))
      self.hparams.append(theano.shared(np.zeros(dims[i]), name='h-%d' % (i+1)))
      self.hmasks.append(theano.shared(np.ones(dims[i]*gates), name='hmask-%d' % (i+1)))
      self.h_0.append(theano.shared(np.zeros(dims[i]), name='h_0-%d' % (i+1)))
      self.outputs_info = h_0
      if self.model.endswith('LSTM'):
        self.c_0.append(theano.shared(np.zeros(dims[i]), name='h_0-%d' % (i+1)))
        self.outputs_info = h_0 + c_0
      self.params =\
          self.Wparams +\
          self.bparams +\
          self.hparams +\
          self.h_0 +\
          (self.C_0 if self.model.endswith('LSTM') else [])
      self.gparams = [theano.shared(np.zeros_like(param.get_value(), name='g'+param.name, dtype='float32')) for param in self.params]
      paramVars =\
          [T.matrix() for Wparam in self.Wparams] +\
          [T.vector() for bparam in self.bparams] +\
          [T.vector() for hparam in self.hparams] +\
          [T.vector() for h_0_l in self.h_0] +\
          ([T.vector() for c_0_l in self.c_0] if self.model.endswith('LSTM') else [])

      #-----------------------------------------------------------------
      # Build the input/output variables
      self.idxs = T.imatrix('idxs')
      self.iparams = []
      for i, lib in enumerate(self.libs):
        self.iparams.append(lib.get_subtensor(idxs[:,i]))



