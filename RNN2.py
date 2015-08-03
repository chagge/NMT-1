#!/usr/bin/env python
import numpy as np
np.float_ = np.float32
np.int_ = np.int32
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
      self.indices = {}
      self.strings = {}
      for i, key in enumerate(keys):
        self.indices[key] = np.int32(i)
        self.strings[i] = key
    elif isinstance(keys, dict):
      if 0 in keys:
        self.strings = keys
        self.indices = {v:k for k, v in keys.iteritems()}
      else:
        self.indices = keys
        self.strings = {v:k for k, v in keys.iteritems()}

    self.L = theano.shared(mat)

    #=====================================================================
    # Convert indices to vectors
    x = T.ivector(name='x')
    self.i2v = theano.function(
        inputs=[x],
        outputs=self.L[x],
        allow_input_downcast=True)

    #=====================================================================
    # Convert vectors to indices 
    v = T.fmatrix(name='v')
    self.v2i = theano.function(
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
  def starts(self):
    """"""

    return self.start

  #=====================================================================
  # Get stop string 
  def stops(self):
    """"""

    return self.stop

  #=====================================================================
  # Get unk string
  def unks(self):
    """"""

    return self.unk

  #=====================================================================
  # Get start index
  def starti(self):
    """"""

    return self.s2i(self.starts())

  #=====================================================================
  # Get stop index 
  def stopi(self):
    """"""

    return self.s2i(self.stops())

  #=====================================================================
  # Get unk index
  def unki(self):
    """"""

    return self.s2i[self.stops()]

  #=====================================================================
  # Get start vector
  def startv(self):
    """"""

    return self.L[self.starti()]

  #=====================================================================
  # Get stop vector
  def stopv(self):
    """"""

    return self.L[self.stopi()]

  #=====================================================================
  # Get unk vector
  def unkv(self):
    """"""

    return self.L[self.unki()]

  #=====================================================================
  # Convert strings to indices
  def s2i(self, strings):
    """"""

    if not hasattr(strings, '__iter__'):
      strings = [strings]
    return np.array([[self.indices[s] for s in strings]], dtype='int32')

  #=====================================================================
  # Convert indices to strings 
  def i2s(self, indices):
    """"""

    if not hasattr(indices, '__iter__'):
      indices = [indices]
    return [self.strings[i] for i in indices]

  #=====================================================================
  # Convert strings to vectors 
  def s2v(self, strings):
    """"""

    return self.i2v(np.array(self.s2i(strings)))

  #=====================================================================
  # Convert vectors to strings
  def v2s(self, vectors):
    """"""

    return self.i2s(self.v2i(np.array(vectors)))

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
  pass

#***********************************************************************
# A (multilayer) basic recurrent network encoder
class LangModel(Opt):
  """"""

  #=====================================================================
  # Initialize the model
  def __init__(self, libs, dims, **kwargs):
    """"""

    #-------------------------------------------------------------------
    # Keyword arguments
    # TODO these later if you feel like it
    #if 'shareLib' in kwargs:
    #  self.shareLib = kwargs['shareLib']
    #else:
    #  self.shareLib = False

    #if 'catContext' in kwargs:
    #  self.catContext = kwargs['catContext']
    #else:
    #  self.catContext = True

    if 'model' in kwargs:
      self.model = kwargs['model']
    else:
      self.model = 'RNN'

    if 'window' in kwargs:
      self.window = kwargs['window']
    else:
      self.window = 1

    if 'L1reg' in kwargs:
      self.L1reg = kwargs['L1reg']
    else:
      self.L1reg = 0.

    if 'L2reg' in kwargs:
      self.L2reg = kwargs['L2reg']
    else:
      self.L2reg = 0.

    #-------------------------------------------------------------------
    # Process the libraries
    self.libs = []
    self.Lparams = []
    ldims = []
    l_0   = []
    l_max = []
    for lib in libs:
      if not isinstance(lib, Library):
        lib = Library(*lib)
      self.libs.append(lib)
      ldim = (dims[-1]*2, len(lib.strings()))
      Lmat = mw.rect_mat(*ldim, dtype='float32')
      self.Lparams.append(theano.shared(Lmat, name='LW-%d' % len(self.libs)))
      ldims.append(ldim)
      l_0.append(lib.startv())
      l_max.append(lib.stopv())
    l_0 = T.concatenate(g_0, axis=1)
    l_max = T.concatenate(g_max, axis=1)
    yhat_0 = [T.zeros(lib.wsize()) for lib in self.libs]
    ldims = np.array(ldims)
    dims.insert(0, np.sum(ldims[:,1])*self.window)

    #-------------------------------------------------------------------
    # Bundle the model params
    self.model = model
    if self.model in ('RNN',)
      gates = 1
    elif self.model in ('GRU', 'FastGRU', 'FastLSTM'):
      gates = 3
    elif self.model in ('LSTM',):
      gates = 4

    Wlparams = []
    Wrparams = []
    blparams = []
    brparams = []
    hlparams = []
    hrparams = []
    hlmasks  = []
    hrmasks  = []
    hl_0     = []
    hr_0     = []
    if model.endswith('LSTM'):
      Cl_0 = []
      Cr_0 = []

    for i in xrange(1, len(dims)):
      W = mw.rect_mat(dims[i-1], dims[i]*gates)
      if i > 1:
        W = np.concatenate([W,W])
      U = mw.diag_mat(dims[i],dims[i])
      if gates > 1:
        U = np.concatenate([U] + [mw.rect_mat(dims[i], dims[i])]*(gates-1), axis=1)
      bparam = np.zeros(dims[i]*gates)
      Wparam = np.concatenate([W, U, b])
      hparam = np.zeros(dims[1])
      hmask  = np.ones(dims[1]*gates)
      h_0    = np.zeros(dims[1])
      if model.endswith('LSTM'):
        C_0 = h_0

      Wlparams.append(theano.shared(Wparam, name='W.l-%d' % (i+1)))
      Wrparams.append(theano.shared(Wparam, name='W.r-%d' % (i+1)))
      blparams.append(theano.shared(bparam, name='b.l-%d' % (i+1)))
      brparams.append(theano.shared(bparam, name='b.r-%d' % (i+1)))
      hlparams.append(theano.shared(hparam, name='hparam.l-%d' % (i+1)))
      hrparams.append(theano.shared(hparam, name='hparam.r-%d' % (i+1)))
      hlmasks.append(theano.shared(hmask, name='hmask.l-%d' % (i+1)))
      hrmasks.append(theano.shared(hmask, name='hmask.r-%d' % (i+1)))
      hl_0.append(theano.shared(h_0, name='h_0.l-%d' % (i+1)))
      hr_0.append(theano.shared(h_0, name='h_0.r-%d' % (i+1)))
      if model.endswith('LSTM'):
        Cl_0.append(theano.shared(C_0, name='C_0.l-%d' % (i+1)))
        Cr_0.append(theano.shared(C_0, name='C_0.r-%d' % (i+1)))

      self.Wparams = Wlparams + Wrparams
      self.bparams = blparams + brparams
      self.hparams = hlparams + hrparams
      self.hmasks  = hlmasks + hrmasks
      self.h_0     = hl_0 + hr_0
      if model.endswith('LSTM'):
        self.C_0   = Cl_0 + Cr_0
      
      if model.endswith('LSTM'):
        self.outputs_info = hl_0 + yhat_0 + Cl_0 + yhat_0 + hr_0 + yhat_0 + Cr_0 + yhat_0
      else:
        self.outputs_info = hl_0 + yhat_0 + hr_0 + yhat_0

    self.params =\
        self.Wparams +\
        self.bparams +\
        self.Lparams +\
        self.hparams +\
        self.h_0 +\
        (self.C_0 if self.model.endswith('LSTM') else [])
    self.gparams = [theano.shared(np.zeros_like(param.get_value(), name='g'+param.name, dtype='float32')) for param in self.params]
    paramVars =\
        [T.matrix() for Wparam in self.Wparams] +\
        [T.vector() for bparam in self.bparams] +\
        [T.matrix() for Lparam in self.Lparams] +\
        [T.vector() for hparam in self.hparams] +\
        [T.vector() for h_0_l  in self.h_0] +\
        ([T.vector() for C_0_l  in self.C_0] if self.model.endswith('LSTM') else [])

    #-------------------------------------------------------------------
    # Build the input/output variables
    # Some of this can be done in the Library class
    idxsl = T.imatrix('idxs')
    idxsr = idxsl[::-1]
    iparamsl = []
    iparamsr = []
    for i, lib in enumerate(self.libs):
      iparamsl.append(lib.get_subtensor(idxsl[:-1,i]))
      iparamsr.append(lib.get_subtensor(idxsr[:-1,i]))
    xl = T.concatenate(iparamsl, axis=1)
    xr = T.concatenate(iparamsr, axis=1)


    self.lparams = []
    for i, lib in enumerate(self.libs)
      self.lparams.append(lib.get_subtensor(self.idxs[:,i]))
    self.xl      = T.concatenate(self.lparams[:-1], axis=1)
    self.xr      = T.concatenate(self.lparams[1::-1], axis=1)
    self.y       = self.idxs[1:]

    #-------------------------------------------------------------------
    # Build the activation variable

    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if self.model == 'RNN':
      ##  The recurrence function for training/evaluating
      def recur(i, *h_tm1):

        # For all but the last layer
        h_t = [self.x[i:i+self.window].flatten()]
        for h_tm1_l, Wparam, Uparam, bparam, hparam, hmask in zip(h_tm1, self.Wparams, self.Uparams, self.bparams, self.hparams, self.hmasks):
          s = T.nnet.sigmoid(2*hparam)
          a = T.dot(h_t[-1], Wparam) +\
              T.dot(h_tm1_l, Uparam) +\
              bparam

          c = s*T.tanh(a) +\
              (1-s)*T.switch(a > 0, a, 0)
          h_t.append(c*hmask)

        # For the last layer
        h_tm1_l = h_t[-1]
        y_t = []
        for Lparam in self.Lparams:
          a = T.dot(h_tm1_l, Lparam) +\
              self.bparams[-1]

          c = T.nnet.softmax(2*a)
          y_t.append(c)

        h_t.append(y_t)

        return h_t[1:]

      ##  The feedback function for generating
      #TODO make this output a probability vector rather than an index vector
      def feedback(idxs_t, *h_tm1):

        # For all but the last layer
        x_t = T.concatenate([lib.get_subtensor(idxs_t[:,i]) for i, lib in enumerate(self.libs)], axis=1)
        h_t = [x_t.flatten()]
        for h_tm1_l, Wparam, Uparam, bparam, hparam, hmask in zip(h_tm1, self.Wparams, self.Uparams, self.bparams, self.hparams, self.hmasks):
          s = T.nnet.sigmoid(2*hparam)
          a = T.dot(h_t[-1], Wparam) +\
              T.dot(h_tm1_l, Uparam) +\
              bparam

          c = s*T.tanh(a) +\
              (1-s)*T.switch(a > 0, a, 0)
          h_t.append(c*hmask)

        # For the last layer
        h_t_lm1 = h_t[-1]
        l_tp1 = []
        for Lparam in self.Lparams:
          a = T.dot(h_t_lm1, Lparam) +\
              self.bparams[-1]

          c = T.nnet.softmax(2*a)
          idxs_tp1.append(T.argmax(c, keepdims=True))

        x_tp1 = T.concatenate([x_t[1:], T.concatenate(l_tp1)[None,:]])

        return [x_tp1]+h_t[1:], theano.scan_module.until(T.eq(h_t[-1], g_max))

    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif self.model == 'GRU':
      ##  The recurrence function for training/evaluating
      def recur(i, *h_tm1):

        # For all but the last layer
        h_t = [self.x[i:i+self.window].flatten()]
        for h_tm1_l, Wparam, Uparam, bparam, hparam, hmask in zip(h_tm1, Wparam, Uparam, bparam, hparam, hmask):

          sliceLen = Wparam.shape[1]/3
          azr_W = T.dot(h_t[-1], Wparam)
          zr_U  = T.dot(h_tm1_l, Uparam[:,sliceLen:])
          s = T.nnet.sigmoid(2*hparam)
          z = T.nnet.sigmoid(2*(
              azr_W[sliceLen:2*sliceLen] +\
              zr_U[:sliceLen] +\
              bparam[sliceLen:2*sliceLen]))
          r = T.nnet.sigmoid(2*(
              azr_W[2*sliceLen:] +\
              zr_U[sliceLen:] +\
              bparam[2*sliceLen:]))
          a = azr_W[:sliceLen] +\
              T.dot(r*h_tm1_l, Uparam[:,:sliceLen]) +\
              bparam[:sliceLen]

          c = s*T.tanh(a) +\
              (1-s)*T.switch(a > 0, a, 0)
          C = ((1-z)*h_tm1_l + z*c)
          h_t.append(C*hmask)

        # For the last layer
        h_tm1_l = h_t[-1]
        for Lparam in self.Lparams:
          a = T.dot(h_tm1_l, Lparam) +\
              self.bparams[-1]

          c = T.nnet.softmax(2*a)
          y_t.append(c)

        h_t.append(y_t)

        return h_t[1:]

      ##  The feedback function for generating
      def feedback(idxs, *h_tm1):

        # For all but the last layer
        x_t = T.concatenate([lib.get_subtensor(idxs[:,i]) for i, lib in enumerate(self.libs)], axis=1)
        h_t = [x_t.flatten()]
        for h_tm1_l, Wparam, Uparam, bparam, hparam, hmask in zip(h_tm1, Wparam, Uparam, bparam, hparam, hmask):

          sliceLen = Wparam.shape[1]/3
          azr_W = T.dot(h_t[-1], Wparam)
          zr_U  = T.dot(h_tm1_l, Uparam[:,sliceLen:])
          s = T.nnet.sigmoid(2*hparam)
          z = T.nnet.sigmoid(2*(
              azr_W[sliceLen:2*sliceLen] +\
              zr_U[:sliceLen] +\
              bparam[sliceLen:2*sliceLen]))
          r = T.nnet.sigmoid(2*(
              azr_W[2*sliceLen:] +\
              zr_U[sliceLen:] +\
              bparam[2*sliceLen:]))
          a = azr_W[:sliceLen] +\
              T.dot(r*h_tm1_l, Uparam[:,:sliceLen]) +\
              bparam[:sliceLen]

          c = s*T.tanh(a) +\
              (1-s)*T.switch(a > 0, a, 0)
          C = ((1-z)*h_tm1_l + z*c)
          h_t.append(C*hmask)

        # For the last layer
        #TODO

    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif self.model == 'FastGRU':
      ##  The recurrence function for training/evaluating
      def recur(i, *h_tm1):

        # For all but the last layer
        h_t = [self.x[i:i+self.window].flatten()]
        for h_tm1_l, Wparam, Uparam, bparam, hparam, hmask in zip(h_tm1, Wparam, Uparam, bparam, hparam, hmask):

          sliceLen = Wparam.shape[1]/3
          azr = T.dot(h_t[-1], Wparam) + \
              T.dot(h_tm1_l, Uparam) +\
              bparam
          s = T.nnet.sigmoid(2*hparam)
          z = T.nnet.sigmoid(2*(azr[:sliceLen]))
          r = T.nnet.sigmoid(2*(azr[sliceLen:2*sliceLen]))
          a = azr[:sliceLen]

          c = s*T.tanh(a) +\
              (1-s)*T.switch(a > 0, a, 0)
          C = ((1-z)*r*h_tm1_l + z*c)
          h_t.append(C*hmask)

        # For the last layer
        h_tm1_l = h_t[-1]
        for Lparam in self.Lparams:
          a = T.dot(h_tm1_l, Lparam) +\
              self.bparams[-1]

          c = T.nnet.softmax(2*a)
          y_t.append(c)

        h_t.append(y_t)

        return h_t[1:]
    
      ##  The feedback function for generating
      def recur(x_t, *h_tm1):

        # For all but the last layer
        h_t = [x_t.flatten()]
        for h_tm1_l, Wparam, Uparam, bparam, hparam, hmask in zip(h_tm1, Wparam, Uparam, bparam, hparam, hmask):

          sliceLen = Wparam.shape[1]/3
          azr = T.dot(h_t[-1], Wparam) + \
              T.dot(h_tm1_l, Uparam) +\
              bparam
          s = T.nnet.sigmoid(2*hparam)
          z = T.nnet.sigmoid(2*(azr[:sliceLen]))
          r = T.nnet.sigmoid(2*(azr[sliceLen:2*sliceLen]))
          a = azr[:sliceLen]

          c = s*T.tanh(a) +\
              (1-s)*T.switch(a > 0, a, 0)
          C = ((1-z)*r*h_tm1_l + z*c)
          h_t.append(C*hmask)

        # For the last layer
        #TODO
    
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif self.model == 'LSTM':
      ##  The recurrence function for training/evaluating
      def recur(i, *hC_tm1):

        C_tm1  = hC_tm1[len(hC_tm1)/2:]
        h_tm1  = hC_tm1[:len(hC_tm1)/2]
        # For all but the last layer
        C_t = []
        h_t = [self.x[i:i+self.window].flatten()]
        for h_tm1_l, C_tm1_l, layerDatum, Wparam, Uparam, bparam, hparam, hmask in zip(h_tm1, C_tm1, layerData, Wparam, Uparam, bparam, hparam, hmask):

          sliceLen = Wparam.shape[1]/4
          aifo = T.dot(h_t[-1], Wparam) +\
              T.dot(h_tm1_l, Uparam) +\
              bparam
          s = T.nnet.sigmoid(2*hparam)
          i = T.nnet.sigmoid(2*aifo[sliceLen:2*sliceLen])
          f = T.nnet.sigmoid(2*aifo[2*sliceLen:3*sliceLen])
          o = T.nnet.sigmoid(2*aifo[3*sliceLen:])
          a = aifo[:sliceLen]

          c = s*T.tanh(a) +\
              (1-s)*T.switch(a > 0, a, 0)
          C = i*c + f*C_tm1_l
          C_t.append(C)

          h = o*(s*T.tanh(C) +\
              (1-s)*T.switch(C > 0, C, 0))
          h_t.append(h*hmask)

        # For the last layer
        h_tm1_l = h_t[-1]
        for Lparam in self.Lparams:
          a = T.dot(h_tm1_l, Lparam) +\
              self.bparams[-1]
          c = T.nnet.softmax(2*a)
          y_t.append(c)

        C_t.append(y_t)
        h_t.append(y_t)

        return h_t[1:] + C_t
    
      ##  The feedback function for generating
      def recur(x_t, *hC_tm1):

        C_tm1  = hC_tm1[len(hC_tm1)/2:]
        h_tm1  = hC_tm1[:len(hC_tm1)/2]
        # For all but the last layer
        C_t = []
        h_t = [x_t.flatten()]
        for h_tm1_l, C_tm1_l, layerDatum, Wparam, Uparam, bparam, hparam, hmask in zip(h_tm1, C_tm1, layerData, Wparam, Uparam, bparam, hparam, hmask):

          sliceLen = Wparam.shape[1]/4
          aifo = T.dot(h_t[-1], Wparam) +\
              T.dot(h_tm1_l, Uparam) +\
              bparam
          s = T.nnet.sigmoid(2*hparam)
          i = T.nnet.sigmoid(2*aifo[sliceLen:2*sliceLen])
          f = T.nnet.sigmoid(2*aifo[2*sliceLen:3*sliceLen])
          o = T.nnet.sigmoid(2*aifo[3*sliceLen:])
          a = aifo[:sliceLen]

          c = s*T.tanh(a) +\
              (1-s)*T.switch(a > 0, a, 0)
          C = i*c + f*C_tm1_l
          C_t.append(C)

          h = o*(s*T.tanh(C) +\
              (1-s)*T.switch(C > 0, C, 0))
          h_t.append(h*hmask)

        # For the last layer
        #TODO
    
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif self.model == 'FastLSTM':
      ##  The recurrence function for training/evaluating
      def recur(i, *hC_tm1):
        C_tm1  = hC_tm1[len(hC_tm1)/2:]
        h_tm1  = hC_tm1[:len(hC_tm1)/2]
        # For all but the last layer
        C_t = []
        h_t = [self.x[i:i+self.window].flatten()]
        for h_tm1_l, C_tm1_l, layerDatum, Wparam, Uparam, bparam, hparam, hmask in zip(h_tm1, C_tm1, layerData, Wparam, Uparam, bparam, hparam, hmask):

          sliceLen = Wparam.shape[1]/4
          aio = T.dot(h_t[-1], Wparam) +\
              T.dot(h_tm1_l, Uparam) +\
              bparam
          s = T.nnet.sigmoid(2*hparam)
          i = T.nnet.sigmoid(2*aio[sliceLen:2*sliceLen])
          o = T.nnet.sigmoid(2*aio[3*sliceLen:])
          a = aio[:sliceLen]

          c = s*T.tanh(a) +\
              (1-s)*T.switch(a > 0, a, 0)
          C = i*c + (1-i)*C_tm1_l
          C_t.append(C)

          h = o*(s*T.tanh(C) +\
              (1-s)*T.switch(C > 0, C, 0))
          h_t.append(h*hmask)

        # For the last layer
        h_tm1_l = h_t[-1]
        for Lparam in self.Lparams:
          a = T.dot(h_tm1_l, Lparam) +\
              self.bparams[-1]
          c = T.nnet.softmax(2*a)
          y_t.append(c)

        C_t.append(y_t)
        h_t.append(y_t)

        return h_t[1:] + C_t

      ##  The feedback function for generating
      def recur(x_t, *hC_tm1):
        C_tm1  = hC_tm1[len(hC_tm1)/2:]
        h_tm1  = hC_tm1[:len(hC_tm1)/2]
        # For all but the last layer
        C_t = []
        h_t = [x_t.flatten()]
        for h_tm1_l, C_tm1_l, layerDatum, Wparam, Uparam, bparam, hparam, hmask in zip(h_tm1, C_tm1, layerData, Wparam, Uparam, bparam, hparam, hmask):

          sliceLen = Wparam.shape[1]/4
          aio = T.dot(h_t[-1], Wparam) +\
              T.dot(h_tm1_l, Uparam) +\
              bparam
          s = T.nnet.sigmoid(2*hparam)
          i = T.nnet.sigmoid(2*aio[sliceLen:2*sliceLen])
          o = T.nnet.sigmoid(2*aio[3*sliceLen:])
          a = aio[:sliceLen]

          c = s*T.tanh(a) +\
              (1-s)*T.switch(a > 0, a, 0)
          C = i*c + (1-i)*C_tm1_l
          C_t.append(C)

          h = o*(s*T.tanh(C) +\
              (1-s)*T.switch(C > 0, C, 0))
          h_t.append(h*hmask)

        # For the last layer
        # TODO

    h, _ = theano.scan(
        fn=recur,
        sequences=T.arange(self.x.shape[0]-(self.window-1)),
        outputs_info=self.outputs_info)
    g, _ = theano.scan(
        fn=feedback,
        sequences=[],
        outputs_info=[g_0]+self.h_0,
        n_steps=512)
    yhat = h[-len(self.libs):]
    y_   = g[0][:,-1]

    #-------------------------------------------------------------------
    # Build the cost variable 
    self.crossentropy = T.sum([T.mean(T.nnet.categorical_crossentropy(yhat[i], self.y[:,i]) for i in xrange(len(self.libs)))])
    #self.perplexity = T.exp(

    self.complexity = 0
    if self.L1reg > 0:
      self.complexity += self.L1reg*T.sum([T.sum(T.abs_(Wparam)) for Wparam in self.Wparams])
      self.complexity += self.L1reg*T.sum([T.sum(T.abs_(Uparam)) for Uparam in self.Uparams])
      self.complexity += self.L1reg*T.sum([T.sum(T.abs_(lparam)) for i, lparam in enumerate(self.lparams) if self.libs[i].mutable()])
    if self.L2reg > 0:
      self.complexity += self.L2reg*T.sum([T.sum(T.sqr(Wparam)) for Wparam in self.Wparams])/2.
      self.complexity += self.L2reg*T.sum([T.sum(T.sqr(Uparam)) for Uparam in self.Uparams])/2.
      self.complexity += self.L2reg*T.sum([T.sum(T.abs_(lparam)) for i, lparam in enumerate(self.lparams) if self.libs[i].mutable()])/2.

    #===================================================================
    # Activate
    self.i2v = theano.function(
        inputs=[self.idxs]
        outputs=yhat
        allow_input_downcast=True)

    #===================================================================
    # Activate
    self.i2p = theano.function(
        inputs=[self.idxs]
        outputs=phat
        allow_input_downcast=True)

    #===================================================================
    # Loss
    self.i2c = theano.function(
        inputs=[self.idxs],
        outputs=self.J,
        allow_input_downcast=True)

    #===================================================================
    # Gradients
    self.i2g = theano.function(
        inputs=[self.idxs, self.y],
        outputs=[self.J] + T.grad(self.J, self.params),
        allow_input_downcast=True)
    
    #===================================================================
    # Update Gradients
    batchSize = T.scalar('batchSize')
    self.update_grad = theano.function(
        inputs=[batchSize]+paramVars,
        outputs=[],
        updates=[(gparam, gparam + paramVar/batchSize) for gparam, paramVar in zip(self.gparams, paramVars)],
        allow_input_downcast=True)

    #===================================================================
    # Reset gradients
    self.reset_grad = theano.function(
        inputs=[],
        outputs=[],
        updates=[(gparam, 0*gparam) for gparam in self.gparams],
        allow_input_downcast=True)

  #=====================================================================
  # Converts a list of strings or string tuples into a vector
  def s2i(self, strings, reverse=False):
    """"""

    if reverse:
      strings.reverse()
      begins = tuple([lib.stop  for lib in self.libs])
      ends   = tuple([lib.start for lib in self.libs])
    else:
      begins = tuple([lib.start for lib in self.libs])
      ends   = tuple([lib.stop  for lib in self.libs])
    # Pad the beginning 
    nbegins = 0
    while tuple(strings[nbegins]) == begins:
      nbegins += 1
    strings = [begins]*(self.window-nbegins) + strings
    # Pad the end
    if tuple(strings[0]) != ends:
      strings.insert(ends)
    if not isinstance(strings[0], (tuple, list)):
      return self.libs[0].s2i(strings)
    else:
      return np.concatenate([lib.s2i([string[i] for string in strings]) for i, lib in enumerate(self.libs)], axis=1)

  #=====================================================================
  # Converts a list of strings or string tuples into a vector
  def s2v(self, strings, reverse=False):
    """"""

    return self.i2v(self.s2i(strings, reverse))

  #=====================================================================
  # Converts a list of inputs and a list of classes into a prediction
  def s2p(self, strings, idxs, reverse=False):
    """"""

    if reverse:
      idxs.reverse()
    return self.i2p(self.s2i(strings, reverse), np.array(idxs))

  #=====================================================================
  # Converts a list of inputs and a list of classes into a cost
  def s2c(self, strings, idxs, reverse=False):
    """"""

    if reverse:
      idxs.reverse()
    return self.i2c(self.s2i(strings, reverse), np.array(idxs))

  #=====================================================================
  # Converts a list of inputs and a list of classes into a cost
  def s2g(self, strings, idxs, reverse=False):
    """"""

    if reverse:
      idxs.reverse()
    return self.i2g(self.s2i(strings, reverse), np.array(idxs))

  #=====================================================================
  # Batch cost
  def convert_dataset(self, dataset, reverse=False):
    """"""

    return [(self.s2i(datum[0], reverse), datum[1]) for datum in dataset]

  #=====================================================================
  # Batch cost
  def cost(self, dataset, workers=2):
    """"""

    cost = 0
    dataQueue = mp.Queue()
    costQueue = mp.Queue()
    processes = []
    for datum in dataset:
      dataQueue.put(datum)
    for worker in xrange(workers):
      dataQueue.put('STOP')
    for worker in xrange(workers):
      process = mp.Process(target=func_worker, args=(dataQueue, costQueue, self.i2c))
      process.start()
      processes.append(process)
    for worker in xrange(workers):
      for cost_ in iter(costQueue.get, 'STOP'):
        cost += cost_
    for process in processes:
      process.join()
    return cost/len(dataset)

  #=====================================================================
  # Minibatch grads
  def train(self, dataset, optimizer, batchSize=60, epochs=1, costEvery=None, testset=None, saveEvery=None, savePipe=None, workers=2):
    """"""
    
    #-------------------------------------------------------------------
    # Saving and printing
    s = ''
    epochd = str(int(np.log10(epochs))+1)
    minibatchd = str(int(np.log10(len(dataset)/batchSize))+1)
    s += ('Minibatch %0'+epochd+'d-%0'+minibatchd+'d') % (0,0)
    cost = []
    cost.append(self.cost(dataset, workers))
    s += ': %.3f train error' % cost[-1]
    if testset is not None:
      test = []
      test.append(self.cost(testset, workers))
      s += ', %.3f test error' % test[-1]
    wps = 0.0
    s += ', %.1f data per second' % wps
    if saveEvery is not None:
      savePipe.send(cost)
      lastSaveTime = time.time()
      s += ', %.1f minutes since saving' % ((time.time()-lastSaveTime)/60)
    s += '        \r'
    print s,
    sys.stdout.flush()
    lastCostTime = time.time()

    #-------------------------------------------------------------------
    # Multiprocessing the minibatch
    dataQueue = mp.Queue()
    gradQueue = mp.Queue()
    recentCost = []
    for t in xrange(epochs):
      np.random.shuffle(dataset)
      for mb in xrange(len(dataset)/batchSize):
        processes = []
        for datum in dataset[mb*batchSize:(mb+1)*batchSize]:
          dataQueue.put(datum)
        for worker in xrange(workers):
          dataQueue.put('STOP')
        for worker in xrange(workers):
          process = mp.Process(target=func_worker, args=(dataQueue, gradQueue, self.i2g))
          process.start()
          processes.append(process)
        for worker in xrange(workers):
          for grad in iter(gradQueue.get, 'STOP'):
            recentCost.append(grad[0])
            self.update_grad(batchSize, *grad[1:])
        for process in processes:
          process.join()
        optimizer()
        self.reset_grad()

        #---------------------------------------------------------------
        # More printing and saving
        if costEvery is not None  and (mb+1) % costEvery == 0:
          cost.append(np.sum(recentCost)/(batchSize*costEvery))
          recentCost = []
          if testset is not None:
            test.append(self.cost(testset, workers))
          thisCostTime = time.time()
        if saveEvery is not None and (mb+1) % saveEvery == 0:
          savePipe.send((cost,) + ((test,) if testset is not None else tuple()))
          lastSaveTime = time.time()
        if costEvery is not None and (mb+1) % costEvery == 0:
          s = ''
          s += ('Minibatch %0'+epochd+'d-%0'+minibatchd+'d') % (t+1,mb+1)
          s += ': %.3f train error' % cost[-1]
          if testset is not None:
            s += ': %.3f test error' % (test[-1])
          if wps == 0:
            wps = ((batchSize*costEvery) / (thisCostTime-lastCostTime))
          else:
            wps = .67*wps + .33*((batchSize*costEvery) / (thisCostTime-lastCostTime))
          s += ', %.1f data per second' % wps
          if saveEvery is not None:
            s += ', %.1f minutes since saving' % ((time.time()-lastSaveTime)/60)
          s += '        \r'
          print s,
          sys.stdout.flush()
          lastCostTime = time.time()

      #-----------------------------------------------------------------
      # If we haven't been printing, print now
      if costEvery is None or (mb+1) % costEvery != 0:
        cost.append(np.sum(recentCost)/(len(dataset) - mb*batchSize))
        recentCost = []
        if testset is not None:
          test.append(self.cost(testset, workers))
        thisCostTime = time.time()
      if saveEvery is not None and (mb+1) % saveEvery != 0:
        savePipe.send((cost,) + ((test,) if testset is not None else tuple()))
        lastSaveTime = time.time()
      s = ''
      s += ('Minibatch %0'+epochd+'d-%0'+minibatchd+'d') % (t+1,mb+1)
      s += ': %.3f train error' % cost[-1]
      if testset is not None:
        s += ': %.3f test error' % test[-1]
      if wps == 0:
        wps = ((batchSize*(mb+1 % costEvery) if costEvery is not None else len(dataset)) / (thisCostTime-lastCostTime))
      else:
        wps = .67*wps + .33*((batchSize*((mb+1) % costEvery) if costEvery is not None else len(dataset)) / (thisCostTime-lastCostTime))
      s += ', %.1f data per second' % wps 
      if saveEvery is not None:
        s += ', %.1f minutes since saving' % ((time.time()-lastSaveTime)/60)
      s += '        \r'
      print s,
      sys.stdout.flush()
      if costEvery is None or mb+1 % costEvery != 0:
        lastCostTime = time.time()

    #-------------------------------------------------------------------
    # Wrap everything up
    if savePipe is not None:
      savePipe.send('STOP')
    print ''
    return cost

#***********************************************************************
# A (multilayer) basic recurrent network encoder
class Encoder(Opt):
  """"""

  #=====================================================================
  # Initialize the model
  def __init__(self, libs, dims, **kwargs):
    """"""

    #-------------------------------------------------------------------
    # Keyword arguments
    if 'updateLibs' in kwargs:
      self.updateLibs = kwargs['updateLibs']
    else:
      self.updateLibs = 1
      
    if 'model' in kwargs:
      self.model = kwargs['model']
    else:
      self.model = 1

    if 'window' in kwargs:
      self.window = kwargs['window']
    else:
      self.window = 1

    if 'ofunc' in kwargs:
      self.ofunc = kwargs['ofunc']
    else:
      self.ofunc = 'linear'

    if self.ofunc == 'linear':
      def ofunc (z): return z
      def Jfunc (yhat, y): return squared_difference(yhat,y)l
    elif self.ofunc == 'sigmoid':
      def ofunc (z): return T.nnet.sigmoid(2*z)
      def Jfunc (yhat, y): return T.nnet.binary_crossentropy(yhat,y)
    elif self.ofunc == 'softmax':
      def ofunc (z):
        #return T.nnet.softmax(2*z) # Doesn't work =(
        e_z = T.exp(z - z.max(axis=0, keepdims=True))
        return e_z / e_z.sum(axis=0, keepdims=True)
      def Jfunc (yhat, y): return T.nnet.categorical_crossentropy(yhat,y)
    else:
      raise ValueError('Bad value for ofunc, must be linear, sigmoid, or softmax')

    if 'L1reg' in kwargs:
      self.L1reg = kwargs['L1reg']
    else:
      self.L1reg = 0.

    if 'L2reg' in kwargs:
      self.L2reg = kwargs['L2reg']
    else:
      self.L2reg = 0.

    #-------------------------------------------------------------------
    # Process the libraries
    self.libs = []
    libsize = 0
    for lib in libs:
      if not isinstance(lib, Library):
        lib = Library(*lib)
      self.libs.append(Library(*lib))
      libsize += lib.L.get_value().shape[1]
    libsize *= self.window
    dims.insert(0, libsize)

    #-------------------------------------------------------------------
    # Bundle the model params
    self.model = model
    if self.model in ('RNN',)
      gates = 1
    elif self.model in ('GRU', 'FastGRU', 'FastLSTM'):
      gates = 3
    elif self.model in ('LSTM',):
      gates = 4
    self.Wparams = []
    self.Uparams = []
    self.bparams = []
    self.hparams = []
    self.hmasks  = []
    self.h_0     = []
    self.layerData = np.ones(len(dims), dtype=np.float32)
    self.layerData[-1] = 0
    for i in xrange(1,len(dims)):
      W = mw.rect_mat(i-1, i*gates)
      self.Wparams.append(theano.shared(W, name='W-%d' % (i+1)))
      if self.ofunc == 'linear' or i < len(dims)-1: 
        self.layerData[i] = 1
        U = np.concatenate([mw.diag_mat(i-1, i), mw.rect_mat(i-1, i*(gates-1))], axis=1)
        self.Uparams.append(theano.shared(U, name='U-%d' % (i+1)))
        h = np.zeros(i*gates)
        self.hparams.append(theano.shared(h, name='h-%d' % (i+1)))
      else:
        self.Uparams.append(None)
        self.hparams.append(None)
      b = np.zeros(i*gates)
      self.bparams.append(theano.shared(b, name='b-%d' % (i+1)))
      h_0 = np.zeros(i*gates)
      self.h_0.append(theano.shared(h_0, name='h_0-%d' % (i+1)))
      hmask = np.ones(i*gates)
      self.hmasks.append(theano.shared(hmask, name='hmask-%d' % (i+1)))
    if self.model.endswith('LSTM'):
      C_0 = [theano.shared(np.zeros_like(h_0_l.get_value()), name='C'+h_0_l.name[1:]) for h_0_l in h_0]
      self.h_0 = self.h_0 + C_0
    self.params =\
        self.Wparams +\
        self.Uparams +\
        self.bparams +\
        self.hparams +\
        self.h_0
    self.gparams = [theano.shared(np.zeros_like(param.get_value(), name='g'+param.name, dtype='float32')) for param in self.params]
    paramVars =\
        [T.matrix() for Wparam in self.Wparams] +\
        [T.matrix() for Uparam in self.Uparams] +\
        [T.vector() for bparam in self.bparams] +\
        [T.vector() for hparam in self.hparams] +\
        [T.vector() for h_0_l  in self.h_0]

    #-------------------------------------------------------------------
    # Build the input/output variables
    # Some of this can be done in the Library class
    self.idxs    = T.imatrix('idxs')
    self.lparams = []
    for i, lib in enumerate(self.libs)
      self.lparams.append(lib.get_subtensor(self.idxs[:,i]))
    self.x       = T.concatenate(self.lparams, axis=1)
    self.y       = T.ivector('y')
    batchSize = T.scalar('batchSize')

    #-------------------------------------------------------------------
    # Build the activation variable

    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if self.model == 'RNN':
      def recur(i, *h_tm1):
        h_t = [self.x[i:i+self.window].flatten()]
        for h_tm1_l, layerDatum, Wparam, Uparam, bparam, hparam, hmask in zip(h_tm1, layerData, Wparam, Uparam, bparam, hparam, hmask):
          
          # Calculate the activation
          if Uparam is not None:
            a = T.dot(h_t[-1], Wparam) +\
                T.dot(h_tm1_l, Uparam) +\
                bparam
          else:
            a = T.dot(h_t[-1], Wparam) +\
                bparam

          # Apply the nonlinearity and dropout
          if layerDatum:
            s = T.nnet.sigmoid(2*hparam)
            c = (s*T.tanh(a) +\
                (1-s)*T.switch(a > 0, a, 0))*hmask
          else:
            c = ofunc(a) 
          h_t.append(c)
        return h_t[1:]

    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif self.model == 'GRU':
      def recur(i, *h_tm1):
        h_t = [self.x[i:i+self.window].flatten()]
        for h_tm1_l, layerDatum, Wparam, Uparam, bparam, hparam, hmask in zip(h_tm1, layerData, Wparam, Uparam, bparam, hparam, hmask):

          # Calculate the activation
          if Uparam is not None:
            sliceLen = Wparam.shape[1]/3
            azr_W = T.dot(h_t[-1], Wparam)
            zr_U  = T.dot(h_tm1_l, Uparam[:,sliceLen:])
            z = T.nnet.sigmoid(2*(
                azr_W[sliceLen:2*sliceLen] +\
                zr_U[:sliceLen] +\
                bparam[sliceLen:2*sliceLen]))
            r = T.nnet.sigmoid(2*(
                azr_W[2*sliceLen:] +\
                zr_U[sliceLen:] +\
                bparam[2*sliceLen:]))
            a = azr_W[:sliceLen] +\
                T.dot(r*h_tm1_l, Uparam[:,:sliceLen]) +\
                bparam[:sliceLen]
          else:
            a = T.dot(h_t[-1], Wparam) +\
                bparam

          # Apply the nonlinearity
          if layerDatum:
            s = T.nnet.sigmoid(2*hparam)
            c = s*T.tanh(a) +\
                (1-s)*T.switch(a > 0, a, 0)
          else:
            c = ofunc(a)

          # Calculate the post-nonlinearity activation
          if Uparam is not None:
            C = ((1-z)*h_tm1_l + z*c)
          else:
            C = c

          # Apply the dropout
          if layerDatum:
            h_t.append(C*hmask)
          else:
            h_t.append(C)
        return h_t[1:]

    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif self.model == 'FastGRU':
      def recur(i, *h_tm1):
        h_t = [self.x[i:i+self.window].flatten()]
        for h_tm1_l, layerDatum, Wparam, Uparam, bparam, hparam, hmask in zip(h_tm1, layerData, Wparam, Uparam, bparam, hparam, hmask):

          # Calculate the activation
          if Uparam is not None:
            sliceLen = Wparam.shape[1]/3
            azr = T.dot(h_t[-1], Wparam) + \
                T.dot(h_tm1_l, Uparam) +\
                bparam

            z = T.nnet.sigmoid(2*(azr[:sliceLen]))
            r = T.nnet.sigmoid(2*(azr[sliceLen:2*sliceLen]))
            a = azr[:sliceLen]
          else:
            a = T.dot(h_t[-1], Wparam) +\
                bparam

          # Apply the nonlinearity
          if layerDatum:
            s = T.nnet.sigmoid(2*hparam)
            c = s*T.tanh(a) +\
                (1-s)*T.switch(a > 0, a, 0)
          else:
            c = ofunc(a)

          # Calculate the post-nonlinearity activation
          if Uparam is not None:
            C = ((1-z)*r*h_tm1_l + z*c)
          else:
            C = c

          # Apply the dropout
          if layerDatum:
            h_t.append(C*hmask)
          else:
            h_t.append(C)
        return h_t[1:]
    
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif self.model == 'LSTM':
      def recur(i, *hC_tm1):
        C_tm1  = hC_tm1[len(hC_tm1)/2:]
        h_tm1  = hC_tm1[:len(hC_tm1)/2]
        h_t = [self.x[i:i+self.window].flatten()]
        C_t = []
        for h_tm1_l, C_tm1_l, layerDatum, Wparam, Uparam, bparam, hparam, hmask in zip(h_tm1, C_tm1, layerData, Wparam, Uparam, bparam, hparam, hmask):

          s = None
          # Calculate the activation
          if Uparam is not None:
            sliceLen = Wparam.shape[1]/4
            aifo = T.dot(h_t[-1], Wparam) +\
                T.dot(h_tm1_l, Uparam) +\
                bparam
            i = T.nnet.sigmoid(2*aifo[sliceLen:2*sliceLen])
            f = T.nnet.sigmoid(2*aifo[2*sliceLen:3*sliceLen])
            o = T.nnet.sigmoid(2*aifo[3*sliceLen:])
            a = aifo[:sliceLen]
          else:
            a = T.dot(h_t[-1], Wparam)+\
                bparam

          # Apply the first nonlinearity
          if layerDatum:
            s = T.nnet.sigmoid(2*hparam)
            c = s*T.tanh(a) +\
                (1-s)*T.switch(a > 0, a, 0)
          else:
            c = ofunc(a)

          # Calculate the cell activation & apply the second nonlinearity
          if Uparam is not None:
            C = i*c + f*C_tm1_l
            C_t.append(C)
            if s is None:
              s = T.nnet.sigmoid(2*hparam)
            h = o*(s*T.tanh(C) +\
                (1-s)*T.switch(C > 0, C, 0))
          else:
            C = c
            C_t.append(C)
            h = C

          # Apply the dropout
          if layerDatum:
            h_t.append(h*hmask)
          else:
            h_t.append(h)
          return h_t[1:] + C_t

    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif self.model == 'FastLSTM':
      def recur(i, *hC_tm1):
        C_tm1  = hC_tm1[len(hC_tm1)/2:]
        h_tm1  = hC_tm1[:len(hC_tm1)/2]
        h_t = [self.x[i:i+self.window].flatten()]
        C_t = []
        for h_tm1_l, C_tm1_l, layerDatum, Wparam, Uparam, bparam, hparam, hmask in zip(h_tm1, C_tm1, layerData, Wparam, Uparam, bparam, hparam, hmask):

          s = None
          # Calculate the activation
          if Uparam is not None:
            sliceLen = Wparam.shape[1]/3
            aio = T.dot(h_t[-1], Wparam) +\
                T.dot(h_tm1_l, Uparam) +\
                bparam
            i = T.nnet.sigmoid(2*aio[sliceLen:2*sliceLen])
            o = T.nnet.sigmoid(2*aio[2*sliceLen:3*sliceLen])
            a = aifo[:sliceLen]
          else:
            a = T.dot(h_t[-1], Wparam)+\
                bparam

          # Apply the first nonlinearity
          if layerDatum:
            s = T.nnet.sigmoid(2*hparam)
            c = s*T.tanh(a) +\
                (1-s)*T.switch(a > 0, a, 0)
          else:
            c = ofunc(a)

          # Calculate the cell activation & apply the second nonlinearity
          if Uparam is not None:
            C = i*c + (1-i)*C_tm1_l
            C_t.append(C)
            if s is None:
              s = T.nnet.sigmoid(2*hparam)
            h = o*(s*T.tanh(C) +\
                (1-s)*T.switch(C > 0, C, 0))
          else:
            C = c
            C_t.append(C)
            h = C

          # Apply the dropout
          if layerDatum:
            h_t.append(h*hmask)
          else:
            h_t.append(h)
          return h_t[1:] + C_t

    self.h, _ = theano.scan(
        fn=recur,
        sequences=T.arange(self.x.shape[0]-(self.window-1)),
        outputs_info=self.h_0)

    #-------------------------------------------------------------------
    # Build the cost variable 
    self.J = T.mean(Jfunc(self.h[-1][-1], self.y))
    if self.L1reg > 0:
      self.J += T.sum([T.sum(T.abs_(Wparam)) for Wparam in self.Wparams])
      self.J += T.sum([T.sum(T.abs_(Uparam)) for Uparam in self.Uparams])
    if self.L2reg > 0:
      self.J += T.sum([T.sum(T.sqr(Wparam)) for Wparam in self.Wparams])
      self.J += T.sum([T.sum(T.sqr(Uparam)) for Uparam in self.Uparams])

    #===================================================================
    # Activate
    self.i2v = theano.function(
        inputs=[self.idxs]
        outputs=self.h[-1][-1],
        allow_input_downcast=True)

    #===================================================================
    # Activate
    self.i2p = theano.function(
        inputs=[self.idxs]
        outputs=T.argmax(self.h[-1][-1]),
        allow_input_downcast=True)

    #===================================================================
    # Loss
    self.i2c = theano.function(
        inputs=[self.idxs, self.y],
        outputs=self.J,
        allow_input_downcast=True)

    #===================================================================
    # Gradients
    self.i2g = theano.function(
        inputs=[self.idxs, self.y],
        outputs=[self.J] + T.grad(self.J, self.params),
        allow_input_downcast=True)
    
    #===================================================================
    # Update Gradients
    batchSize = T.scalar('batchSize')
    self.update_grad = theano.function(
        inputs=[batchSize]+paramVars,
        outputs=[],
        updates=[(gparam, gparam + paramVar/batchSize) for gparam, paramVar in zip(self.gparams, paramVars)]
        allow_input_downcast=True)

    #===================================================================
    # Reset gradients
    self.reset_grad = theano.function(
        inputs=[],
        outputs=[],
        updates=[(gparam, 0*gparam) for gparam in self.gparams],
        allow_input_downcast=True)

  #=====================================================================
  # Converts a list of strings or string tuples into a vector
  def s2i(self, strings, reverse=False)
    """"""

    if reverse:
      strings.reverse()
      begins = tuple([lib.stop  for lib in self.libs])
      ends   = tuple([lib.start for lib in self.libs])
    else:
      begins = tuple([lib.start for lib in self.libs])
      ends   = tuple([lib.stop  for lib in self.libs])
    if tuple(strings[0]) != begin:
      strings.insert(begins)
    nends = 0
    while tuple(strings[-(nends+1)]) == end:
      nends += 1
    for i in xrange(self.window-nends):
      strings.append(end)
    if not isinstance(strings[0], (tuple, list)):
      return self.libs[0].s2i(strings)
    else:
      return np.concatenate([libs.s2i([string[i] for string in strings]) for i, lib in enumerate(self.libs)], axis=1)

  #=====================================================================
  # Converts a list of strings or string tuples into a vector
  def s2v(self, strings, reverse=False):
    """"""

    return self.i2v(self.s2i(strings, reverse))

  #=====================================================================
  # Converts a list of inputs and a list of classes into a prediction
  def s2p(self, strings, idxs, reverse=False):
    """"""

    if reverse:
      idxs.reverse()
    return self.i2p(self.s2i(strings, reverse), np.array(idxs))

  #=====================================================================
  # Converts a list of inputs and a list of classes into a cost
  def s2c(self, strings, idxs, reverse=False):
    """"""

    if reverse:
      idxs.reverse()
    return self.i2c(self.s2i(strings, reverse), np.array(idxs))

  #=====================================================================
  # Converts a list of inputs and a list of classes into a cost
  def s2g(self, strings, idxs, reverse=False):
    """"""

    if reverse:
      idxs.reverse()
    return self.i2g(self.s2i(strings, reverse), np.array(idxs))

  #=====================================================================
  # Batch cost
  def convert_dataset(self, dataset, reverse=False):
    """"""

    return [(self.s2i(datum[0], reverse), datum[1]) for datum in dataset]

  #=====================================================================
  # Batch cost
  def cost(self, dataset, workers=2):
    """"""

    cost = 0
    dataQueue = mp.Queue()
    costQueue = mp.Queue()
    processes = []
    for datum in dataset:
      dataQueue.put(datum)
    for worker in xrange(workers):
      dataQueue.put('STOP')
    for worker in xrange(workers):
      process = mp.Process(target=func_worker, args=(dataQueue, costQueue, self.i2c))
      process.start()
      processes.append(process)
    for worker in xrange(workers):
      for cost_ in iter(costQueue.get, 'STOP'):
        cost += cost_
    for process in processes:
      process.join()
    return cost/len(dataset)

  #=====================================================================
  # Minibatch grads
  def train(self, dataset, optimizer, batchSize=60, epochs=1, costEvery=None, testset=None, saveEvery=None, savePipe=None, workers=2):
    """"""
    
    #-------------------------------------------------------------------
    # Saving and printing
    s = ''
    epochd = str(int(np.log10(epochs))+1)
    minibatchd = str(int(np.log10(len(dataset)/batchSize))+1)
    s += ('Minibatch %0'+epochd+'d-%0'+minibatchd+'d') % (0,0)
    cost = []
    cost.append(self.cost(dataset, workers))
    s += ': %.3f train error' % cost[-1]
    if testset is not None:
      test = []
      test.append(self.cost(testset, workers))
      s += ', %.3f test error' % test[-1]
    wps = 0.0
    s += ', %.1f data per second' % wps
    if saveEvery is not None:
      savePipe.send(cost)
      lastSaveTime = time.time()
      s += ', %.1f minutes since saving' % ((time.time()-lastSaveTime)/60)
    s += '        \r'
    print s,
    sys.stdout.flush()
    lastCostTime = time.time()

    #-------------------------------------------------------------------
    # Multiprocessing the minibatch
    dataQueue = mp.Queue()
    gradQueue = mp.Queue()
    recentCost = []
    for t in xrange(epochs):
      np.random.shuffle(dataset)
      for mb in xrange(len(dataset)/batchSize):
        processes = []
        for datum in dataset[mb*batchSize:(mb+1)*batchSize]:
          dataQueue.put(datum)
        for worker in xrange(workers):
          dataQueue.put('STOP')
        for worker in xrange(workers):
          process = mp.Process(target=func_worker, args=(dataQueue, gradQueue, self.i2g))
          process.start()
          processes.append(process)
        for worker in xrange(workers):
          for grad in iter(gradQueue.get, 'STOP'):
            recentCost.append(grad[0])
            self.update_grad(batchSize, *grad[1:])
        for process in processes:
          process.join()
        optimizer()
        self.reset_grad()

        #---------------------------------------------------------------
        # More printing and saving
        if costEvery is not None  and (mb+1) % costEvery == 0:
          cost.append(np.sum(recentCost)/(batchSize*costEvery))
          recentCost = []
          if testset is not None:
            test.append(self.cost(testset, workers))
          thisCostTime = time.time()
        if saveEvery is not None and (mb+1) % saveEvery == 0:
          savePipe.send((cost,) + ((test,) if testset is not None else tuple()))
          lastSaveTime = time.time()
        if costEvery is not None and (mb+1) % costEvery == 0:
          s = ''
          s += ('Minibatch %0'+epochd+'d-%0'+minibatchd+'d') % (t+1,mb+1)
          s += ': %.3f train error' % cost[-1]
          if testset is not None:
            s += ': %.3f test error' % (test[-1])
          if wps == 0:
            wps = ((batchSize*costEvery) / (thisCostTime-lastCostTime))
          else:
            wps = .67*wps + .33*((batchSize*costEvery) / (thisCostTime-lastCostTime))
          s += ', %.1f data per second' % wps
          if saveEvery is not None:
            s += ', %.1f minutes since saving' % ((time.time()-lastSaveTime)/60)
          s += '        \r'
          print s,
          sys.stdout.flush()
          lastCostTime = time.time()

      #-----------------------------------------------------------------
      # If we haven't been printing, print now
      if costEvery is None or (mb+1) % costEvery != 0:
        cost.append(np.sum(recentCost)/(len(dataset) - mb*batchSize))
        recentCost = []
        if testset is not None:
          test.append(self.cost(testset, workers))
        thisCostTime = time.time()
      if saveEvery is not None and (mb+1) % saveEvery != 0:
        savePipe.send((cost,) + ((test,) if testset is not None else tuple()))
        lastSaveTime = time.time()
      s = ''
      s += ('Minibatch %0'+epochd+'d-%0'+minibatchd+'d') % (t+1,mb+1)
      s += ': %.3f train error' % cost[-1]
      if testset is not None:
        s += ': %.3f test error' % test[-1]
      if wps == 0:
        wps = ((batchSize*(mb+1 % costEvery) if costEvery is not None else len(dataset)) / (thisCostTime-lastCostTime))
      else:
        wps = .67*wps + .33*((batchSize*((mb+1) % costEvery) if costEvery is not None else len(dataset)) / (thisCostTime-lastCostTime))
      s += ', %.1f data per second' % wps 
      if saveEvery is not None:
        s += ', %.1f minutes since saving' % ((time.time()-lastSaveTime)/60)
      s += '        \r'
      print s,
      sys.stdout.flush()
      if costEvery is None or mb+1 % costEvery != 0:
        lastCostTime = time.time()

    #-------------------------------------------------------------------
    # Wrap everything up
    if savePipe is not None:
      savePipe.send('STOP')
    print ''
    return cost

#***********************************************************************
# Test the model
if __name__ == '__main__':
  """"""

  import sys

  wsize = 50
  i = 1
  while i < len(sys.argv):
    flag = sys.argv.pop(0)
    if flag in ('-w', '--wordSize'):
      wsize = sys.argv.pop(0)

  data = pkl.load(open('HobbitOpening-data.pkl'))
  lib  = Library(pkl.load(open('HobbitOpening-vocab.pkl'), wsize))
