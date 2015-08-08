#!/usr/bin/env python

import numpy as np
from matwizard import matwizard
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import cPickle as pkl
import multiprocessing as mp
import sys
sys.setrecursionlimit(50000)
#TODO fix dropout (apply it when calculating grads, not in the optimization funx)
#TODO fix momentum (apply it when calculating grads, not in the optimization funx)

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
# Scaled logistic sigmoid
def sig(x):
  """"""
  
  return np.float32(4)*T.nnet.sigmoid(np.float32(2)*x)

#=======================================================================
# Scaled hyperbolic tangent 
def tanh(x):
  """"""
  
  return np.float32(2)*T.tanh(x)

#=======================================================================
# Scaled rectifier
def relu(x):
  """"""
  
  return np.float32(1)/(np.arctanh(np.sqrt(np.float32(1)/np.float32(3))*np.sqrt(np.float32(1)-np.float32(2)/np.float32(np.pi))))*T.switch(x > 0, x, np.float32(0))

#=======================================================================
# Interpolater between tanh and relu
def func(x, s):
  """"""

  return s*tanh(x) + (np.float32(1)-s)*relu(x)

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
# An interface for sparse optimizaiton functions
class SpOpt:
      
  #=====================================================================
  # Run SGD (with NAG)
  def SGD(self, eta_0=.01, T_eta=1, mu_max=.95, T_mu=1, dropout=1., anneal=0, accel=0):
    """"""

    #-------------------------------------------------------------------
    # Set up the updates & givens
    updates = []
    givens = []

    #-------------------------------------------------------------------
    # Set up a variable to keep track of the iteration
    tau = theano.shared(np.float32(0), name='tau')
    updates.extend([(tau, tau+1)])

    #-------------------------------------------------------------------
    # Set the annealing/acceleration schedule
    eta = eta_0*T.pow(T_eta/(tau+T_eta), anneal)
    mu  = mu_max*(1-T.pow(T_mu/(tau+T_mu), accel))

    #-------------------------------------------------------------------
    # Initialize stored vectors
    vL = theano.shared(np.zeros_like(self.L.get_value()), name='vL')

    #-------------------------------------------------------------------
    # Subtense the tensors
    gidxs = T.ivector('gidxs')
    x = self.L[gidxs]
    gx = self.gL[gidxs]
    vx = vL[gidxs]
    grad_norm  = T.sqrt(T.sum(T.sqr(gx)))
    not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))

    #-------------------------------------------------------------------
    # Build up the updates
    muvx = mu*vx
    updates.append(T.inc_subtensor(x, T.switch(not_finite, -.9*x, -eta*gx)))
    updates.append(T.set_subtensor(vx, T.switch(not_finite, np.float32(0), muvx - eta*gx)))
    givens.append(T.inc_subtensor(x, muvx))

    #-------------------------------------------------------------------
    # Compile the sgd function
    opt = theano.function(
        inputs=[gidxs],
        outputs=[],
        givens=givens,
        updates=updates,
        allow_input_downcast=True)

    #-------------------------------------------------------------------
    # Return the compiled function
    print 'SGD function compiled'
    return opt

  #=====================================================================
  # Run AdaGrad (with NAG)
  def AdaGrad(self, eta_0=.01, T_eta=1, mu_max=.95, T_mu=1, epsilon=1e-6, dropout=1., anneal=0, accel=0):
    """"""

    #-------------------------------------------------------------------
    # Set up the updates & givens
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
    g2L = theano.shared(np.zeros_like(self.L.get_value()), name='g2L')
    vL  = theano.shared(np.zeros_like(self.L.get_value()), name='vL')

    #-------------------------------------------------------------------
    # Subtense the tensors
    gidxs = T.ivector('gidxs')
    x   = self.L[gidxs]
    gx  = self.gL[gidxs]
    vx  = vL[gidxs]
    g2x = g2L[gidxs]
    grad_norm  = T.sqrt(T.sum(T.sqr(gx)))
    not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))

    #-------------------------------------------------------------------
    # Build up the updates
    g2x_t = g2x + T.sqr(gx)
    deltax_t = gx/T.sqrt(g2x_t+epsilon)
    muvx = mu*vx

    updates.append(T.inc_subtensor(x, T.switch(not_finite, -.9*gx, -eta*deltax_t)))
    updates.append(T.set_subtensor(g2x, T.switch(not_finite, np.float32(0), g2x_t)))
    updates.append(T.set_subtensor(vx, T.switch(not_finite, np.float32(0), muvx - eta*deltax_t)))
    givens.append(T.inc_subtensor(x, muvx))

    #-------------------------------------------------------------------
    # Compile the sgd function
    opt = theano.function(
        inputs=[gidxs],
        outputs=[],
        givens=givens,
        updates=updates,
        allow_input_downcast=True)

    #-------------------------------------------------------------------
    # Return the compiled function
    print 'AdaGrad function compiled'
    return opt 

  #=====================================================================
  # Run RMSProp (with NAG)
  def RMSProp(self, eta_0=.01, T_eta=1, rho_0=.9, T_rho=1, mu_max=.95, T_mu=1, epsilon=1e-6, dropout=1., anneal=0, expand=0, accel=0):
    """"""

    #-------------------------------------------------------------------
    # Set up the updates & givens
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
    g2L = theano.shared(np.zeros_like(self.L.get_value(), dtype='float32'), name='g2L')
    vL  = theano.shared(np.zeros_like(self.L.get_value(), dtype='float32'), name='vL')

    #-------------------------------------------------------------------
    # Subtense the tensors
    gidxs    = T.ivector('gidxs')
    x   = self.L[gidxs]
    gx  = self.gL[gidxs]
    vx  = vL[gidxs]
    g2x = g2L[gidxs]
    grad_norm  = T.sqrt(T.sum(T.sqr(gx)))
    not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))

    #-------------------------------------------------------------------
    # Build up the updates
    g2x_t = rho*g2x + (1-rho)*T.sqr(gx)
    deltax_t = gx/T.sqrt(g2x+epsilon)
    muvx = mu*vx

    updates.append(T.inc_subtensor(x, T.switch(not_finite, -.9*gx, -eta*deltax_t)))
    updates.append(T.set_subtensor(g2x, T.switch(not_finite, np.float32(0), g2x_t)))
    updates.append(T.set_subtensor(vx, T.switch(not_finite, np.float32(0), muvx - eta*deltax_t)))
    givens.append(T.inc_subtensor(x, muvx))

    #-------------------------------------------------------------------
    # Compile the rmsprop function
    opt = theano.function(
        inputs=[gidxs],
        outputs=[],
        givens=givens,
        updates=updates,
        allow_input_downcast=True)

    print 'RMSProp function compiled'
    return opt
  
  #=====================================================================
  # Run AdaDelta
  def AdaDelta(self, eta_0=1., T_eta=1, rho_0=.9, T_rho=1, epsilon=1e-6, dropout=1., anneal=0, expand=0):
    """"""

    #-------------------------------------------------------------------
    # Set up the updates & givens
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
    g2L = theano.shared(np.zeros_like(self.L.get_value(), dtype='float32'), name='g2L')
    delta2L = theano.shared(np.zeros_like(self.L.get_value(), dtype='float32'), name='delta2L')
    vL  = theano.shared(np.zeros_like(self.L.get_value(), dtype='float32'), name='vL')

    #-------------------------------------------------------------------
    # Subtense the tensors
    gidxs    = T.ivector('gidxs')
    x       = self.L[gidxs]
    gx      = self.gL[gidxs]
    g2x     = g2L[gidxs]
    delta2x = delta2L[gidxs]
    grad_norm  = T.sqrt(T.sum(T.sqr(gx)))
    not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))

    #-------------------------------------------------------------------
    # Build up the updates
    g2x_t = rho*g2x + (1-rho)*T.sqr(gx)
    deltax_t = gx*T.sqrt(delta2x+epsilon)/T.sqrt(g2x+epsilon)
    delta2x_t = rho*delta2x + (1-rho)*T.sqr(deltax_t)
    
    updates.append(T.inc_subtensor(x, T.switch(not_finite, -.9*x, -eta*deltax_t)))
    updates.append(T.set_subtensor(delta2x, T.switch(not_finite, np.float32(0), delta2x_t)))
    updates.append(T.set_subtensor(g2x, T.switch(not_finite, np.float32(0), g2x_t)))

    #-------------------------------------------------------------------
    # Compile the adadelta function
    opt = theano.function(
        inputs=[gidxs],
        outputs=[],
        givens=givens,
        updates=updates,
        allow_input_downcast=True)

    print 'AdaDelta function compiled'
    return opt

  #=====================================================================
  # Run Adam
  def Adam(self, eta_0=.05, T_eta=1, rho1_0=.9, rho2_0=.99, T_rho=1, epsilon=1e-6, dropout=1., anneal=0, expand=0):
    """"""

    #-------------------------------------------------------------------
    # Set up the updates & givens
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
    mL = theano.shared(np.zeros_like(self.L.get_value(), dtype='float32'), name='mL')
    vL = theano.shared(np.zeros_like(self.L.get_value(), dtype='float32'), name='vL')

    #-------------------------------------------------------------------
    # Subtense the tensors
    gidxs = T.ivector('gidxs')
    x = self.L[gidxs]
    gx = self.gL[gidxs]
    mx = mL[gidxs]
    vx = vL[gidxs]
    grad_norm  = T.sqrt(T.sum(T.sqr(gx)))
    not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))

    #-------------------------------------------------------------------
    # Build up the updates 
    mx_t = rho1*mx + (1-rho1)*gx) / (1-rho1)
    vx_t = rho2*vx + (1-rho2)*T.sqr(gx) / (1-rho2)
    deltax_t = mx_t / (T.sqrt(vx) + epsilon)

    updates.append(T.inc_subtensor(x, T.switch(not_finite, -.9*x, -eta*deltax_t)))
    updates.append(T.set_subtensor(mx, T.switch(not_finite, np.float32(0), mx_t)))
    updates.append(T.set_subtensor(vx, T.switch(not_finite, np.float32(0), vx_t)))

    #-------------------------------------------------------------------
    # Compile the adam function
    opt = theano.function(
        inputs=[],
        outputs=[],
        givens=givens,
        updates=updates,
        allow_input_downcast=True)
    
    print 'Adam function compiled'
    return opt 

#***********************************************************************
# A library
class Library(SpOpt):
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
      mat = np.random.randn(len(keys), mat).astype('float32')
      self.mutable = True
    else:
      assert self.start in keys
      assert self.stop in keys
      assert self.unk in keys
      mat = mat.astype('float32')
    self._wsize = mat.shape[1]

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
    self.gL = theano.shared(zeros_like(mat, dtype='float32'))
    self._gidxs = set()

    #=====================================================================
    # Convert idxs to vectors
    x = T.ivector(name='x')
    self.idxs_to_vecs = theano.function(
        inputs=[x],
        outputs=self.L[x],
        allow_input_downcast=True)

    #=====================================================================
    # Convert vectors to idxs 
    v = T.fmatrix(name='v')
    self.vecs_to_idxs = theano.function(
        inputs=[v],
        outputs=T.argmin(T.sum(squared_difference(self.L[None,:,:], v[:,None,:]), axis=2), axis=1),
        allow_input_downcast=True)

    #=====================================================================
    # Convert vectors to idxs 
    batchSize = T.scalar('batchSize')
    gxparams = T.matrix('gxparams')
    gidxs = T.vector('gidxs')
    update_grads = theano.function(
        inputs=[batchsize, grad, gidxs],
        outputs=[],
        updates=T.inc_subtensor(self.gL[gidxs], grad/batchSize),
        allow_input_downcast=True)

    #===================================================================
    # Reset gradients
    self.reset_grads = theano.function(
        inputs=[gidxs],
        outputs=[],
        updates=set_subtensor(self.gL[gidxs], np.float32(0)*self.gL[gidxs])
        allow_input_downcast=True)

  #=====================================================================
  # Update gradients
  def update_lib_grads(self, batchSize, grad, gidxs):
    """"""

    self.update_grads(batchSize, grad, gidxs)
    self._gidxs.update(gidxs)

  #=====================================================================
  # Reset gradients
  def reset_lib_grads(self):
    """"""
    
    self.reset_grads(self.gidxs())
    self._gidxs = set()
  
  #=====================================================================
  # Get the gradient update indices in array form
  def gidxs(self):

    return np.array(self._gidxs, dtype='int32')

  #=====================================================================
  # Get mutability 
  def mutable(self):
    """"""

    return self.mutable

  #=====================================================================
  # Get word size
  def wsize(self):
    """"""

    return self._wsize

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

    return self.strs_to_idxs(self.start_str())

  #=====================================================================
  # Get stop index 
  def stop_idx(self):
    """"""

    return self.strs_to_idxs(self.stop_str())

  #=====================================================================
  # Get unk index
  def unk_idx(self):
    """"""

    return self.strs_to_idxs[self.stop_str()]

  #=====================================================================
  # Get start vector
  def start_vec(self):
    """"""

    return self.L[self.start_idx()]

  #=====================================================================
  # Get stop vector
  def stop_vec(self):
    """"""

    return self.L[self.stop_idx()]

  #=====================================================================
  # Get unk vector
  def unk_vec(self):
    """"""

    return self.L[self.unk_idx()]

  #=====================================================================
  # Convert strs to idxs
  def strs_to_idxs(self, strs):
    """"""

    if not hasattr(strs, '__iter__'):
      strs = [strs]
    return np.array([[self.idxs[s] for s in strs]], dtype='int32')

  #=====================================================================
  # Convert idxs to strs 
  def idxs_to_strs(self, idxs):
    """"""

    if not hasattr(idxs, '__iter__'):
      idxs = [idxs]
    return [self.strs[i] for i in idxs]

  #=====================================================================
  # Convert strs to vectors 
  def strs_to_vecs(self, strs):
    """"""

    return self.idxs_to_vecs(np.array(self.strs_to_idxs(strs)))

  #=====================================================================
  # Convert vectors to strs
  def vec_to_str(self, vectors):
    """"""

    return self.idxs_to_strs(self.vecs_to_idxs(np.array(vectors)))

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

    for x, gx in self.params, self.gparams:
      #-----------------------------------------------------------------
      # Initialize stored vectors
      vx = theano.shared(np.zeros_like(x.get_value()), name='v%s' % x.name)
      muvx = mu*vx

      #-----------------------------------------------------------------
      # Build up the updates
      updates.append((x, T.switch(not_finite, .1*x, x - eta*gx)))
      updates.append((vx, T.switch(not_finite, np.float32(0), muvx - eta*gx)))
      givens.append((x, x+muvx))

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

  #=====================================================================
  # Initialize the network
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
      l_0.append(lib.start_idx())
      l_max.append(lib.stop_idx())
    ldims = np.sum(ldims)*self.window
    l_0 = T.concatenate(l_0, axis=1)
    l_max = T.concatenate(l_max, axis=1)
    dims.insert(0, ldims)

    #-------------------------------------------------------------------
    # Initialize the model params
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
    self.c_0   = []

    for i in xrange(1, len(dims)):
      W = matwizard(dims[i-1], dims[i]*gates)
      U = matwizard(dims[i-1], dims[i], shape='diag')
      if gates > 1:
        U = np.concatenate([U, matwizard(dims[i-1], dims[i]*(gates-1))], axis=1)
      self.Wparams.append(theano.shared(np.concatenate([W, U]), name='W-%d' % (i+1)))
      self.bparams.append(theano.shared(np.zeros(dims[i]*gates, dtype='float32'), name='b-%d' % (i+1)))
      self.hparams.append(theano.shared(np.zeros(dims[i], dtype='float32'), name='h-%d' % (i+1)))
      self.hmasks.append(theano.shared(np.ones(dims[i]*gates, dtype='float32'), name='hmask-%d' % (i+1)))
      self.h_0.append(theano.shared(np.zeros(dims[i], dtype='float32'), name='h_0-%d' % (i+1)))
      self.c_0.append(theano.shared(np.zeros(dims[i], dtype='float32'), name='c_0-%d' % (i+1)))

    #-----------------------------------------------------------------
    # Build the input/output variables
    self.idxs = T.imatrix('idxs')
    self.xparams = []
    for i, lib in enumerate(self.libs):
      self.xparams.append(lib.get_subtensor(self.idxs[:,i]))
    x = T.concatenate(self.xparams, axis=1)
    self.y = T.fvector('y')

    #-------------------------------------------------------------------
    # Bundle the params
    self.params =\
        self.Wparams +\
        self.bparams +\
        self.hparams +\
        self.h_0 +\
        (self.C_0 if self.model.endswith('LSTM') else [])
    self.gparams = [theano.shared(np.zeros_like(param.get_value(), dtype='float32'), name='g'+param.name) for param in self.params]

    #-----------------------------------------------------------------
    # Build the activation variable
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if self.model == 'RNN':
      def recur(i, *hc_tm1):
        """"""

        h_tm1 = hc_tm1[:len(hc_tm1)/2]
        c_tm1 = hc_tm1[len(hc_tm1)/2:]
        h_t   = [x[i:i+self.window].flatten()]
        c_t   = []
        for h_tm1_l, Wparam, bparam, hparam, hmask in zip(h_tm1, self.Wparams, self.bparams, self.hparams, self.hmasks):
          xparam = T.concatenate([h_t[-1], h_tm1_l])
          s = sig(hparam)

          a = T.dot(Wparam, xparam) + bparam

          c = func(a, s)
          h = c*hmask

          c_t.append(c)
          h_t.append(h)

        return h_t[1:] + c_t

    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif self.model == 'GRU':
      def recur(i, *hc_tm1):
        """"""

        h_tm1 = hc_tm1[:len(hc_tm1)/2]
        c_tm1 = hc_tm1[len(hc_tm1)/2:]
        h_t   = [x[i:i+self.window].flatten()]
        c_t   = []
        for h_tm1_l, Wparam, bparam, hparam, hmask in zip(h_tm1, self.Wparams, self.bparams, self.hparams, self.hmasks):
          sliceLen = Wparam.shape[1]/3
          xparam = T.concatenate([h_t[-1], h_tm1_l])
          s = sig(hparam)

          zr = T.dot(Wparam[:,sliceLen:], xparam) + bparam[sliceLen:]
          z  = sig(zr[:sliceLen])/np.float32(4)
          r  = sig(zr[sliceLen:])
          a  = T.dot(Wparam[:,:sliceLen], T.concatenate([xparam[:len(h_t[-1])], xparam[len(h_t[-1]):]*r]))

          c = z*func(a, s) + (np.float32(1)-z)*h_tm1_l
          h = c*hmask

          c_t.append(c)
          h_t.append(h)

        return h_t[1:] + c_t

    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif self.model == 'FastGRU':
      def recur(i, *hc_tm1):
        """"""

        h_tm1 = hc_tm1[:len(hc_tm1)/2]
        c_tm1 = hc_tm1[len(hc_tm1)/2:]
        h_t   = [x[i:i+self.window].flatten()]
        c_t   = []
        for h_tm1_l, Wparam, bparam, hparam, hmask in zip(h_tm1, self.Wparams, self.bparams, self.hparams, self.hmasks):
          sliceLen = Wparam.shape[1]/3
          xparam = T.concatenate([h_t[-1], h_tm1_l])
          s = sig(hparam)

          azr = T.dot(Wparam, xparam) + bparam
          a   = func(azr[:sliceLen], s)
          z   = sig(zr[sliceLen:2*sliceLen])/4
          r   = sig(zr[2*sliceLen:])

          c = z*a + (np.float32(1)-z)*r*h_tm1_l
          h = c*hmask

          c_t.append(c)
          h_t.append(h)

        return h_t[1:] + c_t

    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif self.model == 'LSTM':
      def recur(i, *hc_tm1):
        """"""

        h_tm1 = hc_tm1[:len(hc_tm1)/2]
        c_tm1 = hc_tm1[len(hc_tm1)/2:]
        h_t   = [x[i:i+self.window].flatten()]
        c_t   = []
        for h_tm1_l, c_tm1_l, Wparam, bparam, hparam, hmask in zip(h_tm1, c_tm1, self.Wparams, self.bparams, self.hparams, self.hmasks):
          sliceLen = Wparam.shape[1]/4
          xparam = T.concatenate([h_t[-1], h_tm1_l])
          s = sig(hparam)

          aifo = T.dot(Wparam, xparam) + bparam
          a    = func(aifo[:sliceLen], s)
          i    = sig(aifo[sliceLen:2*sliceLen])
          f    = sig(aifo[2*sliceLen:3*sliceLen])
          o    = sig(aifo[3*sliceLen:])

          c = i*a + f*c_tm1_l
          h = func(c*o, s)*hmask

          c_t.append(c)
          h_t.append(h)

        return h_t[1:] + c_t

    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif self.model == 'FastLSTM':
      def recur(i, *hc_tm1):
        """"""

        h_tm1 = hc_tm1[:len(hc_tm1)/2]
        c_tm1 = hc_tm1[len(hc_tm1)/2:]
        h_t   = [x[i:i+self.window].flatten()]
        c_t   = []
        for h_tm1_l, c_tm1_l, Wparam, bparam, hparam, hmask in zip(h_tm1, c_tm1, self.Wparams, self.bparams, self.hparams, self.hmasks):
          sliceLen = Wparam.shape[1]/3
          xparam = T.concatenate([h_t[-1], h_tm1_l])
          s = sig(hparam)

          azo = T.dot(Wparam, xparam) + bparam
          a   = func(aifo[:sliceLen], s)
          z   = sig(aifo[sliceLen:2*sliceLen])/np.float32(4)
          o   = sig(aifo[2*sliceLen:])

          c = z*a + (1-z)*c_tm1_l
          h = func(c*o, s)*hmask

          c_t.append(c)
          h_t.append(h)

        return h_t[1:] + c_t

    self.h, _ = theano.scan(
        fn=recur,
        sequences=T.arange(x.shape[0]-(self.window-1)),
        outputs_info = self.h_0 + self.c_0)

    yhat = self.h[-1][-1]

    #-------------------------------------------------------------------
    # Build the cost variable
    # TODO integrate this with the library class
    self.error = T.mean(squared_difference(yhat, self.y))

    self.complexity = theano.shared(np.float32(0)) +\
        (np.float32(self.L1reg)*T.sum([T.sum(T.abs_(Wparam)) for Wparam in self.Wparams]) if self.L1reg > 0 else np.float32(0)) +\
        (np.float32(self.L2reg)*T.sum([T.sum(T.sqr(Wparam)) for Wparam in self.Wparams]) if self.L2reg > 0 else np.float32(0))

    self.cost = self.error + self.complexity 

    #===================================================================
    # Activate
    self.idxs_to_vec = theano.function(
        inputs=[self.idxs],
        outputs=yhat,
        allow_input_downcast=True)

    #===================================================================
    # Error
    self.idxs_to_err = theano.function(
        inputs=[self.idxs, self.y],
        outputs=self.error,
        allow_input_downcast=True)

    #===================================================================
    # Complexity
    self.comp = theano.function(
        inputs=[],
        outputs=self.complexity,
        allow_input_downcast=True)

    #===================================================================
    # Cost
    self.idxs_to_cost = theano.function(
        inputs=[self.idxs, self.y],
        outputs=self.cost,
        allow_input_downcast=True)

    #===================================================================
    # Gradients
    self.idxs_to_grads = theano.function(
        inputs=[self.idxs, self.y],
        outputs=[self.cost] + T.grad(self.cost, self.params),
        allow_input_downcast=True)

    #===================================================================
    # Update gradients
    batchSize = T.scalar('batchSize')
    paramVars =\
        [T.matrix() for Wparam in self.Wparams] +\
        [T.vector() for bparam in self.bparams] +\
        [T.vector() for hparam in self.hparams] +\
        [T.vector() for h_0_l in self.h_0] +\
        ([T.vector() for c_0_l in self.c_0] if self.model.endswith('LSTM') else [])
    self.update_grads = theano.function(
        inputs=[batchSize]+paramVars,
        outputs=[],
        updates=[(gparam, gparam+paramVar/batchSize) for gparam, paramVar in zip(self.gparams, paramVars)],
        allow_input_downcast=True)

    #===================================================================
    # Reset gradients
    self.reset_grad = theano.function(
        inputs=[],
        outputs=[],
        updates=[(gparam, 0*gparam) for gparam in self.gparams],
        allow_input_downcast=True)

  #=====================================================================
  # Converts a list of strings or string tuples into a matrix
  def strs_to_idxs(self, strings):
    """"""

    if not isinstance(strings[0], (tuple, list)):
      return self.libs[0].strs_to_idxs(strings)
    else:
      return np.concatenate([lib.strs_to_idxs([string[i] for string in strings]) for i, lib in enumerate(self.libs)], axis=1)

  #=====================================================================
  # Pad a list of strings or string tuples
  def pad_strs(self, strings):
    """"""

    if self.reverse:
      begins = tuple([lib.stop_str()  for lib in self.libs])
      ends   = tuple([lib.start_str() for lib in self.libs])
    else:
      begins = tuple([lib.start_str() for lib in self.libs])
      ends   = tuple([lib.stop_str()  for lib in self.libs])

    # Pad the beginning
    nbegins = 0
    while tuple(strings[nbegins]) == begins:
      nbegins += 1
    strings = [begins]*(self.window-nbegins) + strings

    # Pad the end
    if tuple(strings[0]) != ends:
      strings.insert(0, ends)

    return strings

  #=====================================================================
  # Pad a matrix of indices
  def pad_idxs(self, indices):
    """"""

    if self.reverse:
      begins = np.concatenate([lib.stop_idx()  for lib in self.libs])
      ends   = np.concatenate([lib.start_idx() for lib in self.libs])
    else:
      begins = np.concatenate([lib.start_idx() for lib in self.libs])
      ends   = np.concatenate([lib.stop_idx()  for lib in self.libs])

    # Calculate the beginning padding
    nbegins = 0
    while np.equal(indices[nbegins], begins):
      nbegins += 1

    # Calculate the ending padding
    nends = 0
    if np.equal(indices[-1], ends):
      nends += 1
    a = np.empty(len(indices)+nbegins+end)
    a[0:(self.window-nbegins)] = begins
    a[-1-(1-nends):-1] = ends
    
    return a

  #=====================================================================
  # Pad a matrix of vectors 
  def pad_vecs(self, vectors):
    """"""

    if self.reverse:
      begins = np.concatenate([lib.stop_vec()  for lib in self.libs])
      ends   = np.concatenate([lib.start_vec() for lib in self.libs])
    else:
      begins = np.concatenate([lib.start_vec() for lib in self.libs])
      ends   = np.concatenate([lib.stop_vec()  for lib in self.libs])

    # Calculate the beginning padding
    nbegins = 0
    while np.equal(vectors[nbegins], begins):
      nbegins += 1

    # Calculate the ending padding
    nends = 0
    if np.equal(vectors[-1], ends):
      nends += 1
    a = np.empty(len(vectors)+nbegins+end)
    a[0:(self.window-nbegins)] = begins
    a[-1-(1-nends):-1] = ends
    
    return a

  #=====================================================================
  # Unpad a list of strings or string tuples
  def unpad_strs(self, strings):
    """"""

    if self.reverse:
      begins = tuple([lib.stop_str()  for lib in self.libs])
      ends   = tuple([lib.start_str() for lib in self.libs])
    else:
      begins = tuple([lib.start_str() for lib in self.libs])
      ends   = tuple([lib.stop_str()  for lib in self.libs])

    # Unpad the beginning
    nbegins = 0
    while tuple(strings[nbegins]) == begins:
      nbegins += 1
    strings = strings[nbegins:]

    # Unpad the end
    if tuple(strings[-1]) == ends:
      strings.pop()

    return strings

  #=====================================================================
  # Unpad a vector of indices 
  def unpad_idxs(self, indices):
    """"""

    if self.reverse:
      begins = np.concatenate([lib.stop_idx()  for lib in self.libs])
      ends   = np.concatenate([lib.start_idx() for lib in self.libs])
    else:
      begins = np.concatenate([lib.start_idx() for lib in self.libs])
      ends   = np.concatenate([lib.stop_idx()  for lib in self.libs])

    # Unpad the beginning
    nbegins = 0
    while tuple(indices[nbegins]) == begins:
      nbegins += 1
    indices = indices[nbegins:]

    # Unpad the end
    if tuple(indices[-1]) == ends:
      indices = indices[:-1]

    return indices

  #=====================================================================
  # Unpad a vector of vectors
  def unpad_vecs(self, vectors):
    """"""

    if self.reverse:
      begins = np.concatenate([lib.stop_vec()  for lib in self.libs])
      ends   = np.concatenate([lib.start_vec() for lib in self.libs])
    else:
      begins = np.concatenate([lib.start_vec() for lib in self.libs])
      ends   = np.concatenate([lib.stop_vec()  for lib in self.libs])

    # Unpad the beginning
    nbegins = 0
    while tuple(vectors[nbegins]) == begins:
      nbegins += 1
    vectors = vectors[nbegins:]

    # Unpad the end
    if tuple(vectors[-1]) == ends:
      vectors = vectors[:-1]

    return vectors

  #=====================================================================
  # Converts a dataset into the expected format (in place)
  def convert_dataset(self, dataset):
    """"""

    for i, datum in enumerate(dataset):
      if self.reverse:
        string = datum[0][::-1]
      else:
        string = datum[0][:]
      dataset[i] = (self.strs_to_idxs(self.pad_strs(string)), datum[1])

  #=====================================================================
  # Converts a list of input strings or string tuples into a vector 
  def strs_to_vec(self, strings):
    """"""

    if self.reverse:
      strings = strings[::-1]
    else:
      strings = strings[:]

    return self.idxs_to_vec(self.strs_to_idxs(self.pad_strs(strings)))

  #=====================================================================
  # Converts a list of input strings or string tuples and an output vector into an error
  def strs_to_err(self, strings, vector):
    """"""

    if self.reverse:
      strings = strings[::-1]
    else:
      strings = strings[:]

    return self.idxs_to_err(self.strs_to_idxs(self.pad_strs(strings)), vector)

  #=====================================================================
  # Converts a list of input strings or string tuples and an output vector into a cost
  def strs_to_cost(self, strings, vector):
    """"""

    if self.reverse:
      strings = strings[::-1]
    else:
      strings = strings[:]

    return self.idxs_to_cost(self.strs_to_idxs(self.pad_strs(strings)), vector)

  #=====================================================================
  # Calculate the cost of a minibatch using multiple cores
  def batch_cost(self, dataset, workers=2):
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
      process = mp.Process(target=func_worker, args=(dataQueue, costQueue, self.idxs_to_cost))
      process.start()
      processes.append(process)
    for worker in xrange(workers):
      for subCost in iter(costQueue.get, 'STOP'):
        cost += subCost
    for process in processes:
      process.join()
    return cost / len(dataset)

  #=====================================================================
  # Calculate the gradients of a minibatch using multiple cores
  def train(self, dataset, optimizer, batchSize=64, epochs=1, costEvery=None, testset=None, saveEvery=None, savePipe=None, workers=2):
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
          process = mp.Process(target=func_worker, args=(dataQueue, gradQueue, self.idxs_to_grads))
          process.start()
          processes.append(process)
        for worker in xrange(workers):
          for grads in iter(gradQueue.get, 'STOP'):
            recentCost.append(grads[0])
            self.update_grads(batchSize, *grads[1:])
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
if __name__ == '__main__':
  """"""

  glove = pkl.load(open('glove.6B.10k.50d.pkl'))
  dataset = zip([list(x) for x in sorted(glove[1], key=glove[1].get)], glove[0])
  vocab = set()
  for datum in dataset:
    vocab.update(datum[0])
  lib = Library(keys=vocab, mat=50)
  encoder = Encoder([lib], [100,100,50])
  encoder.convert_dataset(dataset)
  print 'Works!'

