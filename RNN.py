#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matwizard import matwizard
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams()
from funx import squared_difference, absolute_difference, cosine_similarity, sigmoid, softmax, funx
import cPickle as pkl
import multiprocessing as mp
import sys
sys.setrecursionlimit(50000)
import time
from collections import defaultdict, Counter
from nltk import word_tokenize
import nltk.data
sent_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
import codecs
#theano.config.optimizer='None'

#***********************************************************************
# Helper functions

#=======================================================================
# Pickling worker function
def pkl_worker(childPipe, path='.', name=''):
  """"""
  
  i = 0
  msg = 'START'
  while msg != 'STOP':
    msg = childPipe.recv()
    if msg not in ('START', 'STOP'):
      i += 1
      pkl.dump(msg[0], open(os.path.join(path, '%sstate-%02d.pkl' % (name, i)), 'w'), protocol=pkl.HIGHEST_PROTOCOL)
      pkl.dump(msg[1:], open(os.path.join(path, '%scost-%02d.pkl' % (name, i)), 'w'))
  return True

#***********************************************************************
# Anything that can be optimized
# TODO put the cost/training functions here
class Opt():
  """"""
  
  #=====================================================================
  # Run SGD with NAG
  def SGD(self, eta_0=.01, T_eta=1, mu_max=.95, T_mu=1, dropout=1., anneal=0, accel=0):
    """"""
    
    #-------------------------------------------------------------------
    # Cast everything as float32
    eta_0  = np.float32(eta_0)
    T_eta  = np.float32(T_eta)
    mu_max = np.float32(mu_max)
    T_mu   = np.float32(T_mu)
    anneal = np.float32(anneal)
    accel  = np.float32(accel)
    
    #-------------------------------------------------------------------
    # Set up the updates (see RNN3 for solution if we get non-numeric gradients)
    mupdates  = []
    grupdates = []
    pupdates  = []
    nupdates  = []
    
    #-------------------------------------------------------------------
    # Set up a variable to keep track of the iteration
    tau = theano.shared(np.float32(0), name='tau')
    pupdates.extend([(tau, tau+np.float32(1))])
    
    #-------------------------------------------------------------------
    # Set the annealing/acceleration schedule
    eta = eta_0*T.pow(T_eta/(tau+T_eta), anneal)
    mu  = mu_max*(np.float32(1)-T.pow(T_mu/(tau+T_mu), accel))
    
    #-------------------------------------------------------------------
    # Compute the dropout and gradients
    grads = T.grad(self.cost, self.params+self.xparams)
    givens = []
    if dropout < 1:
      for hmask in self.hmasks:
        givens.append((hmask, srng.binomial(hmask.shape, 1, dropout, dtype='float32')))
    
    #-------------------------------------------------------------------
    # Dense parameters
    for theta, gtheta_i, gtheta in zip(self.params, grads[:len(self.params)], self.gparams):
      vtheta = theano.shared(np.zeros_like(theta.get_value()), name='v%s' % theta.name)
      
      mupdates.append((theta, theta + mu*vtheta))
      grupdates.append((gtheta, gtheta + gtheta_i))
      pupdates.append((theta, theta - eta*gtheta))
      pupdates.append((vtheta, mu*vtheta - eta*gtheta))
      nupdates.append((gtheta, gtheta * np.float32(0)))
    
    #-------------------------------------------------------------------
    # Sparse parameters
    gidxs = []
    for lidx, L, gL, gtheta_i in zip(range(len(self.sparams)), self.sparams, self.gsparams, grads[len(self.params):]):
      vL = theano.shared(np.zeros_like(L.get_value()), name='v%s' % L.name)
      
      gidxs.append(T.ivector('gidxs-%s' % L.name))
      mupdates.append((L, T.inc_subtensor(L[gidxs[-1]], mu*vL[gidxs[-1]])))
      grupdates.append((gL, T.inc_subtensor(gL[self.x[:,lidx]], gtheta_i)))
      pupdates.append((L, T.inc_subtensor(L[gidxs[-1]], -eta*gL[gidxs[-1]])))
      pupdates.append((vL, T.set_subtensor(vL[gidxs[-1]], mu*vL[gidxs[-1]] - eta*gL[gidxs[-1]])))
      nupdates.append((gL, T.set_subtensor(gL[gidxs[-1]], np.float32(0))))
    
    #-------------------------------------------------------------------
    # Compile the functions
    momentizer = theano.function(
      inputs=gidxs,
      updates=mupdates)
    
    gradientizer = theano.function(
      inputs=[self.x, self.y],
      outputs=self.cost,
      givens=givens,
      updates=grupdates)
    
    optimizer = theano.function(
      inputs=gidxs,
      updates=pupdates)
    
    nihilizer = theano.function(
      inputs=gidxs,
      updates=nupdates)
      
    return momentizer, gradientizer, optimizer, nihilizer
  
  #=====================================================================
  # Run RMSProp with NAG
  def RMSProp(self, eta_0=.01, T_eta=1, rho_max=.9, T_rho=1, mu_max=.95, T_mu=1, epsilon=1e-6, dropout=1., anneal=0, expand=0, accel=0):
    """"""
    
    #-------------------------------------------------------------------
    # Cast everything as float32
    eta_0   = np.float32(eta_0)
    T_eta   = np.float32(T_eta)
    rho_max   = np.float32(rho_max)
    T_rho   = np.float32(T_rho)
    mu_max  = np.float32(mu_max)
    T_mu    = np.float32(T_mu)
    epsilon = np.float32(epsilon)
    anneal  = np.float32(anneal)
    accel   = np.float32(accel)
    
    #-------------------------------------------------------------------
    # Set up the updates (see RNN3 for solution if we get non-numeric gradients)
    mupdates  = []
    grupdates = []
    pupdates  = []
    nupdates  = []
    
    #-------------------------------------------------------------------
    # Set up a variable to keep track of the iteration
    tau = theano.shared(np.float32(0), name='tau')
    pupdates.extend([(tau, tau+np.float32(1))])
    
    #-------------------------------------------------------------------
    # Set the annealing/acceleration schedule
    eta = eta_0*T.pow(T_eta/(tau+T_eta), anneal)
    rho = rho_max*(np.float32(1)-T.pow(T_rho/(tau+T_rho), expand))
    mu  = mu_max*(np.float32(1)-T.pow(T_mu/(tau+T_mu), accel))
    
    #-------------------------------------------------------------------
    # Compute the dropout and gradients
    grads = T.grad(self.cost, self.params+self.xparams)
    givens = []
    if dropout < 1:
      for hmask in self.hmasks:
        givens.append((hmask, srng.binomial(hmask.shape, 1, dropout, dtype='float32')))
    
    #-------------------------------------------------------------------
    # Dense parameters
    for theta, gtheta_i, gtheta in zip(self.params, grads[:len(self.params)], self.gparams):
      vtheta = theano.shared(np.zeros_like(theta.get_value()), name='v%s' % theta.name)
      g2theta = theano.shared(np.zeros_like(theta.get_value()), name='g2%s' % theta.name)
      
      g2theta_t = rho*g2theta + (np.float32(1)-rho)*T.sqr(gtheta)
      deltatheta_t = gtheta/T.sqrt(g2theta_t+epsilon)
      
      mupdates.append((theta, theta + mu*vtheta))
      grupdates.append((gtheta, gtheta + gtheta_i))
      pupdates.append((theta, theta - eta*deltatheta_t))
      pupdates.append((g2theta, g2theta_t))
      pupdates.append((vtheta, mu*vtheta - eta*deltatheta_t))
      nupdates.append((gtheta, gtheta * np.float32(0)))
      
    #-------------------------------------------------------------------
    # Sparse parameters
    gidxs = []
    for lidx, L, gL, gtheta_i in zip(range(len(self.sparams)), self.sparams, self.gsparams, grads[len(self.params):]):
      vL = theano.shared(np.zeros_like(L.get_value()), name='v%s' % L.name)
      g2L = theano.shared(np.zeros_like(L.get_value()), name='v%s' % L.name)
      
      gidxs.append(T.ivector('gidxs-%s' % L.name))
      
      g2L_t = rho*g2L[gidxs[-1]] + (np.float32(1)-rho)*T.sqr(gL[gidxs[-1]])
      deltaL_t = gL[gidxs[-1]]/T.sqrt(g2L_t+epsilon)
      
      mupdates.append((L, T.inc_subtensor(L[gidxs[-1]], mu*vL[gidxs[-1]])))
      grupdates.append((gL, T.inc_subtensor(gL[self.x[:,lidx]], gtheta_i)))
      pupdates.append((L, T.inc_subtensor(L[gidxs[-1]], -eta*deltaL_t)))
      pupdates.append((g2L, T.set_subtensor(g2L[gidxs[-1]], g2L_t)))
      pupdates.append((vL, T.set_subtensor(vL[gidxs[-1]], mu*vL[gidxs[-1]] - eta*gL[gidxs[-1]])))
      nupdates.append((gL, T.set_subtensor(gL[gidxs[-1]], np.float32(0))))
    
    #-------------------------------------------------------------------
    # Compile the functions
    momentizer = theano.function(
      inputs=gidxs,
      updates=mupdates)
    
    gradientizer = theano.function(
      inputs=[self.x, self.y],
      outputs=self.cost,
      givens=givens,
      updates=grupdates)
    
    optimizer = theano.function(
      inputs=gidxs,
      updates=pupdates)
    
    nihilizer = theano.function(
      inputs=gidxs,
      updates=nupdates)
      
    return momentizer, gradientizer, optimizer, nihilizer
  
  #=====================================================================
  # Run AdaDelta
  def AdaDelta(self, eta_0=1., T_eta=1, rho_max=.9, T_rho=1, epsilon=1e-6, dropout=1., anneal=0, expand=0):
    """"""
    
    #-------------------------------------------------------------------
    # Cast everything as float32
    eta_0  = np.float32(eta_0)
    T_eta  = np.float32(T_eta)
    rho_max  = np.float32(rho_max)
    T_rho  = np.float32(T_rho)
    epsilon = np.float32(epsilon)
    anneal = np.float32(anneal)
    expand  = np.float32(expand)
    
    #-------------------------------------------------------------------
    # Set up the updates (see RNN3 for solution if we get non-numeric gradients)
    mupdates  = []
    grupdates = []
    pupdates  = []
    nupdates  = []
    
    #-------------------------------------------------------------------
    # Set up a variable to keep track of the iteration
    tau = theano.shared(np.float32(0), name='tau')
    pupdates.extend([(tau, tau+np.float32(1))])
    
    #-------------------------------------------------------------------
    # Set the annealing/acceleration schedule
    eta = eta_0*T.pow(T_eta/(tau+T_eta), anneal)
    rho = rho_max*(np.float32(1)-T.pow(T_rho/(tau+T_rho), expand))
    
    #-------------------------------------------------------------------
    # Compute the dropout and gradients
    grads = T.grad(self.cost, self.params+self.xparams)
    givens = []
    if dropout < 1:
      for hmask in self.hmasks:
        givens.append((hmask, srng.binomial(hmask.shape, 1, dropout, dtype='float32')))
    
    #-------------------------------------------------------------------
    # Dense parameters
    for theta, gtheta_i, gtheta in zip(self.params, grads[:len(self.params)], self.gparams):
      g2theta = theano.shared(np.zeros_like(theta.get_value()), name='g2%s' % theta.name)
      delta2theta = theano.shared(np.zeros_like(theta.get_value()), name='delta2%s' % theta.name)
      
      g2theta_t = rho*g2theta + (np.float32(1)-rho)*T.sqr(gtheta)
      deltatheta_t = T.sqrt(delta2theta+epsilon)/T.sqrt(g2theta_t+epsilon) * gtheta
      delta2theta_t = rho*delta2theta + (np.float32(1)-rho)*T.sqr(deltatheta_t)
      
      grupdates.append((gtheta, gtheta + gtheta_i))
      pupdates.append((theta, theta - eta*deltatheta_t))
      pupdates.append((g2theta, g2theta_t))
      pupdates.append((delta2theta, delta2theta_t))
      nupdates.append((gtheta, gtheta * np.float32(0)))
    
    #-------------------------------------------------------------------
    # Sparse parameters
    gidxs = []
    for lidx, L, gL, gtheta_i in zip(range(len(self.sparams)), self.sparams, self.gsparams, grads[len(self.params):]):
      g2L = theano.shared(np.zeros_like(L.get_value()), name='g2%s' % L.name)
      delta2L = theano.shared(np.zeros_like(L.get_value()), name='delta2%s' % L.name)
      
      gidxs.append(T.ivector('gidxs-%s' % L.name))
      
      g2L_t = rho*g2L[gidxs[-1]] + (np.float32(1)-rho)*T.sqr(gL[gidxs[-1]])
      deltaL_t = T.sqrt(delta2L[gidxs[-1]]+epsilon)/T.sqrt(g2L_t+epsilon) * gL[gidxs[-1]]
      delta2L_t = rho*delta2L[gidxs[-1]] + (np.float32(1)-rho)*T.sqr(deltaL_t)
      
      grupdates.append((gL, T.inc_subtensor(gL[self.x[:,lidx]], gtheta_i)))
      pupdates.append((L, T.inc_subtensor(L[gidxs[-1]], -eta*deltaL_t)))
      pupdates.append((g2L, T.set_subtensor(g2L[gidxs[-1]], g2L_t)))
      pupdates.append((delta2L, T.set_subtensor(delta2L[gidxs[-1]], delta2L_t)))
      nupdates.append((gL, T.set_subtensor(gL[gidxs[-1]], np.float32(0))))
     
    #-------------------------------------------------------------------
    # Compile the functions
    momentizer = theano.function(
      inputs=gidxs,
      updates=mupdates,
      on_unused_input='ignore')
    
    gradientizer = theano.function(
      inputs=[self.x, self.y],
      outputs=self.cost,
      givens=givens,
      updates=grupdates)
    
    optimizer = theano.function(
      inputs=gidxs,
      updates=pupdates)
    
    nihilizer = theano.function(
      inputs=gidxs,
      updates=nupdates)
      
    return momentizer, gradientizer, optimizer, nihilizer
  
  #=====================================================================
  # Run Adam
  def Adam(self, eta_0=.05, T_eta=1, rho1_max=.9, rho2_max=.99, T_rho=1, epsilon=1e-6, dropout=1., anneal=0, expand=0):
    """"""

    #-------------------------------------------------------------------
    # Cast everything as float32
    eta_0  = np.float32(eta_0)
    T_eta  = np.float32(T_eta)
    rho1_max = np.float32(rho1_max)
    rho2_max = np.float32(rho2_max)
    T_rho   = np.float32(T_rho)
    anneal = np.float32(anneal)
    expand  = np.float32(expand)
    
    #-------------------------------------------------------------------
    # Set up the updates (see RNN3 for solution if we get non-numeric gradients)
    mupdates  = []
    grupdates = []
    pupdates  = []
    nupdates  = []
    
    #-------------------------------------------------------------------
    # Set up a variable to keep track of the iteration
    tau = theano.shared(np.float32(0), name='tau')
    pupdates.extend([(tau, tau+np.float32(1))])
    
    #-------------------------------------------------------------------
    # Set the annealing schedule
    eta = eta_0*T.pow(T_eta/(tau+T_eta), anneal)
    rho1 = rho1_max*(np.float32(1)-T.pow(T_rho/(tau+T_rho), expand))
    rho2 = rho2_max*(np.float32(1)-T.pow(T_rho/(tau+T_rho), expand))
    
    #-------------------------------------------------------------------
    # Compute the dropout and gradients
    grads = T.grad(self.cost, self.params+self.xparams)
    givens = []
    if dropout < 1:
      for hmask in self.hmasks:
        givens.append((hmask, srng.binomial(hmask.shape, 1, dropout, dtype='float32')))
    
    #-------------------------------------------------------------------
    # Dense parameters
    for theta, gtheta_i, gtheta in zip(self.params, grads[:len(self.params)], self.gparams):
      mtheta = theano.shared(np.zeros_like(theta.get_value()), name='m%s' % theta.name)
      vtheta = theano.shared(np.zeros_like(theta.get_value()), name='v%s' % theta.name)
      
      mtheta_t = (rho1*mtheta + (np.float32(1)-rho1)*gtheta) / (np.float32(1)-rho1)
      vtheta_t = (rho2*vtheta + (np.float32(1)-rho2)*T.sqr(gtheta)) / (np.float32(1)-rho2)
      deltatheta_t = mtheta_t / (T.sqrt(vtheta_t) + epsilon)
      
      grupdates.append((gtheta, gtheta + gtheta_i))
      pupdates.append((theta, theta - eta*deltatheta_t))
      pupdates.append((mtheta, mtheta_t))
      pupdates.append((vtheta, vtheta_t))
      nupdates.append((gtheta, gtheta * np.float32(0)))
    
    #-------------------------------------------------------------------
    # Sparse parameters
    gidxs = []
    for lidx, L, gL, gtheta_i in zip(range(len(self.sparams)), self.sparams, self.gsparams, grads[len(self.params):]):
      mL = theano.shared(np.zeros_like(L.get_value()), name='m%s' % L.name)
      vL = theano.shared(np.zeros_like(L.get_value()), name='v%s' % L.name)
      
      gidxs.append(T.ivector('gidxs-%s' % L.name))
      
      mL_t = (rho1*mtheta[gidxs[-1]] + (np.float32(1)-rho1)*gL[gidxs[-1]]) / (np.float32(1)-rho1)
      vL_t = (rho2*vtheta[gidxs[-1]] + (np.float32(1)-rho2)*T.sqr(gL[gidxs[-1]])) / (np.float32(1)-rho2)
      deltaL_t = mL_t / (T.sqrt(vL_t) + epsilon)
      
      grupdates.append((gL, T.inc_subtensor(gL[self.x[:,lidx]], gtheta_i)))
      pupdates.append((L, T.inc_subtensor(L[gidxs[-1]], -eta*deltaL_t)))
      pupdates.append((mL, T.set_subtensor(mL[gidxs[-1]], mL_t)))
      pupdates.append((vL, T.set_subtensor(vL[gidxs[-1]], vL_t)))
      nupdates.append((gL, T.set_subtensor(gL[gidxs[-1]], np.float32(0))))
      
    #-------------------------------------------------------------------
    # Compile the functions
    momentizer = theano.function(
      inputs=gidxs,
      updates=mupdates,
      on_unused_input='ignore')
    
    gradientizer = theano.function(
      inputs=[self.x, self.y],
      outputs=self.cost,
      givens=givens,
      updates=grupdates)
    
    optimizer = theano.function(
      inputs=gidxs,
      updates=pupdates)
    
    nihilizer = theano.function(
      inputs=gidxs,
      updates=nupdates)
      
    return momentizer, gradientizer, optimizer, nihilizer
  
  #=====================================================================
  # Calculate the gradients of a minibatch using multiple cores
  def train(self, dataset, momentizer, gradientizer, optimizer, nihilizer, batchSize=64, epochs=1, costEvery=None, testset=None, saveEvery=None, saveName='None'):
    """"""
    
    #-------------------------------------------------------------------
    # Saving and printing
    s = ''
    epochd = str(int(np.log10(epochs))+1)
    minibatchd = str(int(max(0,np.log10(len(dataset[0])/batchSize)))+1)
    s += ('Minibatch %0'+epochd+'d-%0'+minibatchd+'d') %(0,0)
    cost = []
    cost.append(self.batch_cost(dataset))
    s += ': %.3f train error' % cost[-1]
    test = None
    nsaves = 1
    if testset is not None:
      test = []
      test.append(self.batch_cost(testset))
      s += ', %.3f test error' % test[-1]
    wps = 0.0
    s += ', %.1f data per second' % wps
    if saveEvery is not None:
      self.save(saveName+'%02d'%nsaves, cost, test)
      nsaves += 1
      lastSaveTime = time.time()
      s += ', %.1f minutes since saving' % ((time.time()-lastSaveTime)/60)
    s += '        \r'
    print s,
    sys.stdout.flush()
    lastCostTime = time.time()
    
    #-------------------------------------------------------------------
    # Multiprocessing the minibatch
    recentCost = []
    mb=-1
    for t in xrange(epochs):
      dataidxs = np.arange(len(dataset[0])).astype('int32')
      np.random.shuffle(dataidxs)
      for mb in xrange(len(dataset[0])/batchSize):
        self.__train__(dataset, dataidxs[mb*batchSize:(mb+1)*batchSize], recentCost, momentizer, gradientizer, optimizer, nihilizer)
    
        #---------------------------------------------------------------
        # More printing and saving
        if costEvery is not None and (mb+1) % costEvery == 0:
          cost.append(np.mean(recentCost))
          recentCost = []
          if testset is not None:
            test.append(self.batch_cost(testset))
          thisCostTime = time.time()
        if saveEvery is not None and (mb+1) % saveEvery == 0:
          self.save(saveName+'%02d'%nsaves, cost, test)
          nsaves += 1
          lastSaveTime = time.time()
        if costEvery is not None and (mb+1) % costEvery == 0:
          s = ''
          s += ('Minibatch %0'+epochd+'d-%0'+minibatchd+'d') % (t+1, mb+1)
          s += ': %.3f train error' % cost[-1]
          if testset is not None:
            s += ': %.3f test error' % test[-1]
          if wps == 0:
            wps = ((batchSize * costEvery) / (thisCostTime-lastCostTime))
          else:
            wps = .67*wps + .33*((batchSize*costEvery) / (thisCostTime-lastCostTime))
          s += ', %.1f data per second' % wps
          if saveEvery is not None:
            s += ', %.1f minutes since saving' % (time.time()-lastSaveTime)
          s += '        \r'
          print s,
          sys.stdout.flush()
          lastCostTime = time.time()

      #-----------------------------------------------------------------
      # If we haven't been printing, print now
      if not (costEvery is not None and (mb+1) % costEvery == 0):
        cost.append(np.mean(recentCost))
        recentCost = []
        if testset is not None:
          test.append(self.batch_cost(testset))
        thisCostTime = time.time()
      if saveEvery is not None and (mb+1) % saveEvery != 0:
        self.save(saveName+'%02d'%nsaves, cost, test)
        nsaves += 1
        lastSaveTime = time.time()
      s = ''
      s += ('Minibatch %0'+epochd+'d-%0'+minibatchd+'d') % (t+1,mb+1)
      s += ': %.3f train error' % cost[-1]
      if testset is not None:
        s += ': %.3f test error' % test[-1]
      if wps == 0:
        wps = .67*wps + .33*((batchSize*((mb+1) % costEvery) if costEvery is not None else len(dataset[0])) / (thisCostTime-lastCostTime))
      s += ', %.1f data per second' % wps
      if saveEvery is not None:
        s += ', %.1f minutes since saving' % ((time.time() - lastSaveTime)/60)
      s += '        \r'
      print s
      sys.stdout.flush()
      if costEvery is None or (mb+1) % costEvery != 0:
        lastCostTime = time.time()
    
    #-------------------------------------------------------------------
    # Wrap everything up
    self.save(saveName+'%02d'%nsaves, cost, test)
    print ''
    return cost
  
  #=====================================================================
  # Class-specific training function
  def __train__(self):
    pass
  
  #=====================================================================
  # Pickle the model
  def save(self, filename, cost, test=None):
    """"""
    
    with open(filename+'-state.pkl', 'w') as f:
      pkl.dump(self.__dict__, f, protocol=pkl.HIGHEST_PROTOCOL)
    with open(filename+'-cost.pkl', 'w') as f:
      pkl.dump((cost,) + ((test,) if test is not None else tuple()), f, protocol=pkl.HIGHEST_PROTOCOL)
    
  #=====================================================================
  # Load the model
  @classmethod
  def load(cls, filename):
    """"""
    
    with open(filename+'state.pkl') as f:
      return pkl.load(f)
  
#***********************************************************************
# A library
# TODO build in GloVe capabilities
class Library():
  """"""
  
  #=====================================================================
  # Initialize the library
  def __init__(self, keys, mat, **kwargs):
    """"""
    
    #-------------------------------------------------------------------
    # Keyword Arguments
    if 'start' in kwargs:
      self._start = kwargs['start']
    else:
      self._start = '<S>'

    if 'stop' in kwargs:
      self._stop = kwargs['stop']
    else:
      self._stop = '</S>'

    if 'unk' in kwargs:
      self._unk = kwargs['unk']
    else:
      self._unk = '<UNK>'
    
    if 'normalize' in kwargs:
      normalize = kwargs['normalize']
    else:
      normalize = True
    
    #-------------------------------------------------------------------
    # Set up the access keys
    if isinstance(keys, set):
      keys.update((self._start, self._unk, self._stop))
      keys = list(keys)
    elif isinstance(keys, (tuple, list)):
      if isinstance(keys[0], basestring):
        keys = set(keys)
        keys.update((self._start, self._unk, self._stop))
        keys = list(keys)
      elif isinstance(keys[0], (tuple, list)):
        if isinstance(keys[0][0], (int, long)):
          self.strs = keys
          self.idxs = {v:k for k, v in keys.iteritems()}
        else:
          self.idxs = keys
          self.strs = {v:k for k, v in keys.iteritems()}
    elif isinstance(keys, dict):
      if 0 in keys:
        self.strs = keys
        self.idxs = {v:k for k, v in keys.iteritems()}
      else:
        self.idxs = keys
        self.strs = {v:k for k, v in keys.iteritems()}
        
    #-------------------------------------------------------------------
    # Set up the access keys
    if isinstance(mat, (int, long)):
      mat = np.random.randn(size=(len(keys), mat))
    else:
      assert len(mat) == len(keys)
      if normalize:
        mat = (mat-np.mean(mat))/np.std(mat)
    self._wsize = mat.shape[1]
    
    #-------------------------------------------------------------------
    # Set up the Theano variables
    self.hmask = theano.shared(np.ones(self._wsize, dtype='float32'))
    self.L = theano.shared(mat.astype('float32'))
    self.gL = theano.shared(np.zeros_like(mat, dtype='float32'))
    
    #===================================================================
    # Convert idxs to vectors
    x = T.ivector(name='x')
    self.idxs_to_vecs = theano.function(
      inputs=[x],
      outputs=self.L[x])
    
    #===================================================================
    # Convert vectors to idxs
    v = T.fmatrix(name='v')
    self.vecs_to_idxs_cos = theano.function(
      inputs=[v],
      outputs=T.argmax(cosine_similarity(v, self.L.T)))
    self.vecs_to_idxs_abs = theano.function(
      inputs=[v],
      outputs=T.argmin(T.sum(absolute_difference(v[:,None,:], self.L[None,:,:]), axis=2), axis=1))
    self.vecs_to_idxs_sqr = theano.function(
      inputs=[v],
      outputs=T.argmin(T.sum(squared_difference(v[:,None,:], self.L[None,:,:]), axis=2), axis=1))
  
  #=====================================================================
  # Convert vectors to idxs
  def vecs_to_idxs(self, v, dfunc='cosine'):
    """"""
    
    if dfunc == 'cosine':
      return self.vecs_to_idxs_cos(v)
    if dfunc == 'euclidean':
      return self.vecs_to_idxs_sqr(v)
    if dfunc == 'difference':
      return self.vecs_to_idxs_abs(v)
  
  #=====================================================================
  # Get word size
  def wsize(self):
    """"""
    
    return self._wsize
  
  #=====================================================================
  # Get start string
  def start_str(self):
    """"""
    
    return self._start
  
  #=====================================================================
  # Get stop string 
  def stop_str(self):
    """"""
    
    return self._stop
  
  #=====================================================================
  # Get unk string
  def unk_str(self):
    """"""
    
    return self._unk
  
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
    
    return np.array([self.idxs[s] for s in strs if s not in (self.unk_str(), self.stop_str(), self.start_str())], dtype='int32')#[None,:]
  
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
  def vecs_to_strs(self, vectors, dfunc='cosine'):
    """"""
    
    return self.idxs_to_strs(self.vecs_to_idxs(np.array(vectors), dfunc))
  
  #=====================================================================
  # Get tensor variable
  def get_subtensor(self, idxs):
    """"""
    
    return self.L[idxs]
 
#***********************************************************************
# GloVe library
class GloVe(Opt):
  """"""
  
  #=====================================================================
  # Initialize the model
  def __init__(self, wsize, **kwargs):
    """"""
    
    #-------------------------------------------------------------------
    # Arguments
    self._wsize = wsize
    
    #-------------------------------------------------------------------
    # Keyword arguments
    if 'start' in kwargs:
      self._start = kwargs['start']
    else:
      self._start = '<S>'
    
    if 'stop' in kwargs:
      self._stop = kwargs['stop']
    else:
      self._stop = '</S>'
    
    if 'unk' in kwargs:
      self._unk = kwargs['unk']
    else:
      self._unk = '<UNK>'
    
    if 'lower' in kwargs:
      self.lower = kwargs['lower']
    else:
      self.lower = True
    
    if 'hfunc' in kwargs:
      self.hfunc = kwargs['hfunc']
    else:
      self.hfunc = 'soft'
    
    if 'window' in kwargs:
      self.window = np.int32(kwargs['window'])
    else:
      self.window = np.int32(4)
    
    if 'L1reg' in kwargs:
      self.L1reg = np.float32(kwargs['L1reg'])
    else:
      self.L1reg = np.float32(0)
    
    if 'L2reg' in kwargs:
      self.L2reg = np.float32(kwargs['L2reg'])
    else:
      self.L2reg = np.float32(0)
    
    if 'xmax' in kwargs:
      self.xmax = np.float32(kwargs['xmax'])
    else:
      self.xmax = np.float32(100)
    
    if 'xmin' in kwargs:
      self.xmin = np.float32(kwargs['xmin'])
    else:
      self.xmin = np.float32(5)
    
    if 'alpha' in kwargs:
      self.alpha = np.float32(kwargs['alpha'])
    else:
      self.alpha = np.float32(.75)
    
    # V contains the individual counts, C contains the context counts
    self.V = Counter()
    self.C = defaultdict(Counter)
    self._keys = []
  
  #=====================================================================
  # Get the word size
  def wsize(self):
    """"""
    return self._wsize
  
  #=====================================================================
  # Build variables
  def build_vars(self):
    """"""
    
    self._keys = set(filter(lambda k: self.V[k] >= self.xmin, self.V.keys()) + [self._unk])
    self.strs = {}
    self.idxs = {}
    self.X = defaultdict(Counter)
    for index, string in enumerate(self._keys):
      self.strs[index]  = string
      self.idxs[string] = index
    for key1 in self.C:
      if key1 in self._keys:
        idx1 = self.idxs[key1]
      else:
        idx1 = self.idxs[self._unk]
      for key2 in self.C[key1]:
        if key2 in self._keys:
          idx2 = self.idxs[key2]
        else:
          idx2 = self.idxs[self._unk]
        self.X[idx1][idx2] += 1
    
    self.sparams  = []
    self.xparams  = []
    self.gsparams = []
    self.hmasks   = []
    
    self.L = theano.shared(np.random.randn(len(self._keys), self.wsize()).astype('float32'), name='L')
    self.L_tilde = theano.shared(np.random.randn(len(self._keys), self.wsize()).astype('float32'), name='L_tilde')
    self.hmasks.append(theano.shared(np.ones(self._wsize, dtype='float32'), name='hmask-L'))
    self.hmasks.append(theano.shared(np.ones(self._wsize, dtype='float32'), name='hmask-L_tilde'))
    self.Lb = theano.shared(np.zeros(len(self._keys), dtype='float32'), name='Lb')
    self.Lb_tilde = theano.shared(np.zeros(len(self._keys), dtype='float32'), name='Lb_tilde')
    self.params   = [self.L_tilde, self.Lb_tilde]
    self.gparams  = list([theano.shared(np.zeros_like(param.get_value()), name='g'+param.name) for param in self.params])
    self.sparams  = [self.L, self.Lb]
    self.gsparams = list([theano.shared(np.zeros_like(sparam.get_value()), name='g'+sparam.name) for sparam in self.sparams])
    
    if self.hfunc == 'soft':
      func = lambda y: T.tanh(y/(self.alpha*self.xmax))
    elif self.hfunc == 'sharp':
      func = lambda y: T.switch(T.power(y/self.xmax, self.alpha) > 1, np.float32(1), T.power(y/self.xmax, self.alpha))
    
    self.x = T.imatrix('x')
    self.xparams = [self.L[self.x[:,0]], self.Lb[self.x[:,1]]]
    x = (self.xparams[0]*self.hmasks[0]).T
    xb = self.xparams[1]
    self.y = T.fmatrix('y')
    
    yhat = T.dot(self.L_tilde*self.hmasks[1], x) + xb + self.Lb_tilde[:,None]
    logx = T.log(self.y)
    y    = T.switch(T.isinf(logx), np.float32(0), logx)
    f    = func(self.y)
    self.error = T.mean(T.sum(f * squared_difference(yhat, y), axis=0))
    
    self.complexity = theano.shared(np.float32(0))
    if self.L1reg > 0:
      self.complexity += self.L1reg*T.sum(T.abs_(self.L_tilde))
      self.complexity += self.L1reg*T.sum(T.abs_(x))
    if self.L2reg > 0:
      self.complexity += self.L2reg*T.sum(T.sqr(self.L))/np.float32(2)
      self.complexity += self.L2reg*T.sum(T.sqr(x))/np.float32(2)
    
    self.cost = self.error + self.complexity
    
    #===================================================================
    # Activate
    self.idxs_to_vec = theano.function(
      inputs=[self.x],
      outputs=yhat)
    
    #===================================================================
    # Error
    self.idxs_to_err = theano.function(
      inputs=[self.x, self.y],
      outputs=self.error)
    
    #===================================================================
    # Complexity
    self.idxs_to_comp = theano.function(
      inputs=[self.x],
      outputs=self.complexity,
      on_unused_input='ignore')
    
    #===================================================================
    # Cost
    self.idxs_to_cost = theano.function(
      inputs=[self.x, self.y],
      outputs=self.cost)
    
  #=====================================================================
  # Add a tokenized sentence to the statistics
  def add_tokenized_sent(self, sent):
    """"""
    
    sent.insert(0, self._start)
    sent.append(self._stop)
    for i, word in enumerate(sent):
      if self.lower and word not in (self._start, self._stop):
        word = word.lower()
      self.V[word] += 1
      for j in xrange(i-self.window, (i+1)+self.window):
        if j < 0:
          self.C[word][self._start] += 1
        elif j >= len(sent):
          self.C[word][self._stop] += 1
        elif j != i:
          if self.lower and sent[j] not in (self._start, self._stop):
            self.C[word][sent[j].lower()] += 1
          else:
            self.C[word][sent[j]] += 1
   
  #=====================================================================
  # Tokenize and add a sentence to the statistics
  def add_raw_sent(self, sent):
    """"""
    
    sent = word_tokenize(sent)
    self.add_tokenized_sent(sent)
  
  #=====================================================================
  # Add a list of tokenized sentences to the statistics
  def add_tokenized_sents(self, sents):
    """"""
    
    for sent in sents:
      self.add_tokenized_sent(sent)
  
  #=====================================================================
  # Tokenize and add a raw string to the statistics
  def add_raw_sents(self, sents):
    """"""
    
    for sent in sent_splitter.tokenize(sents):
      self.add_raw_sent(sent)
  
  #=====================================================================
  # Add a corpus from a file
  def add_corpus(self, corpus):
    """"""
    
    with codecs.open(corpus, encoding='utf-8') as f:
      for line in f:
        self.add_raw_sents(line.strip())
    self.build_vars()
  
  #=====================================================================
  # Calculate the cost of a minibatch
  def batch_cost(self, dataset, max_width=10000):
    """"""
    
    max_width = np.int32(max_width)
    cost = 0
    mb = -1
    for mb in xrange(len(dataset[0]) / max_width):
      y = np.zeros((len(self.X), max_width), dtype='float32')
      for i, idx1 in enumerate(dataset[0][mb*max_width:(mb+1)*max_width]):
        for idx2, count in self.X[idx1].iteritems():
          y[idx2][i] = count
      x = dataset[0][mb*max_width:(mb+1)*max_width, None]
      cost += self.idxs_to_cost(np.concatenate([x,x], axis=1), y)
    y = np.zeros((len(self.X), len(dataset[0]) % max_width), dtype='float32')
    for i, idx1 in enumerate(dataset[0][(mb+1)*max_width:]):
      for idx2, count in self.X[idx1].iteritems():
        y[idx2][i] = count
    x = dataset[0][(mb+1)*max_width:, None]
    cost += self.idxs_to_cost(np.concatenate([x,x], axis=1), y)
    return cost / len(dataset[0])
  
  #=====================================================================
  # Train the model
  def __train__(self, dataset, dataidxs, recentCost, momentizer, gradientizer, optimizer, nihilizer):
    """"""
    
    # Build the vectors in X for each index in the minibatch
    y = np.zeros((len(self._keys), len(dataidxs)), dtype='float32')
    for i, idx1 in enumerate(dataidxs):
      for idx2, count in self.X[idx1].iteritems():
        y[idx2][i] = count
    x = dataidxs[:,None] 
    
    # Do the thing
    momentizer(dataidxs, dataidxs)
    recentCost.append(gradientizer(np.concatenate([x,x], axis=1), y))
    optimizer(dataidxs, dataidxs)
    nihilizer(dataidxs, dataidxs)
  
#***********************************************************************
# A multilayer neural classifier
class Classifier(Opt):
  """"""
  
  #====================================================================
  # Initialize the network
  def __init__(self, libs, dims, **kwargs):
    """"""
    
    #------------------------------------------------------------------
    # Keyword arguments
    if 'hfunc' in kwargs:
      self.hfunc = kwargs['hfunc']
    else:
      self.hfunc = 'tanh'
      
    if 'L1reg' in kwargs:
      self.L1reg = np.float32(kwargs['L1reg'])
    else:
      self.L1reg = np.float32(0)
      
    if 'L2reg' in kwargs:
      self.L2reg = np.float32(kwargs['L2reg'])
    else:
      self.L2reg = np.float32(0)
    
    self.sparams  = []
    self.xparams  = []
    self.gsparams = []
    
    self.Wparams  = []
    self.Wbparams = []
    self.Lparams  = []
    self.Lbparams = []
    self.hmasks   = []
    
    #-------------------------------------------------------------------
    # Initialize the model params
    for i in xrange(1, len(dims)):
      self.Wparams.append(theano.shared(matwizard(dims[i], dims[i-1], output=self.hfunc, imput=(self.hfunc if i > 1 else '')).T, name='W-%d' % i))
      self.Wbparams.append(theano.shared(np.zeros(dims[i], dtype='float32'), name='b-%d' % i))
      self.hmasks.append(theano.shared(np.ones(dims[i], dtype='float32'), name='hmask-%d' % i))
    
    #-------------------------------------------------------------------
    # Initialize the classifier params
    self.libs = []
    for l, lib in enumerate(libs):
      if not isinstance(lib, Library):
        lib = Library(*lib)
      self.libs.append(lib)
      self.Lparams.append(theano.shared(matwizard(len(lib.idxs), dims[-1], output='softmax').T, name='W%d-L%d' % (i+1, l)))
      self.Lbparams.append(theano.shared(np.zeros(len(lib.idxs), dtype='float32'), name='b%d-L%d' % (i+1, l)))
    
    #-------------------------------------------------------------------
    # Build the input/output variables
    self.x = T.fmatrix('x')
    self.y = T.imatrix('y')
    
    #-------------------------------------------------------------------
    # Bundle the params
    self.params =\
        self.Wparams +\
        self.Lparams +\
        self.Wbparams +\
        self.Lbparams
    self.gparams = [theano.shared(np.zeros_like(param.get_value()), name='g'+param.name) for param in self.params]
    
    #-------------------------------------------------------------------
    # Build the hidden variables
    self.h = [self.x]
    for Wparam, Wbparam, hmask in zip(self.Wparams, self.Wbparams, self.hmasks):
      a = T.dot(self.h[-1], Wparam)
      a += Wbparam
      h = funx[self.hfunc](a)*hmask
      self.h.append(h)
      
    self.prob = []
    for Lparam, Lbparam in zip(self.Lparams, self.Lbparams):
      a = T.dot(self.h[-1], Lparam)
      a += Lbparam
      o = softmax(a)
      self.prob.append(o)
    self.prediction = [T.argmax(o, axis=1) for o in self.prob]
    
    #-------------------------------------------------------------------
    # Build the cost variable
    self.error = np.float32(0)
    for i in xrange(len(self.prob)):
      self.error += T.nnet.categorical_crossentropy(self.prob[i], self.y[:,i])
    self.error = T.mean(self.error)
    
    self.complexity = theano.shared(np.float32(0))
    if self.L1reg > 0:
      self.complexity += self.L1reg*T.sum([T.sum(T.abs_(Wparam)) for Wparam in self.Wparams])
      self.complexity += self.L1reg*T.sum([T.sum(T.abs_(Lparam)) for Lparam in self.Lparams])
    if self.L2reg > 0:
      self.complexity += self.L2reg*T.sum([T.sum(T.sqr(Wparam)) for Wparam in self.Wparams])
      self.complexity += self.L2reg*T.sum([T.sum(T.sqr(Lparam)) for Wparam in self.Lparams])
    
    self.cost = self.error + self.complexity
    
    #===================================================================
    # Activate
    self.vecs_to_vecs = theano.function(
      inputs=[self.x],
      outputs=self.prob)
    
    #===================================================================
    # Predict
    self.vecs_to_idxs = theano.function(
      inputs=[self.x],
      outputs=self.prediction)
    
    #===================================================================
    # Error
    self.vecs_to_error = theano.function(
      inputs=[self.x, self.y],
      outputs=self.error)
    
    #===================================================================
    # Complexity
    self.vecs_to_comp = theano.function(
      inputs=[self.x],
      outputs=self.complexity,
      on_unused_input='ignore')
    
    #===================================================================
    # Cost
    self.vecs_to_cost = theano.function(
      inputs=[self.x, self.y],
      outputs=self.cost)
      
  #=====================================================================
  # Convert the dataset to the expected format
  def convert_dataset(self, dataset, list_of_pairs=True):
    """"""
    
    if list_of_pairs:
      return [np.array([datum[0] for datum in dataset]).astype('float32'), np.array([datum[1] for datum in dataset]).astype('int32')]
    else:
      return [np.array(datum[0]), np.array(datum[1])]
  
  #=====================================================================
  # Calculate the cost of the dataset
  def batch_cost(self, dataset):
    """"""
    
    return self.vecs_to_cost(*dataset)
  
  #=====================================================================
  # Classifier training function
  def __train__(self, dataset, dataidxs, recentCost, momentizer, gradientizer, optimizer, nihilizer):
    """"""
    
    momentizer()
    recentCost.append(gradientizer(dataset[0][dataidxs], dataset[1][dataidxs]))
    optimizer()
    nihilizer()
  
#***********************************************************************
# A multilayer recurrent neural encoder
class Encoder(Opt):
  """"""
  
  #=====================================================================
  # Initialize the model
  def __init__(self, libs, dims, **kwargs):
    """"""
    
    #-------------------------------------------------------------------
    # Keyword arguments
    if 'model' in kwargs:
      self.model = kwargs['model']
    else:
      self.model = 'RNN'
    
    if 'hfunc' in kwargs:
      self.hfunc = kwargs['hfunc']
    else:
      self.hfunc = 'tanh'
    
    if 'window' in kwargs:
      self.window = np.int32(kwargs['window'])
    else:
      self.window = np.int32(1)
    
    if 'reverse' in kwargs:
      self.reverse = kwargs['reverse']
    else:
      self.reverse = False
    
    if 'L1reg' in kwargs:
      self.L1reg = np.float32(kwargs['L1reg'])
    else:
      self.L1reg = np.float32(0)
    
    if 'L2reg' in kwargs:
      self.L2reg = np.float32(kwargs['L2reg'])
    else:
      self.L2reg = np.float32(0)
    
    self.sparams  = []
    self.xparams  = []
    self.gsparams = []
    
    self.Wparams  = []
    self.Wbparams = []
    self.h_0      = []
    self.c_0      = []
    self.hmasks   = []
    
    #-------------------------------------------------------------------
    # Process the libraries
    self.libs = []
    for i, lib in enumerate(libs):
      if not isinstance(lib, Library):
        lib = Library(*lib)
      self.libs.append(lib)
      self.sparams.append(lib.L)
      self.gsparams.append(lib.gL)
      lib.L.name = 'L-%d' % (i+1)
      lib.gL.name = 'gL-%d' % (i+1)
    assert dims[0] == np.sum([lib.wsize() for lib in inlibs])*self.window
    
    #-------------------------------------------------------------------
    # Initialize the model params
    if self.model in ('RNN',):
      gates = 0
    elif self.model in ('FastGRU',): # FastGRU ditches the reset gate
      gates = 1
    elif self.model in ('GRU', 'FastLSTM'): # FastLSTM couples the input/forget gates
      gates = 2
    elif self.model in ('LSTM',):
      gates = 3
    
    for i in xrange(1, len(dims)):
      W = matwizard(dims[i], dims[i-1], shape='rect', output=self.hfunc, imput=(self.hfunc if i > 1 else ''), recur=True)
      U = matwizard(dims[i], dims[i], shape='diag', output=self.hfunc, imput=self.hfunc, recur=True)
      if gates > 1:
        W = np.concatenate([W, matwizard(dims[i]*gates, dims[i-1], output='sigmoid', imput=(self.hfunc if i > 1 else ''), recur=True)], axis=0)
        W = np.concatenate([W, matwizard(dims[i]*gates, dims[i-1], output='sigmoid', imput=(self.hfunc if i > 1 else ''), recur=True)], axis=0)
      self.Wparams.append(theano.shared(np.concatenate([W, U], axis=1), name='W-%d' % i))
      self.Wbparams.append(theano.shared(np.zeros(dims[i]*(gates+1), dtype='float32'), name='b-%d' % i))
      self.hmasks.append(theano.shared(np.ones(dims[i], dtype='float32'), name='hmask-%d' % i))
      self.h_0.append(theano.shared(np.zeros(dims[i], dtype='float32'), name='h_0-%d' % i))
      self.c_0.append(theano.shared(np.zeros(dims[i], dtype='float32'), name='c_0-%d' % i))
    
    #-------------------------------------------------------------------
    # Bundle the params
    self.params =\
      self.Wparams +\
      self.Wbparams +\
      self.h_0 +\
      (self.c_0 if self.model.endswith('LSTM') else [])
    self.gparams = list([theano.shared(np.zeros_like(param.get_value()), name='g'+param.name) for param in self.params])
    
    #-------------------------------------------------------------------
    # Build the input/output variables
    self.x = T.imatrix('x')
    x = self.x[::-1] if self.reverse else self.x
    for i, lib in enumerate(self.libs):
      self.xparams.append(lib.get_subtensor(x[:,i]))
    x_0 = T.concatenate([xparam*lib.hmask for xparam in self.xparams], axis=1)
    self.y = T.fvector('y')
    
    #-------------------------------------------------------------------
    # Build the activation variable
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if self.model == 'RNN':
      def recur(i, *args):
        """"""
        
        c_tm1 = args[:len(self.c_0)]
        h_tm1 = args[len(self.c_0):len(self.c_0)+len(self.h_0)]
        c_t   = []
        h_t   = [x_0[i:i+self.window].flatten()]
        for h_tm1_l, Wparam, Wbparam, hmask in zip(h_tm1, self.Wparams, self.Wbparams, self.hmasks):
          x_t_l = T.concatenate([h_t[-1], h_tm1_l])
          
          a = T.dot(Wparam, x_t_l) + Wbparam
          
          c = a
          h = funx[self.hfunc](c)*hmask
          
          c_t.append(c)
          h_t.append(h)
        
        return c_t + h_t[1:]
      
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif self.model == 'GRU':
      def recur(i, *args):
        """"""
        
        c_tm1 = args[:len(self.c_0)]
        h_tm1 = args[len(self.c_0):len(self.c_0)+len(self.h_0)]
        c_t   = []
        h_t   = [x_0[i:i+self.window].flatten()]
        for h_tm1_l, Wparam, Wbparam, hmask in zip(h_tm1, self.Wparams, self.Wbparams, self.hmasks):
          sliceLen = Wparam.shape[0]/3
          x_t_l = T.concatenate([h_t[-1], h_tm1_l])
          
          zr = T.dot(Wparam[sliceLen:], x_t_l) + Wbparam[sliceLen:]
          z = sigmoid(zr[:sliceLen])
          r = sigmoid(zr[sliceLen:])
          a = T.dot(Wparam[:sliceLen], T.concatenate([x_t_l[:h_t[-1].shape[0]], x_t_l[h_t[-1].shape[0]:]*r])) + Wbparam[:sliceLen]
          
          c = z*funx[self.hfunc](a) + (np.float32(1)-z)*h_tm1_l
          h = c*hmask
          
        return c_t + h_t[1:]
      
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if self.model == 'FastGRU':
      def recur(i, *ch_tm1):
        """"""
        
        c_tm1 = args[:len(self.c_0)]
        h_tm1 = args[len(self.c_0):len(self.c_0)+len(self.h_0)]
        c_t   = []
        h_t   = [x_0[i:i+self.window].flatten()]
        for h_tm1_l, Wparam, Wbparam, hmask in zip(h_tm1, self.Wparams, self.Wbparams, self.hmasks):
          sliceLen = Wparam.shape[0]/2
          x_t_l = T.concatenate([_t[-1], h_tm1_l])
          
          az = T.dot(Wparam, x_t_l) + Wbparam
          a = az[:sliceLen]
          z = sigmoid(az[sliceLen:])
          
          c = z*funx[self.hfunc](a) + (np.float32(1)-z)*h_tm1_l
          h = c*hmask
          
        return c_t + h_t[1:]
      
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if self.model == 'LSTM':
      def recur(i, *ch_tm1):
        """"""
        
        c_tm1 = args[:len(self.c_0)]
        h_tm1 = args[len(self.c_0):len(self.c_0)+len(self.h_0)]
        c_t   = []
        h_t   = [x_0[i:i+self.window].flatten()]
        for h_tm1_l, c_tm1_l, Wparam, Wbparam, hmask in zip(h_tm1, c_tm1, self.Wparams, self.Wbparams, self.hmasks):
          sliceLen = Wparam.shape[0]/4
          x_t_l = T.concatenate([_t[-1], h_tm1_l])
          
          aifo = T.dot(Wparam, x_t_l) + Wbparam
          a    = funx[self.hfunc](aifo[:sliceLen])
          i    = sigmoid(aifo[sliceLen:2*sliceLen])
          f    = sigmoid(aifo[2*sliceLen:3*sliceLen])
          o    = sigmoid(aifo[3*sliceLen:])
          
          c = i*a + f*c_tm1_l
          h = funx[self.hfunc](c*o)*hmask
          
          c_t.append(c)
          h_t.append(h)
        
        return c_t + h_t[1:]
      
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if self.model == 'FastLSTM':
      def recur(i, *ch_tm1):
        """"""
        
        c_tm1 = args[:len(self.c_0)]
        h_tm1 = args[len(self.c_0):len(self.c_0)+len(self.h_0)]
        c_t   = []
        h_t   = [x_0[i:i+self.window].flatten()]
        for h_tm1_l, c_tm1_l, Wparam, Wbparam, hmask in zip(h_tm1, c_tm1, self.Wparams, self.Wbparams, self.hmasks):
          sliceLen = Wparam.shape[0]/3
          x_t_l = T.concatenate([_t[-1], h_tm1_l])
          
          azo = T.dot(Wparam, x_t_l) + Wbparam
          a   = funx[self.hfunc](azo[:sliceLen])
          z   = sigmoid(azo[sliceLen:2*sliceLen])
          o   = sigmoid(azo[2*sliceLen:])
          
          c = z*a + (np.float32(1)-z)*c_tm1_l
          h = funx[self.hfunc](c*o)*hmask
          
          c_t.append(c)
          h_t.append(h)
        
        return c_t + h_t[1:]
    
    states, _ = theano.scan(
      fn=recur,
      sequences=T.arange(self.x.shape[0]-(self.window-1)),
      outputs_info=self.c_0+self.h_0,
      non_sequences=self.Wparams+self.Wbparams+self.hmasks+[x_0],
      strict=True)
    
    self.c = states[:len(states)/2]
    self.h = states[len(states)/2:]
    yhat = self.c[-1][-1]
    
    #-------------------------------------------------------------------
    # Build the cost variable
    self.error = T.sum(squared_difference(yhat, self.y))
    
    self.complexity = theano.shared(np.float32(0))
    if self.L1reg > 0:
      self.complexity += self.L1reg*T.sum([T.sum(T.abs_(Wparam)) for Wparam in self.Wparams])
      self.complexity += self.L1reg*T.sum([T.sum(T.abs_(xparam)) for lib, xparam in zip(self.libs, self.xparams) if lib.mutable()])
 
    if self.L2reg > 0:
      self.complexity += self.L2reg*T.sum([T.sum(T.sqr(Wparam)) for Wparam in self.Wparams])
      self.complexity += self.L2reg*T.sum([T.sum(T.sqr(xparam)) for lib, xparam in zip(self.libs, self.xparams) if lib.mutable()])

    self.cost = self.error + self.complexity
    
    #===================================================================
    # Activate
    self.idxs_to_vec = theano.function(
      inputs=[self.x],
      outputs=yhat)
    
    #===================================================================
    # Error
    self.idxs_to_err = theano.function(
      inputs=[self.x, self.y],
      outputs=self.error)
    
    #===================================================================
    # Complexity
    self.idxs_to_comp = theano.function(
      inputs=[self.x],
      outputs=self.complexity,
      on_unused_input='ignore')
    
    #===================================================================
    # Cost
    self.idxs_to_cost = theano.function(
      inputs=[self.x, self.y],
      outputs=self.cost)
    
  #=====================================================================
  # Pad a list of strings or string tuples
  def pad_strs(self, strings):
    """"""
    
    # Define what we're looking for
    if isinstance(strings[0], basestring):
      strings = list([(s,) for s in strings])
    else:
      strings = list(strings)
    begins = tuple([lib.start_str() for lib in self.libs])
    ends   = tuple([lib.stop_str()  for lib in self.libs])
    
    # Count the beginning 
    nbegins = 0
    while strings[nbegins] == begins:
      nbegins += 1
    
    # Count the end
    nends = 0
    while strings[~nends] == ends:
      nends += 1
    
    if self.reverse:
      if nbegins < 1:
        strings.append(begins)
      while nends < self.window:
        strings.insert(0, ends)
        nends += 1
    else:
      while nbegins < self.window:
        strings.insert(0, begins)
        nbegins += 1
      if nends < 1:
        strings.append(ends)
    return strings
  
  #=====================================================================
  # Unpad a list of strings or string tuples (in place)
  def unpad_strs(self, strings):
    """"""
    
    begins = tuple([lib.start_str() for lib in self.libs])
    ends   = tuple([lib.stop_str()  for lib in self.libs])
    
    while strings[0] == begins:
      strings.pop(0)
    while strings[-1] == ends:
      strings.pop()
    
  #=====================================================================
  # Converts a list of strings or string tuples into a matrix
  def strs_to_idxs(self, strings):
    """"""
    
    strings = self.pad_strs(strings)
    indices = []
    for i, lib in enumerate(self.libs):
      indices.append(lib.strs_to_idxs([s[i] for s in strings])[:,None])
    return np.concatenate(indices, axis=1)
  
  #=====================================================================
  # Converts a list of input strings into a vector
  def strs_to_vec(self, strings):
    """"""
    
    return self.idxs_to_vec(self.strs_to_idxs(strings))
  
  #=====================================================================
  # Converts a list of input strings into a vector
  def strs_to_err(self, strings, vector):
    """"""
    
    return self.idxs_to_err(self.strs_to_idxs(strings), vector)
  
  #=====================================================================
  # Converts a list of input strings into a vector
  def strs_to_comp(self, strings):
    """"""
    
    return self.idxs_to_comp(self.strs_to_idxs(strings))
  
  #=====================================================================
  # Converts a list of input strings into a vector
  def strs_to_err(self, strings, vector):
    """"""
    
    return self.idxs_to_err(self.strs_to_idxs(strings), vector)
  
  #=====================================================================
  # Converts a dataset into the expeted format 
  def convert_dataset(self, dataset, list_of_pairs=True):
    """"""
    
    if list_of_pairs:
      x = list([self.strs_to_idxs(datum[0]) for datum in dataset])
      y = list([datum[1] for datum in dataset])
      return [x, np.array(y)]
    else:
      return [list([self.strs_to_idxs(x) for x in dataset[0]]), np.array(dataset[1])]
   
  #=====================================================================
  # Calculate the cost of a minibatch
  def batch_cost(self, dataset):
    """"""
    
    cost = 0
    for i in xrange(len(dataset)):
      cost += self.idxs_to_cost(dataset[0][i], dataset[1][i])
    return cost / len(dataset)
    
  
  #=====================================================================
  # The encoder's training function
  def __train__(self, dataset, dataidxs, recentCost, momentizer, gradientizer, optimizer, nihilizer):
    """"""
    
    # Collect the sparse indices
    gidxs = list([set() for lib in self.libs])
    for i in xrange(len(self.libs)):
      for idx in dataidxs:
        gidxs[i].update(dataset[0][idx][:,i])
      gidxs[i] = list(gidxs[i])
    
    # Do the thing
    momentizer(*gidxs)
    recentCost.append(0)
    for idx in dataidxs:
      recentCost[-1] += gradientizer(dataset[0][idx], dataset[1][idx])
    recentCost[-1] /= len(dataidxs)
    optimizer(*gidxs)
    nihilizer(*gidxs)
  
##***********************************************************************
## A mulitlayer neural language model
#class LangModel(Opt):
#  """"""
#  
#  #=====================================================================
#  # Initialize the model
#  def __init__(self, inlibs, outlibs, dims, **kwargs):
#    """"""
#    
#    #-------------------------------------------------------------------
#    # Keyword arguments
#    if 'model' in kwargs:
#      self.model = kwargs['model']
#    else:
#      self.model = 'RNN'
#    
#    if 'hfunc' in kwargs:
#      self.hfunc = kwargs['hfunc']
#    else:
#      self.hfunc = 'tanh'
#    
#    if 'window' in kwargs:
#      self.window = np.int32(kwargs['window'])
#    else:
#      self.window = np.int32(1)
#    
#    if 'reverse' in kwargs:
#      self.reverse = kwargs['reverse']
#    else:
#      self.reverse = False
#    
#    if 'L1reg' in kwargs:
#      self.L1reg = np.float32(kwargs['L1reg'])
#    else:
#      self.L1reg = np.float32(0)
#    
#    if 'L2reg' in kwargs:
#      self.L2reg = np.float32(kwargs['L2reg'])
#    else:
#      self.L2reg = np.float32(0)
#    
#    self.sparams  = []
#    self.xparams  = []
#    self.gsparams = []
#    
#    self.Wparams  = []
#    self.Wbparams = []
#    self.Lparams  = []
#    self.Lbparams = []
#    self.h_0      = []
#    self.c_0      = []
#    self.hmasks   = []
#    
#    #-------------------------------------------------------------------
#    # Process the input libraries
#    self.libs = []
#    for i, lib in enumerate(inlibs):
#      if not isinstance(lib, Library):
#        lib = Library(*lib)
#      self.libs.append(lib)
#      self.sparams.append(lib.L)
#      self.gsparams.append(lib.gL)
#      lib.L.name = 'L-%d' % (i+1)
#      lib.gL.name = 'gL-%d' % (i+1)
#    assert dims[0] == np.sum([lib.wsize() for lib in self.libs])*self.window
#    
#    #-------------------------------------------------------------------
#    # Initialize the model params
#    if self.model in ('RNN',):
#      gates = 0
#    elif self.model in ('FastGRU',): # FastGRU ditches the reset gate
#      gates = 1
#    elif self.model in ('GRU', 'FastLSTM'): # FastLSTM couples the input/forget gates
#      gates = 2
#    elif self.model in ('LSTM',):
#      gates = 3
#    
#    for i in xrange(1, len(dims)):
#      W = matwizard(dims[i], dims[i-1], shape='rect', output=self.hfunc, imput=(self.hfunc if i > 1 else ''), recur=True)
#      U = matwizard(dims[i], dims[i], shape='diag', output=self.hfunc, imput=self.hfunc, recur=True)
#      if gates > 1:
#        W = np.concatenate([W, matwizard(dims[i]*gates, dims[i-1], output='sigmoid', imput=(self.hfunc if i > 1 else ''), recur=True)], axis=0)
#        W = np.concatenate([W, matwizard(dims[i]*gates, dims[i-1], output='sigmoid', imput=(self.hfunc if i > 1 else ''), recur=True)], axis=0)
#      self.Wparams.append(theano.shared(np.concatenate([W, U], axis=1), name='W-%d' % i))
#      self.Wbparams.append(theano.shared(np.zeros(dims[i]*(gates+1), dtype='float32'), name='b-%d' % i))
#      self.hmasks.append(theano.shared(np.ones(dims[i], dtype='float32'), name='hmask-%d' % i))
#      self.h_0.append(theano.shared(np.zeros(dims[i], dtype='float32'), name='h_0-%d' % i))
#      self.c_0.append(theano.shared(np.zeros(dims[i], dtype='float32'), name='c_0-%d' % i))
#        
#    #-------------------------------------------------------------------
#    # Bundle the params
#    self.params =\
#      self.Wparams +\
#      self.Wbparams +\
#      self.h_0 +\
#      (self.c_0 if self.model.endswith('LSTM') else [])
#    self.gparams = list([theano.shared(np.zeros_like(param.get_value()), name='g'+param.name) for param in self.params])
#    
#    #-------------------------------------------------------------------
#    # Process the output libraries
#    for l, lib in enumerate(outlibs):
#      if not isinstance(lib, Library):
#        lib = Library(*lib)
#      assert dims[-1] == lib.wsize()
#      self.Lparams.append(lib.L)
#      self.params.append(lib.L)
#      self.gparams.append(lib.gL)
#    assert dims[-1] == np.sum([lib.wsize() for lib in outlibs])
    
##***********************************************************************
## A multilayer recurrent neural decoder
## TODO decoder always needs a lib as input, even if it only contains <S> and </S>
#class Decoder(Opt):
#  """"""
#  
#  #=====================================================================
#  # Initialize the model
#  def __init__(self, libs, dims, **kwargs):
#    """"""
#    
#    #-------------------------------------------------------------------
#    # Keyword arguments
#    if 'model' in kwargs:
#      self.model = kwargs['model']
#    else:
#      self.model = 'RNN'
#    
#    if 'hfunc' in kwargs:
#      self.hfunc = kwargs['hfunc']
#    else:
#      self.hfunc = 'tanh'
#    
#    if 'L1reg' in kwargs:
#      self.L1reg = np.float32(kwargs['L1reg'])
#    else:
#      self.L1reg = np.float32(0)
#    
#    if 'L2reg' in kwargs:
#      self.L2reg = np.float32(kwargs['L2reg'])
#    else:
#      self.L2reg = np.float32(0)
#    
#    self.sparams  = []
#    self.xparams  = []
#    self.gsparams = []
#    self.libs = []
#    
#    self.Wparams  = []
#    self.Wbparams = []
#    self.c_0      = []
#    self.h_0      = []
#    self.o_0      = []
#    self.hmasks   = []
#    
#    #-------------------------------------------------------------------
#    # Initialize the model params
#    if self.model in ('RNN',):
#      gates = 0
#    elif self.model in ('FastGRU',): # FastGRU ditches the reset gate
#      gates = 1
#    elif self.model in ('GRU', 'FastLSTM'): # FastLSTM couples the input/forget gates
#      gates = 2
#    elif self.model in ('LSTM',):
#      gates = 3
#    
#    for i in xrange(1, len(dims)):
#      W = matwizard(dims[i], dims[i-1], shape='rect', output=self.hfunc, imput=(self.hfunc if i > 1 else ''), recur=True)
#      U = matwizard(dims[i], dims[i], shape='diag', output=self.hfunc, imput=self.hfunc, recur=True)
#      if gates > 1:
#        W = np.concatenate([W, matwizard(dims[i]*gates, dims[i-1], output='sigmoid', imput=(self.hfunc if i > 1 else ''), recur=True)], axis=0)
#        U = np.concatenate([U, matwizard(dims[i]*gates, dims[i], output='sigmoid', imput=(self.hfunc if i > 1 else ''), recur=True)], axis=0)
#      self.Wparams.append(theano.shared(np.concatenate([W, U], axis=1), name='W-%d' % i))
#      self.Wbparams.append(theano.shared(np.zeros(dims[i]*(gates+1), dtype='float32'), name='b-%d' % i))
#      self.hmasks.append(theano.shared(np.ones(dims[i], dtype='float32'), name='hmask-%d' % i))
#      self.h_0.append(theano.shared(np.zeros(dims[i], dtype='float32'), name='h_0-%d' % i))
#      self.c_0.append(theano.shared(np.zeros(dims[i], dtype='float32'), name='c_0-%d' % i))
#    self.o_0.append(theano.shared(np.zeros(dims[i], dtype='float32'), name='h_0-%d' % i))
#    
#    #-------------------------------------------------------------------
#    # Bundle the params
#    self.params =\
#      self.Wparams +\
#      self.Wbparams +\
#      self.h_0 +\
#      (self.c_0 if self.model.endswith('LSTM') else [])
#    self.gparams = [theano.shared(np.zeros_like(param.get_value()), name='g'+param.name) for param in self.params]
#    
#    #-------------------------------------------------------------------
#    # Build the input/output variables
#    self.x = T.fvector('x')
#    self.y = T.fmatrix('y')
#    
#    #-------------------------------------------------------------------
#    # Build the activation variable
#    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#    if self.model == 'RNN':
#      def recur(*args):
#        """"""
#        
#        c_tm1 = args[:len(self.c_0)]
#        h_tm1 = args[len(self.c_0):len(self.c_0)+len(self.h_0)]
#        o_tm1 = args[len(self.c_0)+len(self.h_0):len(self.c_0)+len(self.h_0)+len(self.o_0)]
#        c_t   = []
#        h_t   = [T.concatenate([self.x] + o_tm1)]
#        for h_tm1_l, Wparam, bparam, hparam, hmask in zip(h_tm1, self.Wparams, self.bparams, self.hparams, self.hmasks):
#          x_t_l = T.concatenate([h_t[-1], h_tm1_l])
#          
#          a = T.dot(Wparam, x_t_l) + Wbparam
#          
#          c = a
#          h = funx[self.hfunc](c)*hmask
#          
#          c_t.append(c)
#          h_t.append(h)
#        
#        self.o_t = [T.Variable(h_t[-1]) for o in self.o_0]
#        return c_t + h_t[1:] + self.o_t
#    
#    self.until = 
#    states, _ = theano.scan(
#      fn=recur,
#      sequences=None,
#      outputs_info=self.c_0+self.h_0+self.o_0,
#      non_sequences=self.Wparams+self.Wbparams+self.hmasks,
#      strict=True)
#    
#    self.c = states[:len(self.c_0)]
#    self.h = states[len(self.c_0):len(self.h_0)]
#    self.o = states[len(self.c_0)+len(self.h_0):]
#    yhat = self.c[-1]
#    
#    #-------------------------------------------------------------------
#    # Build the cost variable
#    self.error = T.sum(T.mean(squared_difference(yhat, self.y), axis=1))
#    
#    self.complexity = theano.shared(np.float32(0))
#    if self.L1reg > 0:
#      self.complexity += self.L1reg*T.sum([T.sum(T.abs_(Wparam)) for Wparam in self.Wparams])
#    
#    if self.L2reg > 0:
#      self.complexity += self.L2reg*T.sum([T.sum(T.sqr(Wparam)) for Wparam in self.Wparams])
#    
#    self.cost = self.error + self.complexity
#    
#    #===================================================================
#    # Activate
#    self.vecs_to_vec = theano.function(
#      inputs=[self.x],
#      outputs=yhat)
#    
#    #===================================================================
#    # Error
#    self.vecs_to_err = theano.function(
#      inputs=[self.x, self.y],
#      outputs=self.error)
#    
#    #===================================================================
#    # Complexity
#    self.vecs_to_comp = theano.function(
#      inputs=[self.x],
#      outputs=self.complexity,
#      on_unused_input='ignore')
#    
#    #===================================================================
#    # Cost
#    self.vecs_to_cost = theano.function(
#      inputs=[self.x, self.y],
#      outputs=self.cost)
    
#***********************************************************************
# Test the program
if __name__ == '__main__':
  """"""
  
  import os.path
  EPOCHS=200
  PATH='Glove Compartment'
  HFUNC='tanh'
  DIM=300
  glove = GloVe(DIM, xmin=3, xmax=100, window=8, L2reg=1e-5, hfunc='sharp')
  glove.add_corpus('LotR.txt')
  #mom, grad, opt, nihil = glove.SGD(eta_0=.1, anneal=.5, T_eta=1000, dropout=9./10)
  mom, grad, opt, nihil = glove.RMSProp(eta_0=.01, dropout=9./10)
  #mom, grad, opt, nihil = glove.AdaDelta(eta_0=1, dropout=9./10)
  #mom, grad, opt, nihil = glove.Adam(eta_0=.01, dropout=9./10)
  name=os.path.join(PATH, 'glv-SGD-%d-'%DIM)
  glove.train([np.arange(len(glove.X)).astype('int32')], mom, grad, opt, nihil, saveName=name, costEvery=10, epochs=EPOCHS, batchSize=32)
  print len(glove._keys)
  #glove = pkl.load(open('glove.6B.10k.50d-real.pkl'))
  #glove[1]['<S>'] = len(glove[1])
  #glove[1]['</S>'] = len(glove[1])
  #glove[1]['<UNK>'] = len(glove[1])
  
  #=====================================================================
  # Test the Classifier
  #ETA_0=1
  #mat = np.concatenate([glove[0], np.random.randn(3,50)])
  #mat = (mat-np.mean(mat))/np.std(mat)
  #lib = Library(keys=glove[1], mat=mat)
  #classifier = Classifier([lib], [50, 50], hfunc=HFUNC)
  #
  #dataset = []
  #for i in xrange(5):
  #  dataset.extend(list(enumerate(mat+np.random.normal(0,.1,size=mat.shape))))
  #for i, datum in enumerate(dataset):
  #  dataset[i] = (datum[1], [datum[0]])
  #train_data = dataset[:int(.8*len(dataset))]
  #dev_data = dataset[int(.8*len(dataset)):int(.9*len(dataset))]
  #test_data = dataset[int(.9*len(dataset)):]
  #train_data = classifier.convert_dataset(train_data)
  #dev_data = classifier.convert_dataset(dev_data)
  #test_data = classifier.convert_dataset(test_data)
  #
  #mom, grad, opt, nihil = classifier.SGD(eta_0=ETA_0)
  #parentPipe, childPipe = mp.Pipe()
  #process = mp.Process(target=pkl_worker, args=(childPipe,), kwargs={'path':PATH, 'name': 'cls-'})
  #process.start()
  #classifier.train(train_data, mom, grad, opt, nihil, savePipe=parentPipe, costEvery=10, epochs=EPOCHS, testset=dev_data)
  #process.join()
  
  ##=====================================================================
  ## Test the Encoder
  #ETA_0=.01
  #T_ETA = 250
  #mat = np.concatenate([glove[0], np.random.randn(3,50)])
  #mat = (mat-np.mean(mat))/np.std(mat)
  #lib = Library(keys=glove[1], mat=mat)
  #encoder = Encoder([lib], [50,200,50], window=1, hfunc=HFUNC)
  #
  #dataset = []
  #for i in xrange(5):
  #  dataset.extend(zip([list(x) for x in sorted(glove[1], key=glove[1].get) if x not in (lib.unk_str(), lib.stop_str(), lib.start_str())], (mat+np.random.normal(0,.1,size=mat.shape)).astype('float32')))
  #train_data = dataset[:int(.8*len(dataset))]
  #dev_data = dataset[int(.8*len(dataset)):int(.9*len(dataset))]
  #test_data = dataset[int(.9*len(dataset)):]
  #train_data = encoder.convert_dataset(train_data)
  #dev_data = encoder.convert_dataset(dev_data)
  #test_data = encoder.convert_dataset(test_data)
  #
  #mom, grad, opt, nihil = encoder.SGD(eta_0=ETA_0, T_eta=T_ETA, anneal=ANNEAL)
  #parentPipe, childPipe = mp.Pipe()
  #process = mp.Process(target=pkl_worker, args=(childPipe,), kwargs={'path':PATH, 'name': 'enc-'})
  #process.start()
  #encoder.train(train_data, mom, grad, opt, nihil, savePipe=parentPipe, costEvery=10, epochs=EPOCHS, testset=dev_data)
  #process.join()
  
  #=====================================================================
  # Test the Decoder
  #ETA_0 = .01
  #T_ETA = 250
  #mat = np.concatenate([glove[0], np.random.randn(3,50)])
  #mat = (mat-np.mean(mat))/np.std(mat)
  #lib = Library(keys=glove[1], mat=mat)
  #decoder = Decoder([lib], [50,200,50], window=1, hfunc=HFUNC)
  #
  #dataset = []
  #for i in xrange(5):
  #  dataset.extend(zip((mat+np.random.normal(0,.1,size=mat.shape)).astype('float32'), lib.strs_to_vecs([list(x) for x in sorted(glove[1], key=glove[1].get) if x not in (lib.unk_str(), lib.stop_str(), lib.start_str())])))
  #train_data = dataset[:int(.8*len(dataset))]
  #dev_data = dataset[int(.8*len(dataset)):int(.9*len(dataset))]
  #test_data = dataset[int(.9*len(dataset)):]
  #train_data = decoder.convert_dataset(train_data)
  #dev_data = decoder.convert_dataset(dev_data)
  #test_data = decoder.convert_dataset(test_data)
  #
  #mom, grad, opt, nihil = decoder.SGD(eta_0=ETA_0, T_eta=T_ETA, anneal=ANNEAL)
  #parentPipe, childPipe = mp.Pipe()
  #process = mp.Process(target=pkl_worker, args=(childPipe,), kwargs={'path':PATH, 'name': 'enc-'})
  #process.start()
  #decoder.train(train_data, mom, grad, opt, nihil, savePipe=parentPipe, costEvery=10, epochs=EPOCHS, testset=dev_data)
  #process.join()
  