#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matwizard import matwizard
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams()
from funx import splice_funx, clip_funx, error_funx, sigmoid, softmax, squared_difference, reweight_glove
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
# Anything that can be optimized
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
    s += ': %.3f train cost' % cost[-1]
    test = None
    nsaves = 1
    if testset is not None:
      test = []
      test.append(self.batch_cost(testset))
      s += ', %.3f test cost' % test[-1]
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
          s += ': %.3f train cost' % cost[-1]
          if testset is not None:
            s += ': %.3f test cost' % test[-1]
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
      s += ': %.3f train cost' % cost[-1]
      if testset is not None:
        s += ': %.3f test cost' % test[-1]
      if wps == 0:
        wps = .67*wps + .33*((batchSize*((mb+1) % costEvery) if costEvery is not None else len(dataset[0])) / (thisCostTime-lastCostTime))
      s += ', %.1f data per second' % wps
      if saveEvery is not None:
        s += ', %.1f minutes since saving' % ((time.time() - lastSaveTime)/60)
      s += '        \r'
      print s,
      sys.stdout.flush()
      if costEvery is None or (mb+1) % costEvery != 0:
        lastCostTime = time.time()
    
    #-------------------------------------------------------------------
    # Wrap everything up
    self.save(saveName, cost, test)
    print ''
    return cost
  
  #=====================================================================
  # Class-specific training function
  def __train__(self):
    pass
  
  #=====================================================================
  # Pickle the model
  def save(self, basename, cost, test=None):
    """"""
    
    self.dump(open(basename+'-state.pkl', 'w'))
    pkl.dump((cost,) + ((test,) if test is not None else tuple()), open(basename+'-cost.pkl', 'w'), protocol=pkl.HIGHEST_PROTOCOL)
    
  #=====================================================================
  # Dump the model
  def dump(self, f):
    """"""
    
    pkl.dump(self, f, protocol=pkl.HIGHEST_PROTOCOL)
    
  #=====================================================================
  # Load the model
  @classmethod
  def load(cls, f):
    """"""
    
    return pkl.load(f)
  
#***********************************************************************
# GloVe library (expects list of strings or list of tuples)
class GloVe(Opt):
  """"""
  
  #=====================================================================
  # Initialize the model
  def __init__(self, wsizes, **kwargs):
    """"""
    
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
    
    if 'wfunc' in kwargs:
      self.wfunc = kwargs['wfunc']
    else:
      self.wfunc = 'concatenate'
    
    if 'window' in kwargs:
      self.window = np.int32(kwargs['window'])
    else:
      self.window = np.int32(4)
    
    if 'xmax' in kwargs:
      self.xmax = np.float32(kwargs['xmax'])
    else:
      self.xmax = np.float32(100)
    
    if 'xmin' in kwargs:
      self.xmin = np.float32(kwargs['xmin'])
    else:
      self.xmin = np.float32(5)
    
    self._wsizes = wsizes
    if self.wfunc in ('concatenate', 'cat'):
      for wsize in self._wsizes[1:]
        assert self._wsizes[0] == wsize
    self.V_raw = list([Counter() for wsize in self._wsizes])
    self.V     = list([Counter() for wsize in self._wsizes])
    self._strs = list([dict() for wsize in self._wsizes])
    self._idxs = list([dict() for wsize in self._wsizes])
    self.X_raw = defaultdict(Counter)
    self.X     = defaultdict(Counter)
    
  
  #=====================================================================
  # Add a corpus (list of lists of tuples)
  def add_corpus(self, corpus):
    """"""
    
    for sent in corpus:
      self.add_sent(sent)
  
  #=====================================================================
  # Add a tokenized sentence to the statistics
  def add_sent(self, sent):
    """"""
    
    sent.insert(0, self.start_tup())
    sent.append(self.stop_tup())
    for i, wrd1 in enumerate(sent):
      wrd1 = self.recase(wrd1)
      for j, prt in enumerate(wrd1):
        self.V_raw[j][prt] += 1
      for j in xrange(i-self.window, (i+1)+self.window):
        if j < 0:
          self.X_raw[wrd1][self.start_tup()] += 1
        elif j >= len(sent):
          self.X_raw[wrd1][self.stop_tup()] += 1
        elif j != i:
          wrd2 = self.recase(sent[j])
          self.X_raw[wrd1][wrd2] += 1
  
  #=====================================================================
  # Build variables
  def build_vars(self):
    """"""
    
    #------------------------------------------------------------------
    # Aggregate the global statistics
    # Build V, strs, and idxs
    for i in xrange(len(self.wsizes())):
      self.V[i] = Counter()
      V_keys = set(filter(lambda k: self.V_raw[i][k] >= self.xmin, self.V[i].keys()))
      for wrd, cnt in self.V_raw[i].iteritems():
        if wrd in V_keys:
          self.V[word] = cnt
        else:
          self.V[self.unk()] += cnt
      for index, string in enumerate(self.V[i]):
        self._strs[i][index]  = string
        self._idxs[i][string] = index
    
    self.X = defaultdict(Counter)
    for wrd1, ctr in self.X_raw.iteritems():
      wrd1 = self.idxs(None,wrd1)
      for wrd2, cnt in ctr.iteritems():
        wrd2 = self.idxs(None,wrd2)
        self.X[wrd1][wrd2] += cnt
    
    #------------------------------------------------------------------
    # Build model params
    self.theta   = []
    self.gtheta  = []
    self.stheta  = []
    self.theta_x = []
    self.gstheta = []
    self.hmasks  = []
    
    self.theta_L  = []
    self.theta_Lb = []
    self.theta_L_tilde  = []
    self.theta_Lb_tilde = []
    
    #TODO here
    for i, keys, wsize in zip(range(len(self.keys())), self.keys(), self.wsizes()):
      if self.func_w == 'concatenate':
        self.theta_L.append(theano.shared(np.random.randn(len(keys), wsize).astype('float32'), name='L-%d' % (i+1)))
      else:
        self.theta_L.append(theano.shared(np.random.randn(len(keys), np.sum(self.wsizes())).astype('float32'), name='L-%d' % (i+1)))
      if self.func_w_tilde == 'concatenate':
        self.theta_L_tilde.append(theano.shared(np.zeros((len(keys), wsize), dtype='float32'), name='L_tilde-%d' % (i+1)))
      else:
        self.theta_L_tilde.append(theano.shared(np.zeros((len(keys), np.sum(self.wsizes())), dtype='float32'), name='L-%d' % (i+1)))
      self.theta_Lb.append(theano.shared(np.zeros(len(keys), dtype='float32'), name='Lb-%d' % (i+1)))
      self.theta_Lb_tilde.append(theano.shared(np.zeros(len(keys), dtype='float32'), name='Lb_tilde-%d' % (i+1)))
    self.hmasks.append(theano.shared(np.ones(np.sum(self.wsizes()), dtype='float32'), name='hmask'))
    
    self.stheta = self.theta_L + self.theta_Lb + self.theta_L_tilde + self.theta_Lb_tilde
    self.gstheta = list([theano.shared(np.zeros_like(stheta.get_value()), name='g'+stheta.name) for stheta in self.stheta])
    
    #------------------------------------------------------------------
    # Build the input/output variables
    self.x = T.imatrix('x')
    self.y = T.fmatrix('y')
    
    for i in xrange(len(self.stheta)):
      self.theta_x.append(self.stheta[i][self.x[:,i]])
    slen = len(self.theta_x)/4
    x = self.theta_x[:slen]
    xb = self.theta_x[slen:2*slen]
    x_tilde = self.theta_x[2*slen:3*slen]
    xb_tilde = self.theta_x[3*slen:]
    
    w  = splice_funx[self.func_w](x)*self.hmasks[0]
    w_tilde = (splice_funx[self.func_w_tilde](x_tilde)*self.hmasks[0]).T
    wb = T.sum(xb, axis=0)[None,:]
    wb_tilde = T.sum(xb_tilde, axis=0)
    
    #------------------------------------------------------------------
    # Build the cost variables
    yhat = T.dot(w, w_tilde) + wb + wb_tilde
    logy = T.log(self.y)
    y    = T.switch(T.isinf(logy), np.float32(0), logy)
    f    = reweight_glove(self.y)
    self.error = T.mean(T.sum(f * squared_difference(yhat, y), axis=0))
    
    self.complexity = theano.shared(np.float32(0))
    if self.L1reg > 0:
      self.complexity += self.L1reg*T.sum(T.abs_(x))
      self.complexity += self.L1reg*T.sum(T.abs_(x_tilde))
    if self.L2reg > 0:
      self.complexity += self.L2reg*T.sum(T.sqr(x))/np.float32(2)
      self.complexity += self.L2reg*T.sum(T.sqr(x_tilde))/np.float32(2)
    
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
  # Get the unk tokenize
  def start(self):
    """"""
    
    return self._start
  
  #=====================================================================
  # Get the unk tokenize
  def stop(self):
    """"""
    
    return self._stop
  
  #=====================================================================
  # Get the unk tokenize
  def unk(self):
    """"""
    
    return self._unk
  
  #=====================================================================
  # Get the start tuple
  def start_tup(self):
    """"""
    
    return (self.start(),)*len(self.wsize())
  
  #=====================================================================
  # Get the stop tuple
  def stop_tup(self):
    """"""
    
    return (self.stop(),)*len(self.wsize())
  
  #=====================================================================
  # Get the unk tuple
  def unk_tup(self):
    """"""
    
    return (self.unk(),)*len(self.wsize())
  
  #=====================================================================
  # Get the word size
  def wsizes(self, lib=None):
    """"""
    
    if lib is None:
      return self._wsizes
    else:
      return self._wsizes[lib]
  
  #=====================================================================
  # Get the strings of some indices 
  def strs(self, lib=None, wrd=None):
    """"""
    
    if lib is None and wrd is None:
      return self._strs
    elif wrd is None:
      return self._strs[lib]
    elif lib is None:
      wrd = self.recase(wrd)
      newple = []
      for prt, strs in zip(wrd, self._strs):
        newple.append(strs[prt])
      return tuple(newple)
    else:
      return self._strs[lib][wrd]
  
  #=====================================================================
  # Get the indices of some strings 
  def idxs(self, lib=None, wrd=None):
    """"""
    
    if lib is None and wrd is None:
      return self._idxs
    elif wrd is None:
      return self._idxs[lib]
    elif lib is None:
      newple = []
      for prt, idxs in zip(wrd, self._idxs):
          newple.append(idxs[prt])
        else:
          newple.append(idxs[self.unk()])
      return tuple(newple)
    else:
      return self._idxs[lib][wrd]
  
  #=====================================================================
  # Make a raw input token the right case
  def recase(self, wrd):
    """"""
    
    if self.lower:
      newple = []
      for prt in wrd:
        if not isinstance(prt, basestring):
          prt = str(prt)
        if prt not in (self._start, self._stop):
          newple.append(part.lower())
        else:
          newple.append(part)
      return tuple(newple)
    else:
      return tuple(word)
    
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
    return cost
  
  #=====================================================================
  # Train the model
  def __train__(self, dataset, dataidxs, momentizer, gradientizer, optimizer, nihilizer):
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
# Test the program
if __name__ == '__main__':
  """"""
  
  import os.path
  EPOCHS=300
  PATH='Glove Compartment'
  DIM=200
  glove = GloVe(DIM, xmin=3, xmax=100, window=8)
  glove.add_corpus('LotR.txt')
  mom, grad, opt, nihil = glove.Adam(eta_0=.01)
  name=os.path.join(PATH, 'glv-%d-'%DIM)
  glove.train([np.arange(len(glove.X)).astype('int32')], mom, grad, opt, nihil, saveName=name, costEvery=10, epochs=EPOCHS, batchSize=32)
  print glove.batch_cost([np.arange(len(glove.X)).astype('int32')])

  