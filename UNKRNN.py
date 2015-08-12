#!/usr/bin/env python
import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano.tensor.shared_randomstreams import RandomStreams
import cPickle as pkl
import multiprocessing as mp
from copy import deepcopy
import time
import sys
sys.setrecursionlimit(50000)
#TODO window of words
#TODO sparse library updates

#***********************************************************************
# Helper functions

#=======================================================================
# Creates the diagonal of a rectangular matrix
def diagonal(dims, dtype=np.float_):

  W = np.zeros(dims, dtype=dtype)
  m = (float(W.shape[0]-1) / float(W.shape[1]-1)) if float(W.shape[1]-1) > 0 else 1.
  for x in xrange(W.shape[1]):
    y = m*x
    W[np.floor(y),x] += (int(y)+1)-y
    W[min(np.ceil(y), W.shape[0]-1),x]  += y-int(y)
  return W

#=======================================================================
# Creates the tridiagonal of a rectangular matrix
def tridiagonal(dims, dtype=np.float_):

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
def triangle(dims, dtype=np.float_):

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
# Squared difference
def squared_difference(output, target):

  return T.pow(output-target, 2)/2

#=======================================================================
# Absolute difference
def absolute_difference(output, target):

  return T.abs_(output-target)

#=====================================================================
# Worker function
def func_worker(dataQueue, outQueue, func=None):

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
# A (multilayer) basic recurrent network 
class RNN:

  #=====================================================================
  # Initialize the model
  def __init__(self, libs, layers, *args, **kwargs):
    """
    Set up the chain RNN

    libs: [ (ndarray(|V|, dims[0]),
           {symbol : row index},
           {row index : symbol}), ...]
    layers: [ (ndarray(dims[0], dims[1]), # matrix
             ndarray(dims[1], dims[1]), # recur (can be None)
             ndarray(1, dims[1])),      # bias (can be None)
            ...]
    """
    
    #-------------------------------------------------------------------
    # Keyword arguments
    if 'dtype' in kwargs:
      self.dtype = kwargs['dtype']
    else:
      self.dtype = 'float32'

    if 'hfunc' in kwargs:
      self.hfunc = kwargs['hfunc']
    else:
      self.hfunc = 'tanh'

    if 'gfunc' in kwargs:
      self.gfunc = kwargs['gfunc']
    else:
      self.gfunc = 'sigmoid'

    if 'ofunc' in kwargs:
      self.ofunc = kwargs['ofunc']
    else:
      self.ofunc = 'softmax'

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

    if 'startSymb' in kwargs:
      self.startSymb = kwargs['startSymb']
    else:
      self.startSymb = '<S>'

    if 'stopSymb' in kwargs:
      self.stopSymb = kwargs['stopSymb']
    else:
      self.stopSymb = '</S>'

    if 'unkSymb' in kwargs:
      self.unkSymb = kwargs['unkSymb']
    else:
      self.unkSymb = '<UNK>'

    #-------------------------------------------------------------------
    # Process the libraries
    self.libs = [lib[1:] for lib in libs]
    for lib in self.libs:
      for k, v in lib[0].iteritems():
        lib[0][k] = np.int32(v)
      for v, k in lib[1].iteritems():
        lib[1][np.int32(v)] = k
      assert self.startSymb in lib[0]
      assert self.stopSymb in lib[0]
      assert self.unkSymb in lib[0]

    #-------------------------------------------------------------------
    # Process the matrix information
    # layerData: [(use hfunc or ofunc?), (recurrent?), (bias?)]
    self.layerData = np.array([[mat is not None for mat in layer] for layer in layers], dtype=np.int8)
    self.layerData[-1,0] = 0
    dims   = [layers[0][0].shape[0]] + [layers[i][0].shape[1] for i in xrange(len(self.layerData))]
    assert dims[0] == np.sum([lib[0].shape[1] for lib in libs])
    
    #-------------------------------------------------------------------
    # Bundle the model params
    self.Lparams = [theano.shared(lib[0], name='L-%d' % (i+1)) for i, lib in enumerate(libs)]
    self.Wparams = [theano.shared(np.concatenate([arg[i][0] for arg in [layers] + list(args) if arg[i][0] is not None], axis=1), name='W-%d' % (i+1)) for i in xrange(len(self.layerData))]
    self.Uparams = [theano.shared(np.concatenate([arg[i][1] for arg in [layers] + list(args) if arg[i][1] is not None], axis=1), name='U-%d' % (i+1)) if self.layerData[i,1] else None for i in xrange(len(self.layerData))]
    if self.layerData.shape[1] == 4:
      self.Vparams = [theano.shared(np.concatenate([arg[i][2] for arg in [layers] + list(args) if arg[i][2] is not None], axis=1), name='V-%d' % (i+1)) if self.layerData[i,1] else None for i in xrange(len(self.layerData))]
    self.bparams = [theano.shared(np.concatenate([arg[i][-1] for arg in [layers] + list(args) if arg[i][-1] is not None], axis=1), name='b-%d' % (i+1)) if self.layerData[i,-1] else None for i in xrange(len(self.layerData))]
    self.params  = filter(lambda x: x is not None,
        self.Wparams +
        self.Uparams +
       (self.Vparams if self.layerData.shape[1] == 4 else []) +
        self.bparams +
        self.Lparams)
    paramVars = filter(lambda x: x is not None,
        [T.matrix() for Wparam in self.Wparams if Wparam is not None] +
        [T.matrix() for Uparam in self.Uparams if Uparam is not None] +
       ([T.matrix() for Vparam in self.Vparams if Vparam is not None] if self.layerData.shape[1] == 4 else []) +
        [T.vector() for bparam in self.bparams if bparam is not None] +
        [T.matrix() for Lparam in self.Lparams if Lparam is not None])
    self.gparams = [theano.shared(np.zeros_like(param.get_value(), dtype=self.dtype)) for param in self.params]
    self.hmasks  = [theano.shared(np.ones(dims[i+1], dtype=self.dtype), name='hmask-%d' % (i+1)) for i in xrange(len(self.layerData))]
    self.h_0     = [theano.shared(np.zeros(dims[i+1], dtype=self.dtype), name='h_0-%d' % (i+1)) for i in xrange(len(self.layerData))]
    batchSize = T.scalar('batchSize')
    
    #-------------------------------------------------------------------
    # Build the input/output variables
    self.idxs    = T.imatrix('idxs')
    self.lparams = [Lparam[self.idxs[:,i]] for i, Lparam in enumerate(self.Lparams)]
    self.x       = T.concatenate(self.lparams, axis=1)
    self.y       = T.ivector('y')

    #-------------------------------------------------------------------
    # Build the recurrent function (only this should change for LSTMs)
    self.__build_recur__()

    #-------------------------------------------------------------------
    # Compute the activations/cost/grads
    self.activate = theano.function(
        inputs=[self.idxs],
        outputs=self.h[-1][-1],
        allow_input_downcast=True)

    self.loss = theano.function(
        inputs=[self.idxs,self.y],
        outputs=self.J,
        allow_input_downcast=True)

    self.grad = theano.function(
        inputs=[self.idxs, self.y],
        outputs=[self.J] + T.grad(self.J, self.params),
        allow_input_downcast=True)

    self.update_grad = theano.function(inputs=[batchSize]+paramVars,
        outputs=[],
        updates=[(gparam, gparam + paramVar/batchSize) for gparam, paramVar in zip(self.gparams, paramVars)],
        allow_input_downcast=True)

    self.reset_grad = theano.function(inputs=[],
        outputs=[],
        updates=[(gparam, 0*gparam) for gparam in self.gparams],
        allow_input_downcast=True)

  #=====================================================================
  # Returns the recurrent function
  def __build_recur__(self):

    #-------------------------------------------------------------------
    # Pull out the appropriate functions
    hfunc, gfunc, ofunc, Jfunc = self.getFunx()
  
    #-------------------------------------------------------------------
    # Build the scanning variables
    def recur(x_t, *h_tm1):
      h_t = [x_t]
      for h_tm1_l, layerDatum, Wparam, Uparam, bparam, hmask in zip(h_tm1, self.layerData, self.Wparams, self.Uparams, self.bparams, self.hmasks):
        h_t.append(
            (hfunc if layerDatum[0] else ofunc)(
              T.dot(h_t[-1], Wparam) + 
              (T.dot(h_tm1_l, Uparam) if layerDatum[1] else 0) +
              (bparam if layerDatum[-1] else 0))*hmask)
      return h_t[1:]
  
    self.h, _ = theano.scan(fn=recur,
        sequences=self.x,
        outputs_info=self.h_0,
        profile=False)

    self.J = T.mean(Jfunc(self.h[-1][-1],self.y)) +\
        (self.L1reg*T.sum([T.sum(T.abs_(Wparam))  for Wparam in self.Wparams]) if self.L1reg != 0 else 0) +\
        (self.L2reg*T.sum([T.sum(T.pow(Wparam,2)) for Wparam in self.Wparams]) if self.L2reg != 0 else 0)

  #=====================================================================
  # Converts a string of symbols to indices
  def s2i(self, symbs, reverse=False):
    s = np.empty((len(symbs) + 2, len(self.libs)), dtype=np.int32)
    symbs = symbs[::(-1 if reverse else 1)]
    for i in xrange(s.shape[0]):
      if   i == 0:
        for j in xrange(s.shape[1]):
          s[i,j] = np.int32(self.libs[j][0][self.startSymb])
      elif i == s.shape[0] - 1:
        for j in xrange(s.shape[1]):
          s[i,j] = np.int32(self.libs[j][0][self.stopSymb])
      else:
        for j in xrange(s.shape[1]):
          if symbs[i-1][j] in self.libs[j][0]:
            s[i,j] = np.int32(self.libs[j][0][symbs[i-1][j]])
          else:
            s[i,j] = np.int32(self.libs[j][0][self.unkSymb])
    return s

  #=====================================================================
  # Returns the squashing functions
  def getFunx(self):
    
    #-------------------------------------------------------------------
    # Hidden / gate / output functions
    # Hidden function
    if self.hfunc == 'relu':
      def hfunc (z): return T.switch(z > 0, z, 0)
    elif self.hfunc == 'resqrt':
      def hfunc (z): return T.switch(z > 0, T.sqrt(z+.25)-.5, 0)
    elif self.hfunc == 'rethu':
      def hfunc (z):
        tanh_z = T.tanh(z)
        return T.switch(tanh_z > 0, tanh_z, 0)
    elif self.hfunc == 'softplus':
      def hfunc (z): return T.nnet.softplus(z)
    elif self.hfunc == 'sqrt':
      def hfunc (z): return T.switch(z < 0, -T.sqrt(-z), T.sqrt(z))
    elif self.hfunc == 'tanh':
      def hfunc (z): return T.tanh(z)
    elif self.hfunc == 'recip':
      def hfunc (z): return T.switch(z==0, 0, T.pow(z,-1))
    elif self.hfunc == 'gauss':
      def hfunc (z): return T.exp(-T.pow(z,2)/2)
    elif self.hfunc == 'sigmoid':
      def hfunc (z): return T.nnet.sigmoid(z)
    elif self.hfunc == 'hard sigmoid':
      def hfunc (z): return T.nnet.hard_sigmoid(z)
    elif self.hfunc == 'fast sigmoid':
      def hfunc (z): return T.nnet.ultra_fast_sigmoid(z)
    elif self.hfunc == '2sigmoid':
      def hfunc (z): return T.nnet.sigmoid(2*z)
    else:
      raise ValueError('%s is not current a valid keyword for hfunc' % hfunc)

    # Gate function
    if self.gfunc == 'rethu':
      def gfunc (z):
        tanh_z = T.tanh(z)
        return T.switch(tanh_z > 0, tanh_z, 0)
    elif self.gfunc == 'gauss':
      def gfunc (z): return T.exp(-T.pow(z,2)/2)
    elif self.gfunc == 'sigmoid':
      def gfunc (z): return T.nnet.sigmoid(z)
    elif self.gfunc == 'hard sigmoid':
      def gfunc (z): return T.nnet.hard_sigmoid(z)
    elif self.gfunc == 'fast sigmoid':
      def gfunc (z): return T.nnet.ultra_fast_sigmoid(z)
    elif self.gfunc == '2sigmoid':
      def gfunc (z): return T.nnet.sigmoid(2*z)
    else:
      raise ValueError('%s is not current a valid keyword for gfunc' % hfunc)

    # Output function
    if self.ofunc == 'linear':
      def ofunc (z): return z
      def Jfunc (yhat, y): return squared_difference(yhat,y)
    elif self.hfunc == 'sqrt':
      def ofunc (z): return T.switch(z < 0, -T.sqrt(-z), T.sqrt(z))
      def Jfunc (yhat, y): return squared_difference(yhat,y)
    elif self.ofunc == 'sigmoid':
      def ofunc (z): return T.nnet.sigmoid(z)
      def Jfunc (yhat, y): return T.nnet.binary_crossentropy(yhat,y)
    elif self.ofunc == 'hard sigmoid':
      def ofunc (z): return T.nnet.hard_sigmoid(z)
      def Jfunc (yhat, y): return T.nnet.binary_crossentropy(yhat,y)
    elif self.ofunc == 'fast sigmoid':
      def ofunc (z): return T.nnet.ultra_fast_sigmoid(z)
      def Jfunc (yhat, y): return T.nnet.binary_crossentropy(yhat,y)
    elif self.ofunc == '2sigmoid':
      def ofunc (z): return T.nnet.sigmoid(2*z)
      def Jfunc (yhat, y): return T.nnet.binary_crossentropy(yhat,y)
    elif self.ofunc == 'rethu':
      def ofunc (z):
        tanh_z = T.tanh(z)
        return T.switch(tanh_z > 0, tanh_z, 0)
      def Jfunc (yhat, y): return T.nnet.binary_crossentropy(yhat,y)
    elif self.ofunc == 'gauss':
      def ofunc (z): return T.exp(-T.pow(z,2)/2)
      def Jfunc (yhat, y): return T.nnet.binary_crossentropy(yhat,y)
    elif self.ofunc == 'softmax':
      def ofunc (z):
        #return T.nnet.softmax(z) # Doesn't work =(
        e_z = T.exp(z - z.max(axis=0, keepdims=True))
        return e_z / e_z.sum(axis=0, keepdims=True)
      def Jfunc (yhat, y):
        return T.nnet.categorical_crossentropy(yhat,y) 
    else:
      raise ValueError('%s is not current a valid keyword for ofunc' % ofunc)

    return (hfunc, gfunc, ofunc, Jfunc)

  #=====================================================================
  # Batch cost
  def cost(self, dataset, workers=2):

    cost = 0
    dataQueue = mp.Queue()
    costQueue = mp.Queue()
    processes = []
    for datum in dataset:
      dataQueue.put(datum)
    for worker in xrange(workers):
      dataQueue.put('STOP')
    for worker in xrange(workers):
      process = mp.Process(target=func_worker, args=(dataQueue, costQueue, self.loss))
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
          process = mp.Process(target=func_worker, args=(dataQueue, gradQueue, self.grad))
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
      
  #=====================================================================
  # Run SGD (with NAG)
  def SGD(self, eta_0=.01, T_eta=1, mu_max=.95, T_mu=1, dropout=1., anneal=0, accel=0):

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
# A (multilayer) gated recurrent network
class GRU (RNN):

  #=====================================================================
  # Returns the recurrent function
  def __build_recur__(self):

    #-------------------------------------------------------------------
    # Pull out the appropriate functions
    hfunc, gfunc, ofunc, Jfunc = self.getFunx()

    #-------------------------------------------------------------------
    # Build the scanning variables 
    def recur(x_t, *h_tm1):
      h_t = [x_t]
      for h_tm1_l, layerDatum, Wparam, Uparam, bparam, hmask in zip(h_tm1, self.layerData, self.Wparams, self.Uparams, self.bparams, self.hmasks):
        if layerDatum[1]:

          sliceLen = Wparam.shape[1]/3
          czr_W = T.dot(h_t[-1], Wparam)
          zr_U = T.dot(h_tm1_l, Uparam[:,sliceLen:])

          z = gfunc(
              czr_W[sliceLen:2*sliceLen] +
              zr_U[:sliceLen] +
              (bparam[sliceLen:2*sliceLen] if layerDatum[-1] else 0))

          r = gfunc(
              czr_W[2*sliceLen:] +
              zr_U[sliceLen:] +
              (bparam[2*sliceLen:] if layerDatum[-1] else 0))

          c = (hfunc if layerDatum[0] else ofunc)(
              czr_W[:sliceLen] + 
              T.dot(r*h_tm1_l, Uparam[:,:sliceLen]) +
              (bparam[:sliceLen] if layerDatum[-1] else 0))
              
          h_t.append(((1-z)*h_tm1_l + z*c)*hmask)
        else:
          h_t.append(
              (hfunc if layerDatum[0] else ofunc)(
                T.dot(h_t[-1], Wparam) + 
                (bparam if layerDatum[-1] else 0)) * hmask)
      return h_t[1:]

    self.h, _ = theano.scan(fn=recur,
        sequences=self.x,
        outputs_info=self.h_0,
        profile=False)

    self.J = T.mean(Jfunc(self.h[-1][-1],self.y)) +\
        (self.L1reg*T.sum([T.sum(T.abs_(Wparam))  for Wparam in self.Wparams]) if self.L1reg != 0 else 0) +\
        (self.L2reg*T.sum([T.sum(T.pow(Wparam,2)) for Wparam in self.Wparams]) if self.L2reg != 0 else 0)

#***********************************************************************
# A (multilayer) fast gated recurrent network
class FastGRU (RNN):

  #=====================================================================
  # Returns the recurrent function
  def __build_recur__(self):

    #-------------------------------------------------------------------
    # Pull out the appropriate functions
    hfunc, gfunc, ofunc, Jfunc = self.getFunx()

    #-------------------------------------------------------------------
    # Build the scanning variables 
    def recur(x_t, *h_tm1):
      h_t = [x_t]
      for h_tm1_l, layerDatum, Wparam, Uparam, bparam, hmask in zip(h_tm1, self.layerData, self.Wparams, self.Uparams, self.bparams, self.hmasks):
        if layerDatum[1]:
          sliceLen = Wparam.shape[1]/3
          Czr = T.dot(h_t[-1], Wparam) + \
              T.dot(h_tm1_l, Uparam) + \
              (bparam if layerDatum[-1] else 0)

          C = (hfunc if layerDatum[0] else ofunc)(Czr[:sliceLen])
          z = gfunc(Czr[sliceLen:2*sliceLen])
          r = gfunc(Czr[2*sliceLen:])

          h_t.append(((1-z)*r*h_tm1_l + z*C)*hmask)
        else:
          h_t.append(
              (hfunc if layerDatum[0] else ofunc)(
                T.dot(h_t[-1], Wparam) + \
                    (bparam if layerDatum[-1] else 0)) * hmask)
      return h_t[1:]

    self.h, _ = theano.scan(fn=recur,
        sequences=self.x,
        outputs_info=self.h_0,
        profile=False)

    self.J = T.mean(Jfunc(self.h[-1][-1],self.y)) +\
        (self.L1reg*T.sum([T.sum(T.abs_(Wparam))  for Wparam in self.Wparams]) if self.L1reg != 0 else 0) +\
        (self.L2reg*T.sum([T.sum(T.pow(Wparam,2)) for Wparam in self.Wparams]) if self.L2reg != 0 else 0)

#***********************************************************************
# A (multilayer) long short-term memory network (Graves 2013)
class LSTM (RNN):

  #=====================================================================
  # Returns the recurrent function
  def __build_recur__(self):

    #-------------------------------------------------------------------
    # Pull out the appropriate functions
    hfunc, gfunc, ofunc, Jfunc = self.getFunx()

    #-------------------------------------------------------------------
    # Build the scanning variables 
    def recur(x_t, *Ch_tm1):
      C_tm1  = Ch_tm1[:len(Ch_tm1)/2]
      h_tm1  = Ch_tm1[len(Ch_tm1)/2:]
      h_t    = [x_t]
      C_t    = []
      for h_tm1_l, C_tm1_l, layerDatum, Wparam, Uparam, Vparam, bparam, hmask in zip(h_tm1, C_tm1, self.layerData, self.Wparams, self.Uparams, self.Vparams, self.bparams, self.hmasks):
        if layerDatum[1]:

          sliceLen = Wparam.shape[1]/4

          cifo_UW = T.dot(h_t[-1], Wparam) + \
              T.dot(h_tm1_l, Uparam) + \
              (bparam if layerDatum[-1] else 0)
          if_V = T.dot(C_tm1_l, Vparam[:,:2*sliceLen])

          c = (hfunc if layerDatum[0] else ofunc)(cifo_UW[:sliceLen])
          i = gfunc(cifo_UW[1*sliceLen:2*sliceLen] + if_V[:sliceLen])
          f = gfunc(cifo_UW[2*sliceLen:3*sliceLen] + if_V[sliceLen:])
          C_t.append(i*c + f*C_tm1_l)
          o = gfunc(
              cifo_UW[3*sliceLen:] +
              T.dot(C_t[-1], Vparam[:,2*sliceLen:]))

          h_t.append(o*(hfunc if layerDatum[0] else ofunc)(C_t[-1])*hmask)
        else:
          C_t.append(C_tm1_l)
          h_t.append(
              (hfunc if layerDatum[0] else ofunc)(
                T.dot(h_t[-1], Wparam) +
                (bparam if layerDatum[-1] else 0)) * hmask)
      return C_t + h_t[1:]

    self.h, _ = theano.scan(fn=recur,
        sequences=self.x,
        outputs_info=self.h_0*2,
        profile=False)

    self.J = T.mean(Jfunc(self.h[-1][-1],self.y)) +\
        (self.L1reg*T.sum([T.sum(T.abs_(Wparam))  for Wparam in self.Wparams]) if self.L1reg != 0 else 0) +\
        (self.L2reg*T.sum([T.sum(T.pow(Wparam,2)) for Wparam in self.Wparams]) if self.L2reg != 0 else 0)

#***********************************************************************
# A (multilayer) long short-term memory network without peepholes
class FastLSTM (RNN):

  #=====================================================================
  # Returns the recurrent function
  def __build_recur__(self):

    #-------------------------------------------------------------------
    # Pull out the appropriate functions
    hfunc, gfunc, ofunc, Jfunc = self.getFunx()

    #-------------------------------------------------------------------
    # Build the scanning variables 
    def recur(x_t, *Ch_tm1):
      C_tm1  = Ch_tm1[:len(Ch_tm1)/2]
      h_tm1  = Ch_tm1[len(Ch_tm1)/2:]
      h_t    = [x_t]
      C_t    = []
      for h_tm1_l, C_tm1_l, layerDatum, Wparam, Uparam, bparam, hmask in zip(h_tm1, C_tm1, self.layerData, self.Wparams, self.Uparams, self.bparams, self.hmasks):
        if layerDatum[1]:
          Cifo = T.dot(h_t[-1], Wparam) + \
              T.dot(h_tm1_l, Uparam) + \
              (bparam if layerDatum[-1] else 0)

          sliceLen = Cifo.shape[0]/4
          C = (hfunc if layerDatum[0] else ofunc)(Cifo[:sliceLen])
          i = gfunc(Cifo[1*sliceLen:2*sliceLen])
          f = gfunc(Cifo[2*sliceLen:3*sliceLen])
          o = gfunc(Cifo[3*sliceLen:])

          C_t.append(i*C + f*C_tm1_l)
          h_t.append(o*(hfunc if layerDatum[0] else ofunc)(C_t[-1])*hmask)
        else:
          C_t.append(C_tm1_l)
          h_t.append(
              (hfunc if layerDatum[0] else ofunc)(
                T.dot(h_t[-1], Wparam) + \
                    (bparam if layerDatum[-1] else 0)) * hmask)
      return C_t + h_t[1:]

    self.h, _ = theano.scan(fn=recur,
        sequences=self.x,
        outputs_info=self.h_0*2,
        profile=False)

    self.J = T.mean(Jfunc(self.h[-1][-1],self.y)) +\
        (self.L1reg*T.sum([T.sum(T.abs_(Wparam))  for Wparam in self.Wparams]) if self.L1reg != 0 else 0) +\
        (self.L2reg*T.sum([T.sum(T.pow(Wparam,2)) for Wparam in self.Wparams]) if self.L2reg != 0 else 0)

#***********************************************************************
if __name__ == '__main__':

  import sys
  import os.path
  np.random.seed(248832)

  glove = pkl.load(open('glove.6B.10k.50d-real.pkl'))
  vocab = set()
  [vocab.update(list(key)) for key in glove[1].keys()]
  vocab.update(['<UNK>', '<S>', '</S>'])
  vocab = list(vocab)

  insize = 50
  libsize = len(vocab)
  large = False
  model = 'RNN'
  opt   = 'SGD'
  hfunc = 'tanh'
  gfunc = '2sigmoid'
  epochs = 5
  workers = 2
  hidsize = 200 
  outsize = 50
  path = '.'
  flags = sys.argv[1:]
  while len(flags) > 0:
    flag = flags.pop(0)
    if flag == '-m' or flag == '--model':
      model = flags.pop(0)
    elif flag == '-2' or flag == '--two-layer':
      large = True
    elif flag == '-g' or flag == '--gate':
      gfunc = flags.pop(0)
    elif flag == '-f' or flag == '--function':
      hfunc = flags.pop(0)
    elif flag == '-o' or flag == '--optimizer':
      opt = flags.pop(0)
    elif flag == '-t' or flag == '--times':
      epochs = int(flags.pop(0))
    elif flag == '-w' or flag == '--workers':
      workers = int(flags.pop(0))
    elif flag == '-i' or flag == '--input-size':
      insize = int(flags.pop(0))
    elif flag == '-h' or flag == '--hidden-size':
      hidsize = int(flags.pop(0))
    elif flag == '-p' or flag == '--path':
      path = flags.pop(0)
    else:
      print 'Flag %s not recognized' % flag

  mat = matwizard(libsize, insize, normalize=True)
  s2i = {v: k for k, v in enumerate(vocab)}
  i2s = {k: v for k, v in enumerate(vocab)}
  lib = (mat, s2i, i2s)

  W1 = (matwizard(insize,  hidsize, shape='1', dist='uniform', scale='in+out'),
        matwizard(hidsize, hidsize, shape='I'),
        matwizard(hidsize, shape='0'))

  W2 = (matwizard(hidsize, hidsize, shape='1', dist='uniform', scale='in+out'),
        matwizard(hidsize, hidsize, shape='I'),
        matwizard(hidsize, shape='0'))

  W3 = (matwizard(hidsize, outsize, shape='1', dist='uniform', scale='in+out'),
        matwizard(outsize, outsize, shape='I'),
        matwizard(outsize, shape='0'))

  I1 = (matwizard(insize,  hidsize, shape='1', dist='uniform', scale='in+out'),
        matwizard(hidsize, hidsize, shape='I'),
        matwizard(hidsize, shape='0'))

  I2 = (matwizard(hidsize, hidsize, shape='1', dist='uniform', scale='in+out'),
        matwizard(hidsize, hidsize, shape='I'),
        matwizard(hidsize, shape='0'))

  I3 = (matwizard(hidsize, outsize, shape='1', dist='uniform', scale='in+out'),
        matwizard(outsize, outsize, shape='I'),
        matwizard(outsize, shape='0'))

  F1 = (matwizard(insize,  hidsize, shape='1', dist='uniform', scale='in+out'),
        matwizard(hidsize, hidsize, shape='I'),
        matwizard(hidsize, shape='0'))

  F2 = (matwizard(hidsize, hidsize, shape='1', dist='uniform', scale='in+out'),
        matwizard(hidsize, hidsize, shape='I'),
        matwizard(hidsize, shape='0'))

  F3 = (matwizard(hidsize, outsize, shape='1', dist='uniform', scale='in+out'),
        matwizard(outsize, outsize, shape='I'),
        matwizard(outsize, shape='0'))
        
  O1 = (matwizard(insize,  hidsize, shape='1', dist='uniform', scale='in+out'),
        matwizard(hidsize, hidsize, shape='I'),
        matwizard(hidsize, shape='0'))

  O2 = (matwizard(hidsize, hidsize, shape='1', dist='uniform', scale='in+out'),
        matwizard(hidsize, hidsize, shape='I'),
        matwizard(hidsize, shape='0'))

  O3 = (matwizard(hidsize, outsize, shape='1', dist='uniform', scale='in+out'),
        matwizard(outsize, outsize, shape='I'),
        matwizard(outsize, shape='0'))

######
  U1 = (matwizard(insize,  hidsize, shape='I'),
        matwizard(hidsize, hidsize, shape='I'),
        None,
        matwizard(hidsize, shape='0'))

  U2 = (matwizard(hidsize, hidsize, shape='I'),
        matwizard(hidsize, hidsize, shape='I'),
        matwizard(hidsize, hidsize, shape='I'),
        matwizard(hidsize, shape='0'))

  U3 = (matwizard(hidsize, outsize, shape='1'),
        None,
        None,
        matwizard(outsize, shape='0'))

  J1 = (matwizard(insize,  hidsize, shape='I'),
        matwizard(hidsize, hidsize, shape='I'),
        matwizard(hidsize, hidsize, shape='I'),
        matwizard(hidsize, shape='0'))

  J2 = (matwizard(hidsize, hidsize, shape='I'),
        matwizard(hidsize, hidsize, shape='I'),
        matwizard(hidsize, hidsize, shape='I'),
        matwizard(hidsize, shape='0'))

  J3 = (None, None, None, None)

  G1 = (matwizard(insize,  hidsize, shape='I'),
        matwizard(hidsize, hidsize, shape='I'),
        matwizard(hidsize, hidsize, shape='I'),
        matwizard(hidsize, shape='0'))

  G2 = (matwizard(hidsize, hidsize, shape='I'),
        matwizard(hidsize, hidsize, shape='I'),
        matwizard(hidsize, hidsize, shape='I'),
        matwizard(hidsize, shape='0'))

  G3 = (None, None, None, None)

  P1 = (matwizard(insize,  hidsize, shape='I'),
        matwizard(hidsize, hidsize, shape='I'),
        matwizard(hidsize, hidsize, shape='I'),
        matwizard(hidsize, shape='0'))

  P2 = (matwizard(hidsize, hidsize, shape='I'),
        matwizard(hidsize, hidsize, shape='I'),
        matwizard(hidsize, hidsize, shape='I'),
        matwizard(hidsize, shape='0'))
  P3 = (None, None, None, None)

  mats = [[W1, W2, W3], [I1, I2, I3], [F1, F2, F3], [O1, O2, O3]]
  LSTMmats= [[U1, U2, U3], [J1, J2, J3], [G1, G2, G3], [P1, P2, P3]]
  if model == 'RNN':
    NN = RNN
    mats = mats[:1]
  elif model == 'SRU':
    NN = SRU
    mats = mats[:3]
  elif model == 'SimpleSRU':
    NN = SimpleSRU
    mats = mats[:2]
  elif model == 'GRU':
    NN = GRU
    mats = mats[:3]
  elif model == 'FastGRU':
    NN = FastGRU
    mats = mats[:3]
  elif model == 'SimpleGRU':
    NN = SimpleGRU
    mats = mats[:3]
  elif model == 'LSTM':
    NN = LSTM
    mats = LSTMmats
  elif model == 'FastLSTM':
    NN = FastLSTM
  else:
    print 'Model not recognized, using RNN'
    NN = RNN
    mats = mats[:1]

  if not large:
    mats = [[mat[0], mat[2]] for mat in mats]

  path = os.path.join(path, ('Small' if not large else '')+model, hfunc, opt)
  if not os.path.exists(path):
    os.makedirs(path)
  print 'Saving to path %s' % path

  NN = NN([lib], *mats, ofunc='linear', hfunc=hfunc, gfunc=gfunc)

  data = []
  for word in glove[1].keys():
    data.append((np.array([s2i['<S>']]+[s2i[char] for char in word]+[s2i['</S>']], dtype=np.int32)[:,None], glove[0][glove[1][word]]))
  train_data = data[:int(.8*len(data))]
  dev_data   = data[int(.8*len(data)):int(.9*len(data))]
  test_data  = data[int(.9*len(data)):]

  if opt == 'SGD':
    opt = NN.SGD()
  elif opt == 'AdaGrad': 
    opt = NN.AdaGrad()
  elif opt == 'RMSProp':
    opt = NN.RMSProp(eta_0=.005, rho_0=.95, mu_max=.999, T_mu=32, accel=1.)
  elif opt == 'AdaDelta':
    opt = NN.AdaDelta()
  elif opt == 'Adam':
    opt = NN.Adam()
  else:
    'Optimizer not recognized, using SGD'
    opt = NN.SGD()

  parentPipe, childPipe = mp.Pipe()
  process = mp.Process(target=NN.train, args=(train_data, opt), kwargs={'savePipe': childPipe, 'batchSize':120, 'costEvery': 10, 'workers':workers, 'epochs': epochs, 'testset': dev_data})
  process.start()
  i = 0
  msg = 'START'
  while msg != 'STOP':
    msg = parentPipe.recv() #Always pickles at the end
    if msg != 'START':
      i+=1
      pkl.dump(msg, open(os.path.join(path, 'cost-%02d.pkl' % i), 'w'))
      pkl.dump(NN, open(os.path.join(path, 'state-%02d.pkl' % i), 'w'), protocol=pkl.HIGHEST_PROTOCOL)
 
  process.join()
