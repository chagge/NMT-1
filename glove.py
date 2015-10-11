#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Matrix operations
import numpy as np
np.random.seed(9841265)
import theano
#theano.config.optimizer='None'
#theano.config.exception_verbostiy='high'
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams 
srng = RandomStreams(9841265)
from funx import splice_funx, sum_squared_error, sum_absolute_error, squared_difference, absolute_difference, reweight_glove
# Data structures
from collections import defaultdict, Counter
import codecs
# Pickling
import cPickle as pkl
import sys
sys.setrecursionlimit(50000)
# Utilities
import time
import warnings
warnings.simplefilter("error")

CLIPSIZE = 32

# TODO methods for probing the matrices, i.e. neighbors, t-sne, semantic orientation, word analogies
# TODO Word2Vec
# TODO predict left and right contexts separately
# TODO do matrices

#***********************************************************************
# An optimizable library class
class LOpt():
  """"""
  
  def build_optimizer(self, delta_moment=False, max_force=False, eta=1e-3, a_eta=0, b_eta=1, rho1=.49, rho2=.95, a_rho=5, b_rho=.5, epsilon=1e-4, dropout=1.):
    """"""
    
    #-------------------------------------------------------------------
    # Cast everything as float32
    eta  = np.float32(eta)
    rho1 = np.float32(rho1)
    rho2 = np.float32(rho2)
    a_eta = np.float32(a_eta)
    b_eta = np.float32(b_eta)
    a_rho = np.float32(a_rho)
    b_rho = np.float32(b_rho)
    epsilon = np.float32(epsilon)
    
    #-------------------------------------------------------------------
    # Set up the updates
    grupdates = []
    pupdates  = []
    nupdates  = []
    
    #-------------------------------------------------------------------
    # Set up a variable to keep track of the minibatch size
    ndata = theano.shared(np.float32(0), name='ndata')
    grupdates.append((ndata, ndata+np.float32(1)))
    nupdates.append((ndata, np.float32(0)))
    
    #-------------------------------------------------------------------
    # Compute the dropout 
    if dropout < 1:
      givens = [(self.hmask, srng.binomial(self.hmask.shape, 1, dropout, dtype='float32'))]
    else:
      givens = []
    
    #-------------------------------------------------------------------
    # Sparse parameters
    group = [[self.x[i] for i in xrange(len(self.wsizes()))] + [self.x_tilde],
             zip(self.theta_L + self.theta_L_tilde, self.theta_Lb + self.theta_Lb_tilde),
             zip(self.gtheta_L + self.gtheta_L_tilde, self.gtheta_Lb + self.gtheta_Lb_tilde),
             zip(self.gstheta_x + self.gstheta_x_tilde, self.gstheta_xb + self.gstheta_xb_tilde)]
    gidxs = []
    for x, Ls, gLs, gxs in zip(*group):
      gidxs.append(T.ivector('gidxs-%s' % Ls[0].name))
      
      tau = theano.shared(np.zeros(Ls[0].get_value().shape[0], dtype='float32'), name='tau-%s' % Ls[0].name)
      pupdates.append((tau, T.inc_subtensor(tau[gidxs[-1]], np.float32(1))))
      
      for L, gL, gx in zip(Ls, gLs, gxs):
        Lshape = L.get_value().shape
        if len(Lshape) > 1:
          tau_t = tau[gidxs[-1]][:,None]
        else:
          tau_t = tau[gidxs[-1]]
        if a_eta > 0:
          eta_t = eta*T.pow(tau_t/a_eta + np.float32(1), -b_eta)
        else:
          eta_t = eta
        if a_rho > 0:
          rho1_t = np.float32(1)-rho1*T.pow(tau_t/a_rho + np.float32(1), -b_rho)
          rho2_t = np.float32(1)-rho2*T.pow(tau_t/a_rho + np.float32(1), -b_rho)
        else:
          rho1_t = rho1
          rho2_t = rho2
        
        g_t = gL[gidxs[-1]] / ndata
        grupdates.append((gL, T.inc_subtensor(gL[x], T.clip(gx, -CLIPSIZE, CLIPSIZE))))
        if rho2 > 0:
          vL = theano.shared(np.zeros(Lshape, dtype='float32'), name='v%s' % L.name)
          if max_force:
            v_t = T.maximum(rho2_t*vL[gidxs[-1]], T.abs_(g_t))
            v_t_hat = v_t + epsilon
          else:
            v_t = rho2_t*vL[gidxs[-1]] + (np.float32(1)-rho2_t)*T.sqr(g_t)
            if a_rho == 0:
              v_t_hat = v_t / (1-rho2_t**(tau_t+1))
            else:
              v_t_hat = v_t
            v_t_hat = T.sqrt(v_t_hat) + epsilon
          pupdates.append((vL, T.set_subtensor(vL[gidxs[-1]], v_t)))
        else:
          v_t_hat = np.float32(1)
        if rho1 > 0:
          mL = theano.shared(np.zeros(Lshape, dtype='float32'), name='m%s' % L.name)
          if delta_moment:
            if a_rho == 0:
              d_tm1_hat = T.switch(tau_t == 0, np.float32(0), mL[gidxs[-1]] / (np.float32(1)-rho1_t**(tau_t)))
            else:
              d_tm1_hat = mL[gidxs[-1]]
            d_t = -(T.sqrt(d_tm1_hat) + eta_t*epsilon) / v_t_hat * g_t
            m_t = rho1_t*mL[gidxs[-1]] + (np.float32(1)-rho1_t)*T.sqr(d_t)
          else:
            m_t = rho1_t*mL[gidxs[-1]] + (np.float32(1)-rho1_t)*g_t
            if a_rho == 0:
              m_t_hat = m_t / (1-rho1_t**(tau_t+1))
            else:
              m_t_hat = m_t
            d_t = -eta_t*m_t_hat / v_t_hat
          pupdates.append((mL, T.set_subtensor(mL[gidxs[-1]], m_t)))
        else:
          d_t = -eta_t*g_t / v_t_hat
        pupdates.append((L, T.inc_subtensor(L[gidxs[-1]], d_t)))
        nupdates.append((gL, T.set_subtensor(gL[gidxs[-1]], np.float32(0))))
     
    #-------------------------------------------------------------------
    # Compile the functions
    gradientizer = theano.function(
      inputs=[self.x, self.x_tilde, self.y],
      outputs=self.cost,
      givens=givens,
      updates=grupdates,
      name='gradientizer')
    
    optimizer = theano.function(
      inputs=gidxs,
      updates=pupdates,
      name='optimizer')
    
    nihilizer = theano.function(
      inputs=gidxs,
      updates=nupdates,
      name='nihilizer')
      
    return gradientizer, optimizer, nihilizer
  
  #=====================================================================
  # Train the model
  def train(self, gradientizer, optimizer, nihilizer, batch_size=1, split=1, epochs=15, print_every=None, cost_every=1, save_every=None, save_name='None'):
    """"""
    
    #-------------------------------------------------------------------
    # Split the dataset
    np.random.shuffle(self.dataset)
    trainset = self.dataset[~(split*len(self.dataset)):]
    testset  = self.dataset[:~(split*len(self.dataset))]
    
    #-------------------------------------------------------------------
    # Saving and printing
    train_cost        = []
    test_cost         = []
    recent_train_cost = []
    recent_test_cost  = []
    
    if cost_every is not None:
      train_cost.append(self.batch_cost(trainset))
      if testset:
        test_cost.append(self.batch_cost(testset))
    else:
      train_cost.append(np.nan)
      if testset:
        test_cost.append(np.nan)
    
    if batch_size < 1:
      batch_size = int(len(trainset)*batch_size)
    
    epochd     = str(int(np.log10(epochs))+1)
    minibatchd = str(int(max(0, np.log10(len(trainset)/batch_size)))+1)
    wps = np.nan
    nsaves = 0
    
    s = ''
    s += ('Minibatch %0'+epochd+'d-%0'+minibatchd+'d') % (0, 0)
    s += ': %.3f train cost' % train_cost[-1]
    if testset:
      s += ', %.3f test cost' % test_cost[-1]
    s += ', %.1f data per second' % wps
    if save_every is not None and save_every % total_mb == 0:
      self.save(save_name+'%02d'%nsaves, cost, gut=False)
      last_save = time.time()
      nsaves += 1
      s += ', %.1f minutes since saving' % ((time.time()-last_save)/60)
    s += '        '
    print s
    cost_time = time.time()
    recent_train_cost.append(0)
    print_size = 0
    
    #-------------------------------------------------------------------
    # Process the minibatch
    try:
      mb = -1
      for t in xrange(epochs):
        np.random.shuffle(trainset)
        for mb in xrange(len(trainset)/batch_size):
          recent_train_cost[-1] += self.__train__(trainset[mb*batch_size:(mb+1)*batch_size], gradientizer, optimizer, nihilizer)
          print_size += batch_size
          
          #---------------------------------------------------------------
          # More saving and printing
          if print_every is not None and (mb+1) % print_every == 0:
            
            recent_train_cost[-1] /= print_size
            if len(recent_train_cost) > 1:
              recent_train_cost[-1] = .67*recent_train_cost[-2] + .33*recent_train_cost[-1]
            
            if testset:
              recent_test_cost.append(self.batch_cost(testset))
           
            if np.isnan(wps):
              wps = print_size / (time.time() - cost_time)
            else:
              wps = .67*wps + .33*print_size / (time.time() - cost_time)
            
            if save_every is not None and (save_every % (t*mb)) == 0:
              self.save(save_name+'%02d'%nsaves, cost, gut=True)
              last_save = time.time()
              nsaves += 1
            
            s = ''
            s += ('Minibatch %0'+epochd+'d-%0'+minibatchd+'d') % (t+1, mb+1)
            s += ': %.3f train cost' % recent_train_cost[-1]
            if testset:
              s += ', %.3f test cost' % recent_test_cost[-1]
            s += ', %.1f data per second' % wps
            if save_every is not None:
              s += ', %.1f minutes since saving' % ((time.time()-last_save)/60)
            s += '        \r'
            print s,
            sys.stdout.flush()
            cost_time = time.time()
            recent_train_cost.append(0)
            print_size = 0
            
        #-----------------------------------------------------------------
        # Finish off the data
        recent_train_cost[-1] += self.__train__(trainset[(mb+1)*batch_size:], gradientizer, optimizer, nihilizer)
        print_size += len(trainset[(mb+1)*batch_size:])
        if print_every is not None:
          
          recent_train_cost[-1] /= print_size
          if len(recent_train_cost) > 1:
            recent_train_cost[-1] = .67*recent_train_cost[-2] + .33*recent_train_cost[-1]
          
          if testset:
            recent_test_cost.append(self.batch_cost(testset))
          
          if np.isnan(wps):
            wps = (print_size / (time.time()-cost_time))
          else:
            wps = .67*wps + .33*(print_size / (time.time()-cost_time))
          
          if save_every is not None and (save_every % (t*mb)) == 0:
            self.save(save_name+'%02d'%nsaves, cost, gut=True)
            last_save = time.time()
            nsaves += 1
          
          s = ''
          s += ('Minibatch %0'+epochd+'d-%0'+minibatchd+'d') % (t+1, mb+2)
          s += ': %.3f train cost' % recent_train_cost[-1]
          if testset:
            s += ', %.3f test cost' % recent_test_cost[-1]
          s += ', %.1f data per second' % wps
          if save_every is not None:
            s += ', %.1f minutes since saving' % ((time.time()-last_save)/60)
          s += '        \r'
          print s,
          sys.stdout.flush()
        
        if cost_every is not None:
          train_cost.append(self.batch_cost(trainset))
          if testset:
            test_cost.append(self.batch_cost(testset))
        
        if print_every is None:
          wps = print_size / (time.time() - cost_time)
        
        s = ''
        s += ('Minibatch %0'+epochd+'d-%0'+minibatchd+'d') % (t+1, mb+2)
        s += ': %.3f train cost' % train_cost[-1]
        if testset:
          s += ', %.3f test cost' % test_cost[-1]
        s += ', %.1f data per second' % wps
        if save_every is not None:
          s += ', %.1f minutes since saving' % ((time.time()-last_save)/60)
        s += '        '
        print s
        cost_time = time.time()
        recent_train_cost.append(0)
    
    except KeyboardInterrupt:
      print 'Training interrupted. Press <enter> to save or <ctrl-c> to cancel.                '
      raw_input()
    
    #-------------------------------------------------------------------
    # Wrap everything up
    self.save(save_name, (recent_train_cost, recent_test_cost, train_cost, test_cost), gut=False)
    print ''
    return (recent_train_cost, recent_test_cost, train_cost, test_cost)
  
  #=====================================================================
  # Class-specific training function
  def __train__(self):
    pass
  
  #=====================================================================
  # Pickle the model
  def save(self, basename, cost, gut=False):
    """"""
    
    self.dump(open(basename+'-state.pkl', 'w'), gut=gut)
    pkl.dump(cost, open(basename+'-cost.pkl', 'w'), protocol=pkl.HIGHEST_PROTOCOL)
  
  #=====================================================================
  # Dump the model
  def dump(self, f, gut=False):
    """"""
    
    if gut:
      V_raw       = self.V_raw
      V_tilde_raw = self.V_tilde_raw
      C_raw       = self.C_raw
      V           = self.V
      V_tilde     = self.V_tilde
      C           = self.C
      dataset     = self.dataset
      self.V_raw       = None
      self.V_tilde_raw = None
      self.C_raw       = None
      self.V           = None
      self.V_tilde     = None
      self.C           = None
      self.dataset     = None
    pkl.dump(self, f, protocol=pkl.HIGHEST_PROTOCOL)
    if gut:
      self.V_raw       = V_raw
      self.V_tilde_raw = V_tilde_raw
      self.C_raw       = C_raw
      self.V           = V
      self.V_tilde     = V_tilde
      self.C           = C
      self.datase      = dataset
  
  #=====================================================================
  # Load the model
  @classmethod
  def load(cls, f):
    """"""
    
    return pkl.load(f)
  
#***********************************************************************
# GloVe library (input should be list of tuples)
class GloVe(LOpt):
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
      self._start = unicode('<S>')
    
    if 'stop' in kwargs:
      self._stop = kwargs['stop']
    else:
      self._stop = unicode('</S>')
    
    if 'unk' in kwargs:
      self._unk = kwargs['unk']
    else:
      self._unk = unicode('<UNK>')
    
    if 'lower' in kwargs:
      self.lower = kwargs['lower']
    else:
      self.lower = True
    
    if 'power' in kwargs:
      self.power = np.float32(kwargs['power'])
    else:
      self.power = np.float32(1)
    
    if 'window' in kwargs:
      self.window = np.int32(kwargs['window'])
    else:
      self.window = np.int32(7)
    
    if 'xmax' in kwargs:
      self.xmax = np.float32(kwargs['xmax'])
    else:
      self.xmax = np.float32(10)
    
    if 'xmin' in kwargs:
      self.xmin = np.float32(kwargs['xmin'])
    else:
      self.xmin = np.float32(5)
    
    if 'absolute_difference' in kwargs:
      self.absolute_difference = kwargs['absolute_difference']
    else:
      self.absolute_difference = False
    
    if 'L1reg' in kwargs:
      self.L1reg = np.float32(kwargs['L1reg'])
    else:
      self.L1reg = np.float32(0)
    
    if 'L2reg' in kwargs:
      self.L2reg = np.float32(kwargs['L2reg'])
    else:
      self.L2reg = np.float32(0)
    
    #-------------------------------------------------------------------
    # Declare some variables
    self._wsizes = wsizes
    self.V_raw = list([Counter() for wsize in self._wsizes])
    self.V_tilde_raw = Counter()
    self.C_raw = defaultdict(Counter)
  
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
      self.V_tilde_raw[wrd1] += 1
      for j, feat in enumerate(wrd1):
        self.V_raw[j][feat] += 1
      for j in xrange(i-self.window, (i+1)+self.window):
        if j < 0:
          self.C_raw[wrd1][self.start_tup()] += 1./abs(i-j)
        elif j >= len(sent):
          self.C_raw[wrd1][self.stop_tup()] += 1./abs(i-j)
        elif j != i:
          wrd2 = self.recase(sent[j])
          self.C_raw[wrd1][wrd2] += 1./abs(i-j)
  
  #=====================================================================
  # Build variables
  def build_vars(self):
    """"""
    
    #------------------------------------------------------------------
    # Aggregate the global statistics
    self.V       = [Counter() for wsize in self._wsizes]
    self.V_tilde = Counter()
    self.C       = defaultdict(Counter)
    self._strs = [dict() for wsize in self._wsizes]
    self._idxs = [dict() for wsize in self._wsizes]
    self._strs_tilde = dict()
    self._idxs_tilde = dict()
    self.dataset = []
    
    #-------------------------------------------------------------------
    # Filter rare word features from the word feature libraries
    for i in xrange(len(self.wsizes())):
      V_keys = set(filter(lambda k: self.V_raw[i][k] >= self.xmin, self.V_raw[i].keys()))
      self.V[i][self.unk()] = 0
      for wrd, cnt in self.V_raw[i].iteritems():
        if wrd in V_keys:
          self.V[i][wrd] += cnt
        else:
          self.V[i][self.unk()] += cnt
      for index, string in enumerate(self.V[i]):
        self._strs[i][index]  = string
        self._idxs[i][string] = np.int32(index)
    
    #-------------------------------------------------------------------
    # Filter rare words from the context library
    V_tilde_raw_ = Counter()
    for wrd, cnt in self.V_tilde_raw.iteritems():
      V_tilde_raw_[self.reunk(wrd)] += cnt
    V_tilde_keys = set(filter(lambda k: V_tilde_raw_[k] >= self.xmin, V_tilde_raw_.keys()))
    self.V_tilde[(self.unk(),)] = 0
    for wrd, cnt in V_tilde_raw_.iteritems():
      if wrd in V_tilde_keys:
        self.V_tilde[wrd] += cnt
      else:
        self.V_tilde[(self.unk(),)] += cnt
    for index, string in enumerate(self.V_tilde):
      self._strs_tilde[index] = string
      self._idxs_tilde[string] = np.int32(index)
    
    #-------------------------------------------------------------------
    # Rebuild the count matrix with rare parts/words replaced with unk
    Lb_tilde     = np.zeros(len(self.V_tilde))
    Lb_tilde_cnt = np.zeros(len(self.V_tilde))
    for wrd1, ctr in self.C_raw.iteritems():
      wrd1 = self.reunk(wrd1)
      for wrd2, cnt in ctr.iteritems():
        wrd2 = self.reunk_tilde(wrd2)
        self.C[wrd1][wrd2] += cnt
        Lb_tilde[self.idxs_tilde(wrd2)] += cnt
        Lb_tilde_cnt[self.idxs_tilde(wrd2)] += 1
    Lb_tilde_cnt[np.where(Lb_tilde == 0)] = 1
    
    for wrd1, ctr in self.C.iteritems():
      self.dataset.append((np.array(self.idxs(None, wrd1), dtype='int32'), # x
                           np.array(map(self.idxs_tilde, ctr.keys())), # x_tilde
                           np.array(ctr.values(), dtype='float32'))) # y
    
    #-------------------------------------------------------------------
    # Build model params
    self.theta_L        = []
    self.theta_L_tilde  = []
    self.theta_Lb       = []
    self.theta_Lb_tilde = []
    self.hmask   = theano.shared(np.ones(sum(self.wsizes()), dtype='float32'), name='hmask')
    
    for i, idxs, wsize in zip(range(len(self.idxs())), self.idxs(), self.wsizes()):
      #self.theta_L.append(theano.shared(np.random.normal(0,1, (len(idxs), wsize)).astype('float32'), name='L-%d' % (i+1)))
      self.theta_L.append(theano.shared(np.random.randn(len(idxs), wsize).astype('float32'), name='L-%d' % (i+1)))
      self.theta_Lb.append(theano.shared(np.zeros(len(idxs), dtype='float32'), name='Lb-%d' % (i+1)))
    #self.theta_L_tilde.append(theano.shared(np.random.normal(0,1/np.sqrt(len(self.V_tilde)), size=(len(self.V_tilde), sum(self.wsizes()))).astype('float32'), name='L_tilde'))
    self.theta_L_tilde.append(theano.shared(np.zeros((len(self.V_tilde), sum(self.wsizes())), dtype='float32'), name='L_tilde'))
    #self.theta_Lb_tilde.append(theano.shared(np.zeros(len(self.V_tilde), dtype='float32'), name='L_tilde'))
    temp = np.log(Lb_tilde/Lb_tilde_cnt)
    self.theta_Lb_tilde.append(theano.shared(np.sign(temp)*np.power(np.abs(temp), 1/self.power).astype('float32'), name='Lb_tilde'))
    #self.theta_Lb_tilde.append(theano.shared(np.sign(temp)*np.power(np.abs(temp), 1./len(self.wsizes())).astype('float32'), name='Lb_tilde'))
    self.theta = self.theta_L + self.theta_L_tilde + self.theta_Lb + self.theta_Lb_tilde
    
    #-------------------------------------------------------------------
    # Build gradient accumulators
    self.gtheta_L        = [theano.shared(np.zeros_like(theta_L.get_value()), name='g'+theta_L.name) for theta_L in self.theta_L]
    self.gtheta_Lb       = [theano.shared(np.zeros_like(theta_Lb.get_value()), name='g'+theta_Lb.name) for theta_Lb in self.theta_Lb]
    self.gtheta_L_tilde  = [theano.shared(np.zeros_like(theta_L_tilde.get_value()), name='g'+theta_L_tilde.name) for theta_L_tilde in self.theta_L_tilde]
    self.gtheta_Lb_tilde = [theano.shared(np.zeros_like(theta_Lb_tilde.get_value()), name='g'+theta_Lb_tilde.name) for theta_Lb_tilde in self.theta_Lb_tilde]
    self.gtheta = self.gtheta_L + self.gtheta_L_tilde + self.gtheta_Lb + self.gtheta_Lb_tilde
    
    #-------------------------------------------------------------------
    # Build the input/output variables
    self.x       = T.ivector('x')
    self.x_tilde = T.ivector('x_tilde')
    self.y       = T.fvector('y')
    
    self.stheta_x        = []
    self.stheta_xb       = []
    self.stheta_x_tilde  = []
    self.stheta_xb_tilde = []
    
    for i in xrange(len(self.theta_L)):
      self.stheta_x.append(self.theta_L[i][self.x[i]])
      self.stheta_xb.append(self.theta_Lb[i][self.x[i]])
    self.stheta_x_tilde.append(self.theta_L_tilde[0][self.x_tilde])
    self.stheta_xb_tilde.append(self.theta_Lb_tilde[0][self.x_tilde])
    self.stheta = self.stheta_x + self.stheta_x_tilde + self.stheta_xb + self.stheta_xb_tilde
    
    w        = splice_funx['cat'](self.stheta_x)*self.hmask
    wb       = splice_funx['sum'](self.stheta_xb)
    w_tilde  = self.stheta_x_tilde[0]*self.hmask
    wb_tilde = self.stheta_xb_tilde[0]
    
    #-------------------------------------------------------------------
    # Build the cost variables
    yhat_      = T.dot(w_tilde, w) + wb + wb_tilde
    yhat       = T.sgn(yhat_)*T.power(T.abs_(yhat_), self.power)/self.power
    logy       = T.log(self.y)
    y          = T.switch(T.isinf(logy), np.float32(0), logy)
    f          = reweight_glove(self.y, self.xmax)
    if self.absolute_difference:
      self.error = T.sum(f * absolute_difference(yhat, y))
    else:
      self.error = T.sum(f * squared_difference(yhat, y))
    
    self.complexity = theano.shared(np.float32(0))
    if self.L1reg > 0:
      self.complexity += self.L1reg*sum_absolute_error(self.x, np.float32(0))
      self.complexity += self.L1reg*sum_absolute_error(self.x_tilde, np.float32(0))
    if self.L2reg > 0:
      self.complexity += self.L2reg*sum_squared_error(self.x, np.float32(0))
      self.complexity += self.L2reg*sum_squared_error(self.x_tilde, np.float32(0))
    
    self.cost = self.error + self.complexity
    
    #-------------------------------------------------------------------
    # Build the gradient variables
    self.gstheta = T.grad(self.cost, self.stheta)
    i = 0
    self.gstheta_x        = self.gstheta[i:i+len(self.stheta_x)]
    i += len(self.stheta_x)
    self.gstheta_x_tilde  = self.gstheta[i:i+len(self.stheta_x_tilde)]
    i += len(self.stheta_x_tilde)
    self.gstheta_xb       = self.gstheta[i:i+len(self.stheta_xb)]
    i += len(self.stheta_xb)
    self.gstheta_xb_tilde = self.gstheta[i:]
    
    #===================================================================
    # Activate
    self.idxs_to_val = theano.function(
      inputs=[self.x, self.x_tilde],
      outputs=yhat)
    
    #===================================================================
    # Get w vector
    w = T.imatrix('w')
    self.idxs_to_w = theano.function(
      inputs=[w],
      outputs=T.concatenate([self.theta_L[i][w[:,i]] for i in np.arange(len(self.wsizes()))], axis=1))
    
    #===================================================================
    # Get w_tilde vector
    w_tilde = T.ivector('w_tilde')
    self.idxs_to_w_tilde = theano.function(
      inputs=[w_tilde],
      outputs=self.theta_L_tilde[0][w_tilde])
    
    #===================================================================
    # Error
    self.idxs_to_err = theano.function(
      inputs=[self.x, self.x_tilde, self.y],
      outputs=self.error)
    
    #===================================================================
    # Complexity
    self.idxs_to_comp = theano.function(
      inputs=[self.x, self.x_tilde],
      outputs=self.complexity,
      on_unused_input='ignore')
    
    #===================================================================
    # Cost
    self.idxs_to_cost = theano.function(
      inputs=[self.x, self.x_tilde, self.y],
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
    
    return (self.start(),)*len(self.wsizes())
  
  #=====================================================================
  # Get the stop tuple
  def stop_tup(self):
    """"""
    
    return (self.stop(),)*len(self.wsizes())
  
  #=====================================================================
  # Get the unk tuple
  def unk_tup(self):
    """"""
    
    return (self.unk(),)*len(self.wsizes())
  
  #=====================================================================
  # Make a raw input token the right case (expects string tuple)
  def recase(self, wrd):
    """"""
    
    if self.lower:
      return tuple(unicode(feat).lower() if feat not in (self._start, self._stop, self._unk) else unicode(feat) for feat in wrd)
    else:
      return tuple(unicode(feat) for feat in wrd)
    
  #=====================================================================
  # Replace rare words with the UNK token
  def reunk(self, wrd):
    """"""
    
    return tuple(feat if feat in self.V[i] else self.unk() for i, feat in enumerate(wrd))
  
  #=====================================================================
  # Replace rare tuples with the UNK token
  def reunk_tilde(self, wrd):
    """"""
    
    wrd = self.reunk(wrd)
    return wrd if wrd in self.V_tilde else (self.unk(),)
  
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
      return tuple(strs[feat] for feat, strs in zip(self.recase(wrd), self._strs))
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
      return tuple(idxs.get(feat, idxs[self.unk()]) for feat, idxs in zip(wrd, self._idxs))
    else:
      return self._idxs[lib][wrd]
    
  #=====================================================================
  # Get the strings of some indices 
  def strs_tilde(self, wrd=None):
    """"""
    
    if wrd is None:
      return self._strs_tilde
    else:
      return self._strs[wrd]
  
  #=====================================================================
  # Get the indices of some strings 
  def idxs_tilde(self, wrd=None):
    """"""
    
    if wrd is None:
      return self._idxs_tilde
    else:
      return self._idxs_tilde[wrd]
  
  #=====================================================================
  # Calculate the cost of the dataset
  def batch_cost(self, dataset):
    """"""
    
    cost = 0
    for datum in dataset:
      cost += self.idxs_to_cost(*datum)
    
    return cost/len(dataset)
  
  #=====================================================================
  # The class specific training function
  def __train__(self, dataset, gradientizer, optimizer, nihilizer):
    """"""
    
    cost = 0
    xidxs = [set() for wsize in self.wsizes()]
    xidxs_tilde = set()
    for datum in dataset:
      for i in xrange(len(xidxs)):
        xidxs[i].add(datum[0][i])
      xidxs_tilde.update(datum[1])
    gidxs = [np.array(list(xidx), dtype='int32') for xidx in xidxs] + [np.array(list(xidxs_tilde))]
    for datum in dataset:
      grads = gradientizer(*datum)
      cost += grads
    optimizer(*gidxs)
    nihilizer(*gidxs)
    return cost
  
#***********************************************************************
# Test the program
if __name__ == '__main__':
  """"""
  
  import os.path
  import glob
  import codecs
  
  EPOCHS=10
  PATH='Glove Compartment'
  OPT = 'Adam'
  ETA = 1e-3
  MODEL = 'spd'
  EPSILON = 1e-4
  POWER = 1
  
  i = 0
  while len(sys.argv[1:]) > 0:
    arg = sys.argv.pop(0)
    if arg in ('-o', '--opt'):
      OPT = sys.argv.pop(0)
    elif arg in ('-p', '--power'):
      POWER = float(sys.argv.pop(0))
    elif arg in ('-n', '--nepochs'):
      EPOCHS = int(sys.argv.pop(0))
    elif arg in ('-l', '--learning-rate'):
      ETA = float(sys.argv.pop(0))
    elif arg in ('-m', '--model'):
      MODEL = sys.argv.pop(0)
    elif arg in ('-e', '--epsilon'):
      EPSILON = float(sys.argv.pop(0))
  
  if MODEL == 'spd':
    glove = GloVe([200,50,50], xmin=2, xmax=4, window=7, power=POWER)
    for corpus in glob.glob('Pickle Jar/*.pkl'):
      print corpus
      sents = pkl.load(open(corpus))
      glove.add_corpus(sents)
  elif MODEL == 'sd':
    glove = GloVe([200,100], xmin=3, xmax=6, window=7, power=POWER)
    for corpus in glob.glob('Pickle Jar/*.pkl'):
      print corpus
      sents = pkl.load(open(corpus))
      for i, sent in enumerate(sents):
        sents[i] = [wrd[0:1]+wrd[2:3] for wrd in sent]
      glove.add_corpus(sents)
  elif MODEL == 'sp':
    glove = GloVe([200,100], xmin=4, xmax=8, window=7, power=POWER)
    for corpus in glob.glob('Pickle Jar/*.pkl'):
      print corpus
      sents = pkl.load(open(corpus))
      for i, sent in enumerate(sents):
        sents[i] = [wrd[0:1]+wrd[1:2] for wrd in sent]
      glove.add_corpus(sents)
  elif MODEL == 's':
    glove = GloVe([300], xmin=5, xmax=10, window=7, power=POWER)
    for corpus in glob.glob('Pickle Jar/*.pkl'):
      print corpus
      sents = pkl.load(open(corpus))
      for i, sent in enumerate(sents):
        sents[i] = [wrd[0:1] for wrd in sent]
      glove.add_corpus(sents)
  
  glove.build_vars()
  
  if OPT == 'SGD':
    grad, opt, nihil = glove.build_optimizer(eta=ETA, rho1=0, rho2=0)
  elif OPT == 'NAG':
    grad, opt, nihil = glove.build_optimizer(eta=ETA, rho2=0)
  elif OPT == 'RMSProp':
    grad, opt, nihil = glove.build_optimizer(eta=ETA, rho1=0)
  elif OPT == 'AdaDelta':
    grad, opt, nihil = glove.build_optimizer(eta=ETA, delta_moment=True)
  elif OPT == 'Adam':
    grad, opt, nihil = glove.build_optimizer(eta=ETA, epsilon=EPSILON)
  elif OPT == 'AdaMax':
    grad, opt, nihil = glove.build_optimizer(eta=ETA, max_force=True)
    
  print ETA, EPSILON
  #name=os.path.join(PATH, (('glv'+'-%s'*len(DIM)+'-%s%.1e') % (tuple(DIM)+(OPT,ETA))))
  name=os.path.join(PATH, 'glv-%d-%s-%s' % (POWER,OPT,MODEL))
  glove.train(grad, opt, nihil, save_name=name, print_every=10, epochs=EPOCHS, batch_size=.01)
  