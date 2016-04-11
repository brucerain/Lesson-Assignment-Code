import numpy as np

##########################################################################
#How to understand the gradient changes in the recurrent neural network ## 
##########################################################################

"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for the backward pass.
  """
  next_h, cache = None, None
  ##############################################################################
  # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
  # hidden state and any values you need for the backward pass in the next_h   #
  # and cache variables respectively.                                          #
  ##############################################################################
  next_h = tanh(x.dot(Wx) +  prev_h.dot(Wh)+b)
  cache = (x, prev_h,Wx, Wh,b, next_h)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return next_h, cache


def rnn_step_backward(dnext_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN.
  
  Inputs:
  - dnext_h: Gradient of loss with respect to next hidden state(N,H)
  - cache: Cache object from the forward pass
  
  Returns a tuple of:
  - dx: Gradients of input data, of shape (N, D)
  - dprev_h: Gradients of previous hidden state, of shape (N, H)
  - dWx: Gradients of input-to-hidden weights, of shape (D, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """
  dx, dprev_h, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
  #                                                                            #
  # HINT: For the tanh function, you can compute the local derivative in terms #
  # of the output value from tanh.                                             #
  ##############################################################################
  (x,prev_h, Wx, Wh, b, next_h) = cache
  dtanh = dnext_h *(np.ones_like(next_h) - next_h * next_h)
  dWx = x.T.dot(dtanh)
  dx = dtanh.dot(Wx.T)
  dWh = prev_h.T.dot(dtanh)
  dprev_h = dtanh.dot(Wh.T)
  db = np.sum(dtanh,axis=0) 
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.
  
  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)
  
  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  h, cache = None, None
  ##############################################################################
  # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
  # input data. You should use the rnn_step_forward function that you defined  #
  # above.                                                                     #
  ##############################################################################
  N, T, D = x.shape
  H = h0.shape[1]
  x_trans = x.transpose(1,0,2)  # (T,N,D)
  middle = np.zeros((T,N,H))
  middle[0], cache[0] = rnn_step_forward(x_trans[0],h0,Wx,Wh,b)
  for i in range(1,T):
    tmp, _ =rnn_step_forward(x_trans[i], middle[i-1], Wx,Wh,b)
    middle[i] = tmp
  h = middle.transpose(1,0,2)
  cache = (x,x_trans,h0,Wx,Wh,b,h, middle) 
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return h, cache


def rnn_backward(dh, cache):
  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data.
  
  Inputs:
  - dh: Upstream gradients of all hidden states, of shape (N, T, H)
  
  Returns a tuple of:
  - dx: Gradient of inputs, of shape (N, T, D)
  - dh0: Gradient of initial hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - db: Gradient of biases, of shape (H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a vanilla RNN running an entire      #
  # sequence of data. You should use the rnn_step_backward function that you   #
  # defined above.                                                             #
  ##############################################################################
  (x, x_trans, h0, Wx, Wh, b,h, middle) = cache
  N, T,D = x.shape
  H = b.shape[0]
  dx_trans = np.ones_like(x_trans)
  dmiddle = np.zeros_like(middle)
  dWx = np.zeros((D,H))
  dWh = np.zeros((H,H))
  db = np.zeros(H)
  dh_trans = dh.transpose(1,0,2)
  for i in range(T-1,0,-1): 
    tmp_cache = (x_trans[i], middle[i-1],Wx,Wh,b,middle[i]) 
    dx_trans[i], dmiddle[i-1], dWx_l, dWh_l, db_l = rnn_step_backward(dmiddle[i] + dh_trans[i] ,tmp_cache)
    dWx += dWx_l
    dWh += dWh_l 
    db += db_l

  tmp_cache = (x_trans[0], h0, Wx, Wh,b, middle[0])
  dx_trans[0], dh0, dWx_l, dWh_l, db_l = rnn_step_backward(dmiddle[0] + dh_trans[0], tmp_cache)
  dWx += dWx_l
  dWh += dWh_l 
  db += db_l
  # Back propagation to first layer
  dx = dx_trans.transpose(1,0,2) 
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dh0, dWx, dWh, db

def word_embedding_forward(x, W):
  """
  Forward pass for word embeddings. We operate on minibatches of size N where
  each sequence has length T. We assume a vocabulary of V words, assigning each
  to a vector of dimension D.
  
  Inputs:
  - x: Integer array of shape (N, T) giving indices of words. Each element idx
    of x muxt be in the range 0 <= idx < V.
  - W: Weight matrix of shape (V, D) giving word vectors for all words.
  
  Returns a tuple of:
  - out: Array of shape (N, T, D) giving word vectors for all input words.
  - cache: Values needed for the backward pass
  """
  out, cache = None, None
  ##############################################################################
  # TODO: Implement the forward pass for word embeddings.                      #
  #                                                                            #
  # HINT: This should be very simple.                                          #
  ##############################################################################
  N, T = x.shape 
  V, D = W.shape 
  out  = np.zeros((N,T,D))
  for i in range(N):
    for j in range(T):
      out[i][j] = W[x[i][j]]
  
  cache = (x, W)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return out, cache


def word_embedding_backward(dout, cache):
  """
  Backward pass for word embeddings. We cannot back-propagate into the words
  since they are integers, so we only return gradient for the word embedding
  matrix.
  
  HINT: Look up the function np.add.at
  
  Inputs:
  - dout: Upstream gradients of shape (N, T, D)
  - cache: Values from the forward pass
  
  Returns:
  - dW: Gradient of word embedding matrix, of shape (V, D).
  """
  dW = None
  ##############################################################################
  # TODO: Implement the backward pass for word embeddings.                     #
  #                                                                            #
  # HINT: Look up the function np.add.at                                       #
  ##############################################################################
  (x, W) = cache
  N,T,D  = dout.shape
  dW = np.zeros_like(W)
  for i in range(N):
    for j in range(T):
      np.add.at(dW,[x[i][j]], dout[i][j])
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dW


def sigmoid(x):
  """
  A numerically stable version of the logistic sigmoid function.
  """
  pos_mask = (x >= 0)
  neg_mask = (x < 0)
  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])
  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)

def tanh(x):
  """
  A implemention of the logistic tanh function 
  """
  pos_mask = (x>=0)
  neg_mask = (x<0)
  pos_z = np.exp(x)
  neg_z = np.exp(-x)
  #neg_z[pos_mask] = np.exp(-x[pos_mask]) 
  #neg_z[neg_mask] = np.exp(x[neg_mask])
  return (pos_z - neg_z) /(pos_z + neg_z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  """
  Forward pass for a single timestep of an LSTM.
  
  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.
  
  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)
  
  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  next_h, next_c, cache = None, None, None
  #############################################################################
  # TODO: Implement the forward pass for a single timestep of an LSTM.        #
  # You may want to use the numerically stable sigmoid implementation above.  #
  #############################################################################
  N, D = x.shape
  H = prev_h.shape[1]
  affine = x.dot(Wx) + prev_h.dot(Wh) + b    # (N,4H)
  input_gate = sigmoid(affine[:,:H]) 
  forget_gate = sigmoid(affine[:,H:2 * H])
  output_gate = sigmoid(affine[:,2 * H:3 * H])
  block_input = np.tanh(affine[:,3 * H:4 * H])
  
  next_c = forget_gate * prev_c + input_gate * block_input
  next_h = output_gate * np.tanh(next_c)
  
  cache = (x,prev_h,prev_c, next_h, next_c,Wx,Wh,b,affine)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
  """
  Backward pass for a single timestep of an LSTM.
  
  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for a single timestep of an LSTM.       #
  #                                                                           #
  # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
  # the output value from the nonlinearity.                                   #
  #############################################################################
  (x,prev_h,prev_c , next_h, next_c,Wx, Wh,b, affine) = cache
  H = prev_h.shape[1]  
 
  input_gate = sigmoid(affine[:,:H]) 
  forget_gate = sigmoid(affine[:,H:2 * H])
  output_gate = sigmoid(affine[:,2 * H:3 * H])
  block_input = np.tanh(affine[:,3 * H:4 * H])
  
  doutput_gate = dnext_h * np.tanh(next_c)
  # Take notes : dnext_c += dnext_c 
  dnext_c = dnext_c + dnext_h * output_gate*(1-np.tanh(next_c) * np.tanh(next_c))
 
  dforget_gate = dnext_c * prev_c
  dprev_c = dnext_c * forget_gate
  dinput_gate = dnext_c * block_input
  dblock_input = dnext_c * input_gate
  daffine_s = np.hstack((input_gate *(1-input_gate) *dinput_gate, forget_gate *(1-forget_gate)*dforget_gate, output_gate *(1-output_gate) *doutput_gate, (1- block_input * block_input) *dblock_input)) 
  
  dWx = x.T.dot(daffine_s)
  dx =  daffine_s.dot(Wx.T)
  dWh = prev_h.T.dot(daffine_s)
  db = np.sum(daffine_s,axis=0)
  dprev_h = daffine_s.dot(Wh.T)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.
  
  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.
  
  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)
  
  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  h, cache = None, None
  #############################################################################
  # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
  # You should use the lstm_step_forward function that you just defined.      #
  #############################################################################
  cache_forward = {}
  next_c = {}
  N,T,D = x.shape
  H = h0.shape[1]
  prev_c = np.zeros_like(h0)
  h = np.zeros((N,T,H))
  h[:,0,:], next_c[0], cache_forward[0] = lstm_step_forward(x[:,0,:], h0,prev_c, Wx, Wh, b)
  for i in range(1,T): 
    h[:,i,:], next_c[i],cache_forward[i] = lstm_step_forward(x[:,i,:], h[:,i-1,:], next_c[i-1], Wx, Wh,b)

  cache = (x,h0, Wx, Wh,b, cache_forward)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return h, cache


def lstm_backward(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]
  
  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
  # You should use the lstm_step_backward function that you just defined.     #
  #############################################################################
  N,T,H = dh.shape
  (x, h0, Wx, Wh, b, cache_forward) = cache
  D = x.shape[2]
  dx = np.zeros((N,T,D))
  dWx = np.zeros((D, 4*H))
  dWh = np.zeros((H, 4*H))
  db = np.zeros(4*H) 
  dprev_c = np.zeros((N,H))
  dprev_h = np.zeros((N,H))

  for i in range(T-1,0,-1): 
    dx[:,i,:], dprev_h , dprev_c, dWx_l, dWh_l, db_l = lstm_step_backward(dh[:,i,:] + dprev_h,dprev_c, cache_forward[i])
    dWx += dWx_l 
    dWh += dWh_l 
    db += db_l 

  dx[:,0,:], dh0, dprev_c, dWx_l, dWh_l ,db_l = lstm_step_backward(dh[:,0,:] + dprev_h , dprev_c, cache_forward[0])
  dWx += dWx_l 
  dWh += dWh_l 
  db += db_l

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
   
  return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
  """
  Forward pass for a temporal affine layer. The input is a set of D-dimensional
  vectors arranged into a minibatch of N timeseries, each of length T. We use
  an affine function to transform each of those vectors into a new vector of
  dimension M.

  Inputs:
  - x: Input data of shape (N, T, D)
  - w: Weights of shape (D, M)
  - b: Biases of shape (M,)
  
  Returns a tuple of:
  - out: Output data of shape (N, T, M)
  - cache: Values needed for the backward pass
  """
  N, T, D = x.shape
  M = b.shape[0]
  out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
  cache = x, w, b, out
  return out, cache


def temporal_affine_backward(dout, cache):
  """
  Backward pass for temporal affine layer.

  Input:
  - dout: Upstream gradients of shape (N, T, M)
  - cache: Values from forward pass

  Returns a tuple of:
  - dx: Gradient of input, of shape (N, T, D)
  - dw: Gradient of weights, of shape (D, M)
  - db: Gradient of biases, of shape (M,)
  """
  x, w, b, out = cache
  N, T, D = x.shape
  M = b.shape[0]

  dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
  dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
  db = dout.sum(axis=(0, 1))

  return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
  """
  A temporal version of softmax loss for use in RNNs. We assume that we are
  making predictions over a vocabulary of size V for each timestep of a
  timeseries of length T, over a minibatch of size N. The input x gives scores
  for all vocabulary elements at all timesteps, and y gives the indices of the
  ground-truth element at each timestep. We use a cross-entropy loss at each
  timestep, summing the loss over all timesteps and averaging across the
  minibatch.

  As an additional complication, we may want to ignore the model output at some
  timesteps, since sequences of different length may have been combined into a
  minibatch and padded with NULL tokens. The optional mask argument tells us
  which elements should contribute to the loss.

  Inputs:
  - x: Input scores, of shape (N, T, V)
  - y: Ground-truth indices, of shape (N, T) where each element is in the range
       0 <= y[i, t] < V
  - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss.

  Returns a tuple of:
  - loss: Scalar giving loss
  - dx: Gradient of loss with respect to scores x.
  """

  N, T, V = x.shape
  
  x_flat = x.reshape(N * T, V)
  y_flat = y.reshape(N * T)
  mask_flat = mask.reshape(N * T)
  
  probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
  dx_flat = probs.copy()
  dx_flat[np.arange(N * T), y_flat] -= 1
  dx_flat /= N
  dx_flat *= mask_flat[:, None]
  
  if verbose: print 'dx_flat: ', dx_flat.shape
  
  dx = dx_flat.reshape(N, T, V)
  
  return loss, dx

