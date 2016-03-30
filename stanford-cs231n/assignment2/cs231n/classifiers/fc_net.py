import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *

def affine_bn_relu_forward(x, w, b, gamma,beta, bn_param): 
  """ 
  Convenience layer that performs an affine transform followed by 
  batch normalization and relu layers
  
  Inputs: 
  - x : Input to the affine layer
  - w, b : weights for the affine layer
  - gamma, beta:  paras for batch normalization layer
  - bn_param: dictionary with the param keys

  Return a tuple of: 
  - out: out from the ReLu 
  - Cache: objects to pass to the backward pass 
  
  """
  affine_out, fc_cache = affine_forward(x,w,b)
  norm_out , norm_cache = batchnorm_forward(affine_out, gamma,beta, bn_param)
  out, relu_cache = relu_forward(norm_out)
  cache = (fc_cache, norm_cache, relu_cache)
  return out, cache

def affine_bn_relu_backward(dout, cache): 
   """
   Backward pass for the affine bn relu backward convenience
   """ 
   fc_cache, norm_cache, relu_cache = cache
   da = relu_backward(dout, relu_cache)
   dnorm_input, dgamma, dbeta = batchnorm_backward(da,norm_cache)
   dx, dw,db  = affine_backward(dnorm_input, fc_cache)
   return dx, dw, db, dgamma, dbeta
   

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['W2'] = weight_scale * np.random.randn(hidden_dim,num_classes)
    self.params['b2'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    reg = self.reg
    #X_shape = np.reshape(X,(X.shape[0],-1))
    W1, b1 = self.params["W1"], self.params["b1"]
    W2, b2 = self.params["W2"], self.params["b2"]
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    af1_relu_out, af1_relu_cache = affine_relu_forward(X,W1, b1)
    
    af2_out, af2_cache = affine_forward(af1_relu_out,W2,b2)
   
    scores = af2_out
    C = scores.shape[1] 
    N = X.shape[0]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    # !!!!! Commented by clyu
    # There remains problems in the code of softmax loss function and dX, dW
    #row_max = np.max(scores, axis=1)
    #max_scores = np.tile(row_max, (C,1)).T
    #scores_l = scores - max_scores            #(N,C)
    #sum_exp = np.sum(np.exp(scores_l),axis=1)
    #correct_scores = scores_l[np.arange(N),y]
    #loss = np.sum(np.log(sum_exp)-correct_scores)
    loss, matr = softmax_loss(scores,y)
    loss += 0.5*reg*(np.sum(W1*W1) + np.sum(W2*W2))

    #sum_exp = np.tile(sum_exp,(C,1)).T
    #matr = np.exp(scores_l)/sum_exp 
    #matr[np.arange(N),y] =-1
    #matr corresponding to soft-max
    drelu_out, dW2 , db2 = affine_backward(matr,af2_cache)
    _, dW1, db1 = affine_relu_backward(drelu_out , af1_relu_cache)
    
    grads["W2"] = dW2 + reg * W2
    grads["W1"] = dW1 + reg * W1
    grads["b1"] = db1
    grads["b2"] = db2
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    self.params['W1'] = np.random.normal(0,weight_scale, (input_dim, hidden_dims[0]))
    self.params['b1'] = np.zeros(hidden_dims[0])
    for i in range(1, len(hidden_dims)): 
      self.params['W'+ str(i+1)] = np.random.normal(0, weight_scale,(hidden_dims[i-1],hidden_dims[i]))
      self.params['b'+ str(i+1)] = np.zeros(hidden_dims[i])
    self.params['W' + str(self.num_layers)] = np.random.normal(0, weight_scale, (hidden_dims[len(hidden_dims)-1],num_classes))
    self.params['b' + str(self.num_layers)] = np.zeros(num_classes)

    ## added by using batch normalization 
    for i in range(1, self.num_layers): 
      if use_batchnorm: 
        self.params['gamma' + str(i)] = np.ones(hidden_dims[i-1])
        self.params['beta' + str(i)] = np.zeros(hidden_dims[i-1])
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = {}
    cache = {}
    score = None
    dropout_cache = {}
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    hidden_num = self.num_layers -1 
    if not self.use_batchnorm:
      if not self.use_dropout:         
        scores[1],cache[1] = affine_relu_forward(X, self.params['W1'], self.params['b1'])
      else: 
        scores[1],cache[1] = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        scores[1], dropout_cache[1] = dropout_forward(scores[1], self.dropout_param)
        
    else:
      if not self.use_dropout: 
        scores[1],cache[1] = affine_bn_relu_forward(X, self.params['W1'], self.params['b1'], self.params['gamma1'], self.params['beta1'], self.bn_params[0])
      else: 
        scores[1],cache[1] = affine_bn_relu_forward(X, self.params['W1'], self.params['b1'], self.params['gamma1'], self.params['beta1'], self.bn_params[0])	
        scores[1], dropout_cache[1] = dropout_forward(scores[1], self.dropout_param)

    for i in range(2, hidden_num+1):
        if not self.use_batchnorm:
	  if not self.use_dropout:  
    	    scores[i], cache[i] = affine_relu_forward(scores[i-1], self.params['W'+ str(i)], self.params['b' + str(i)])
          else:
    	    scores[i], cache[i] = affine_relu_forward(scores[i-1], self.params['W'+ str(i)], self.params['b' + str(i)])
            scores[i], dropout_cache[i] = dropout_forward(scores[i], self.dropout_param)

        if self.use_batchnorm:
          if not self.use_batchnorm:
    	    scores[i], cache[i] = affine_bn_relu_forward(scores[i-1], self.params['W'+ str(i)], self.params['b' + str(i)], self.params['gamma'+ str(i)], self.params['beta'+ str(i)], self.bn_params[i-1])
          else: 
    	    scores[i], cache[i] = affine_bn_relu_forward(scores[i-1], self.params['W'+ str(i)], self.params['b' + str(i)], self.params['gamma'+ str(i)], self.params['beta'+ str(i)], self.bn_params[i-1])
            scores[i], dropout_cache[i] = dropout_forward(scores[i], self.dropout_param)

    scores[hidden_num+1], cache[hidden_num+1] = affine_forward(scores[hidden_num], self.params['W'+ str(hidden_num+1)], self.params['b' + str(hidden_num+1)])
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    score = scores[hidden_num+1]
    # If test mode return early
    if mode == 'test':
      return score

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss, matr = softmax_loss(score,y)
    for i in range(1,self.num_layers+1):
    	loss += 0.5* self.reg * np.sum(self.params['W'+str(i)] * self.params['W'+ str(i)])
    
    
    tmp = self.num_layers
    #Affine_backward
    dx = {}
    dw = {}
    db = {}
    dgamma = {}
    dbeta = {}
    if not self.use_batchnorm: 
      dx[tmp],dw[tmp],db[tmp] = affine_backward(matr,cache[self.num_layers])
      for i in range(self.num_layers-1,0,-1): 
         if not self.use_dropout:
	   dx[i],dw[i],db[i] = affine_relu_backward(dx[i+1],cache[i])
         else: 
           dx[i] = dropout_backward(dx[i+1],dropout_cache[i])
	   dx[i],dw[i],db[i] = affine_relu_backward(dx[i],cache[i])

    if self.use_batchnorm: 
      dx[tmp],dw[tmp],db[tmp] = affine_backward(matr,cache[self.num_layers])
      for i in range(self.num_layers-1, 0,-1):
        if not self.use_dropout:
          dx[i], dw[i], db[i],dgamma[i], dbeta[i] = affine_bn_relu_backward(dx[i+1], cache[i])
        else: 
          dx[i] = dropout_backward(dx[i+1],dropout_cache[i])
          dx[i], dw[i], db[i],dgamma[i], dbeta[i] = affine_bn_relu_backward(dx[i], cache[i])
        grads['gamma' + str(i)]  = dgamma[i]
        grads['beta' + str(i)] = dbeta[i]

    for i in range(1,tmp+1):
        grads['W'+str(i)] = dw[i] + self.reg * self.params['W'+str(i)]
        grads['b' + str(i)] = db[i]

   
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
