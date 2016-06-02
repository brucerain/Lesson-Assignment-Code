import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train): 
    scores = X[i].dot(W)
    scores -= np.max(scores)
    correct_class_score = scores[y[i]]
    sum_exp = np.sum(np.exp(scores))
    loss += (np.log(sum_exp)-correct_class_score)
    
    for j in xrange(num_classes):
        dW[:,j] +=(X[i]* np.exp(scores[j])/sum_exp)
    dW[:,y[i]] -= X[i]
            
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss = loss/num_train + 0.5*reg*np.sum(W*W)
  dW = dW/num_train
 
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_classes = W.shape[1] #C
  num_train = X.shape[0]  #N
  num_dim = X.shape[1]  #D

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  row_max = np.max(scores, axis=1)
  max_scores = np.tile(row_max,(num_classes,1)).T
  scores = scores - max_scores
  sum_exp = np.sum(np.exp(scores),axis=1)  # (N,) sum of each point exp 
  correct_scores = scores[np.arange(num_train),y] #(N,) correct scores of each 
  loss = np.sum(np.log(sum_exp)- correct_scores)

  sum_exp = np.tile(sum_exp,(num_classes,1)).T
  matr = np.exp(scores)/sum_exp  #(N,C) W = X.T * x * W 
  matr[np.arange(num_train),y] -=1   
  dW = X.T.dot(matr) 
 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  loss = loss/num_train + 0.5*reg*np.sum(W*W)
  dW = dW/num_train + reg * W
  return loss, dW

