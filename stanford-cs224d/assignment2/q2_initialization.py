import numpy as np
import tensorflow as tf

def xavier_weight_init():
  """
  Returns function that creates random tensor. 

  The specified function will take in a shape (tuple or 1-d array) and must
  return a random tensor of the specified shape and must be drawn from the
  Xavier initialization distribution.

  Hint: You might find tf.random_uniform useful.
  """
  def _xavier_initializer(shape, **kwargs):
    """Defines an initializer for the Xavier distribution.

    This function will be used as a variable scope initializer.

    https://www.tensorflow.org/versions/r0.7/how_tos/variable_scope/index.html#initializers-in-variable-scope

    Args:
      shape: Tuple or 1-d array that species dimensions of requested tensor.
    Returns:
      out: tf.Tensor of specified shape sampled from Xavier distribution.
    """
    ### YOUR CODE HERE
    dim = [i for i in shape]
    ## Xavier Initialization: A[m,n] uniformly from [-a,a], a = sqrt(6)/sqrt(m+n)
    parm = np.sqrt(6)/np.sqrt(np.sum(dim))
    out = tf.random_uniform(shape,-parm,parm)
    ## sign, think too much, waste so much time ##
    #with  tf.variable_scope("xavier", initializer = initializer) as scope:
    # out = tf.get_variable("out", shape)
    
    ### END YOUR CODE
    return out
  # Returns defined initializer function.
  return _xavier_initializer

def test_initialization_basic():
  """
  Some simple tests for the initialization.
  """
  print "Running basic tests..."
  xavier_initializer = xavier_weight_init()
  shape = (1,)
  xavier_mat = xavier_initializer(shape)
  with tf.Session() as sess:
    print sess.run(xavier_mat)
  assert xavier_mat.get_shape() == shape

  shape = (1, 2,3)
  xavier_mat = xavier_initializer(shape)
  with tf.Session() as sess:
    print sess.run(xavier_mat)
  assert xavier_mat.get_shape() == shape

  print "Basic (non-exhaustive) Xavier initialization tests pass\n"

def test_initialization():
  """ 
  Use this space to test your Xavier initialization code by running:
      python q1_initialization.py 
  This function will not be called by the autograder, nor will
  your tests be graded.
  """
  print "Running your tests..."
  ### YOUR CODE HERE
  ### END YOUR CODE  

if __name__ == "__main__":
    test_initialization_basic()
