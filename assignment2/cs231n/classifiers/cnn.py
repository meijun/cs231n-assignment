import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    F = num_filters
    HH, WW = filter_size, filter_size
    S = 1 # stride
    P = (filter_size - 1) / 2

    W1 = np.random.normal(scale=weight_scale, size=(F, C, HH, WW))
    b1 = np.zeros(F)

    H_ = (H + 2 * P - HH) / S + 1
    W_ = (W + 2 * P - WW) / S + 1

    # conv: (N, C, H, W) -> (N, F, H_, W_)

    # pool: (N, F, H_, W_) -> (N, F, H_/2, W_/2)

    # hidden: (N, F * H_/2 * W_/2) -> (N, Hi)
    Hi = hidden_dim
    W2 = np.random.normal(scale=weight_scale, size=(F * H_/2 * W_/2, Hi))
    b2 = np.zeros(Hi)
    # out: (N, Hi) -> (N, Nc)
    Nc = num_classes
    W3 = np.random.normal(scale=weight_scale, size=(Hi, Nc))
    b3 = np.zeros(Nc)

    self.params = {
      'W1':W1,
      'W2':W2,
      'W3':W3,
      'b1':b1,
      'b2':b2,
      'b3':b3,
    }
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    X1, conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    X2, hi_cache = affine_relu_forward(X1, W2, b2)
    X3, out_cache = affine_forward(X2, W3, b3)
    scores = X3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dX3 = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
    dX2, dW3, db3 = affine_backward(dX3, out_cache)
    dX1, dW2, db2 = affine_relu_backward(dX2, hi_cache)
    dX, dW1, db1, = conv_relu_pool_backward(dX1, conv_cache)
    grads = {
      'W1':dW1 + self.reg * W1,
      'W2':dW2 + self.reg * W2,
      'W3':dW3 + self.reg * W3,
      'b1':db1,
      'b2':db2,
      'b3':db3,
    }
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
