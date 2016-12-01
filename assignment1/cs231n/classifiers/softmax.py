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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[1]
  for i in xrange(N):
    f_y = X[i].dot(W)
    loss -= f_y[y[i]]
    loss += np.log(np.sum(np.exp(f_y)))
    dW[:,y[i]] -= X[i].T
    for j in xrange(C):
      dW[:,j] += np.exp(f_y[j]) / np.sum(np.exp(f_y)) * X[i].T
  loss /= N
  dW /= N
  loss += 0.5 * reg * np.sum(W*W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[1]
  F = X.dot(W)
  loss -= np.sum(F[np.arange(N), y])
  CC = np.zeros((N, C))
  CC[np.arange(N), y] = 1
  dW -= X.T.dot(CC)
  EF = np.exp(F)
  sum_ef = np.sum(EF, axis=1)
  EFd = EF / sum_ef.reshape((-1,1))
  # for i in xrange(N):
  #   for j in xrange(C):
  #     dW[:,j] += EFd[i, j] * X[i].T
  dW += X.T.dot(EFd)
  loss += np.sum(np.log(sum_ef))
  loss /= N
  dW /= N
  loss += 0.5 * reg * np.sum(W*W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

