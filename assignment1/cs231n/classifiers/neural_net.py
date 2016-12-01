import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    # X=(N, D)  W1=(D, H)  b1=(H,)  W2=(H, C)  b2=(C,)
    H, C = W2.shape
    S1 = X.dot(W1) + b1 # (N, H)
    X1 = np.maximum(S1, 0) # (N, H)
    S2 = X1.dot(W2) + b2 # (N, C)
    scores = S2 #(N, C)
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################
    # E_S2 = np.exp(S2)                            # (1) (N, C)
    # S_E_S2 = np.sum(E_S2, axis=1).reshape(-1,1)  # (2) (N, 1)
    # Ned = E_S2 / S_E_S2                          # (3) (N, C)
    # Log_Ned = -np.log(Ned)                       # (4) (N, C)
    # L = Log_Ned[np.arange(N), y]                 # (5) (N,)
    # sumL = np.sum(L)                             # (6) (1,)
    # avg_L = sumL / N                             # (7) (1,)
    # loss = avg_L + \
    #        0.5 * reg * (                         # (8)
    #          np.sum(W1*W1) +
    #          np.sum(W2*W2)
    #        )
    L = -S2[np.arange(N), y] + np.log(np.sum(np.exp(S2), axis=1))
    loss = np.average(L) + 0.5 * reg * (np.sum(W1*W1) + np.sum(W2*W2))
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    # d_loss = 1
    # d_avg_L = d_loss  # (8)
    # d_W1 = d_loss * reg * W1  # (8)
    # d_W2 = d_loss * reg * W2  # (8)
    # d_sumL = d_avg_L * (1 / N)  # (7)
    # d_L = d_sumL * np.ones_like(L)  # (6)
    # d_Log_Ned = np.ones_like(Log_Ned)  # (5)
    # d_Log_Ned[np.arange(N), y] *= d_L  # (5)
    # d_Ned = d_Log_Ned * (-1 / Ned)  # (4)
    # d_E_S2 = d_Ned / S_E_S2  # (3)
    # d_S_E_S2 = np.sum(d_Ned, axis=1).reshape(-1, 1) * (-np.sum(E_S2, axis=1).reshape(-1, 1) / (S_E_S2**2))  # (3)
    # d_E_S2 += d_S_E_S2 * E_S2  # (2)
    # d_S2 = d_E_S2 * E_S2  # (1)

    d_W1 = reg * W1
    d_W2 = reg * W2
    d_S2 = np.zeros_like(S2)
    d_S2[np.arange(N), y] += -1
    d_S2 += np.exp(S2) / np.sum(np.exp(S2), axis=1).reshape(-1, 1)
    d_S2 /= N

    # S2 = X1.dot(W2) + b2
    d_b2 = np.sum(d_S2, axis=0)
    d_W2 += X1.T.dot(d_S2)
    d_X1 = d_S2.dot(W2.T)
    # X1 = np.maximum(S1, 0)
    d_S1 = d_X1 * (X1 > 0)
    # S1 = X.dot(W1) + b1
    d_b1 = np.sum(d_S1, axis=0)
    d_W1 += X.T.dot(d_S1)

    grads = {
      'W1':d_W1,
      'b1':d_b1,
      'W2':d_W2,
      'b2':d_b2,
    }
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      N, D = X.shape
      choice = np.random.choice(N, batch_size)
      X_batch = X[choice]
      y_batch = y[choice]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      self.params['W1'] += -grads['W1']*learning_rate
      self.params['W2'] += -grads['W2']*learning_rate
      self.params['b1'] += -grads['b1']*learning_rate
      self.params['b2'] += -grads['b2']*learning_rate
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    y_pred = np.argmax(np.maximum(X.dot(self.params['W1'])+self.params['b1'], 0).dot(self.params['W2']) + self.params['b2'], axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


