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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    # instead: first shift the values of f so that the highest number is 0:
    scores -= np.max(scores) # f becomes [-666, -333, 0]
    loss -= np.log(np.exp(scores[y[i]]) / np.sum(np.exp(scores))) # safe to do, gives the correct answer
    dW[:,y[i]] -= X[i]
    for j in range(num_classes):
        dW[:,j] += np.exp(scores[j]) / np.sum(np.exp(scores)) * X[i]
    dW
  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += reg * 2 * W
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
  scores = np.dot(X, W)
  scores -= np.max(scores,1)[:,np.newaxis]
  exp_scores = np.exp(scores)
  # compute loss
  loss = np.sum(-np.log(exp_scores[range(X.shape[0]),y] / np.sum(exp_scores, axis=1)))
  loss /= X.shape[0]
  loss += reg * np.sum(W * W)
  # compute gradient
  count = exp_scores / (np.sum(exp_scores, axis=1)[:,np.newaxis])
  count[range(X.shape[0]),y] -= 1
  dW = np.dot(X.T, count)
  dW /= X.shape[0]
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

