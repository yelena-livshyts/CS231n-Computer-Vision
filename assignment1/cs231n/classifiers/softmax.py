from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange
import math


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
    squared_norm_W = np.sum(W*W)
    loss = 0.0
    dW = np.zeros_like(W)
    

   
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes=W.shape[1]
    for i in range(num_train):
      scores = X[i].dot(W)
      sum= np.sum(np.exp(scores))
      Li = math.log(sum)-scores[y[i]]
      loss+=Li
      for j in range(num_classes):
        #the j columns of dW, dW[:,j], is the sum over i of
        # (Sj-delta(y[i],j))*xi
        dW[:,j] +=  ((math.exp(scores[j])/sum)- (j==y[i]))*X[i]
    loss=loss*(1/num_train)
    loss+=(0.5*reg*squared_norm_W) 
    dW=dW*(1/num_train)
    dW+=reg*squared_norm_W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.

    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    scores = X.dot(W) #shape is (num_train,num_classes)
    scores -=   np.max(scores, axis=1, keepdims=True) #for numerical stability,
    #                                                  doesnt change the result.
    scores_exp = np.exp(scores)
    scores_exp /= np.sum(scores_exp, axis=1, keepdims=True)  

    # Li -=scores[y[i]] so L -=sum(scores[y[i]]) for all i, similar to naive 
    #                                                            impementation.

    loss -= np.sum(np.log(scores_exp[np.arange(num_train), y]))
    loss /= num_train
    loss += 0.5 * reg * np.sum(W**2)

    
    scores_exp[np.arange(num_train), y] -= 1  #if (i==y[i]) then -X[i], similar
    #                                                    to naive implementation
    dW = np.dot(X.T, scores_exp) 
    dW /= num_train
    dW += reg * W
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
