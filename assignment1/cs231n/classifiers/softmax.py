from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    # compute all scores
    for i in range(num_train):
        scores = X[i].dot(W)
        scores_exp = np.exp(scores)
        sum_scores_exp = np.sum(scores_exp)
        probab_normal = scores_exp/sum_scores_exp
        # take only correct class probability
        probab_normal_corr_class = probab_normal[y[i]]
        loss += -np.log(probab_normal_corr_class)

        for j in range(num_classes):
          if j != y[i]:
            dW[:,j] += X[i].dot(probab_normal[j])
        dW[:,y[i]] += X[i].dot(probab_normal[y[i]] - 1)
        
    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += 2 * reg * W

    pass

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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


     # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = np.zeros((num_train,num_classes))
    
    # compute all scores
    scores = X.dot(W)
    # exponentiate all scores to find unnormalized probabilities
    scores_exp = np.exp(scores)
    # summ all probabilities
    sum_scores_exp = np.sum(scores_exp, axis = 1)
    sum_scores_exp = sum_scores_exp[:,np.newaxis]
    
    # take only correct class scores
    corr_scores_exp = scores_exp[np.arange(num_train),y]
    corr_scores_exp = corr_scores_exp[:,np.newaxis]
    
    # find normalized probabilities
    probab_normal = corr_scores_exp/sum_scores_exp
    
    # compute loss
    loss = -np.log(probab_normal)
    loss = np.sum(loss)
    loss /= num_train
    loss += reg * np.sum(W * W)

    # compute dW
    all_prob_normal = scores_exp/sum_scores_exp

    # take away 1 from the probab of correct class
    corr_class_arr = np.zeros_like(all_prob_normal)
    corr_class_arr[np.arange(num_train), y] = 1
    a = all_prob_normal - corr_class_arr

    b = X.T

    dW= b.dot(a)

    dW /= num_train
    dW += 2 * reg * W



    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
