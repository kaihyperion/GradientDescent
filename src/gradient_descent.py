import numpy as np
from your_code import HingeLoss, SquaredLoss
from your_code import L1Regularization, L2Regularization





class GradientDescent:
    """
    This is a linear classifier similar to the one you implemented in the
    linear regressor homework. This is the classification via regression
    case. The goal here is to learn some hyperplane, y = w^T x + b, such that
    when features, x, are processed by our model (w and b), the result is
    some value y. If y is in [0.0, +inf), the predicted classification label
    is +1 and if y is in (-inf, 0.0) the predicted classification label is
    -1.

    The catch here is that we will not be using the closed form solution,
    rather, we will be using gradient descent. In your fit function you
    will determine a loss and update your model (w and b) using gradient
    descent. More details below.

    Arguments:
        loss - (string) The loss function to use. Either 'hinge' or 'squared'.
        regularization - (string or None) The type of regularization to use.
            One of 'l1', 'l2', or None. See regularization.py for more details.
        learning_rate - (float) The size of each gradient descent update step.
        reg_param - (float) The hyperparameter that controls the amount of
            regularization to perform. Must be non-negative.
    """
    def __init__(self, loss, regularization=None,
                 learning_rate=0.01, reg_param=0.05):
        self.learning_rate = learning_rate

        # Select regularizer
        if regularization == 'l1':
            regularizer = L1Regularization(reg_param)
        elif regularization == 'l2':
            regularizer = L2Regularization(reg_param)
        elif regularization is None:
            regularizer = None
        else:
            raise ValueError(
                'Regularizer {} is not defined'.format(regularization))

        # Select loss function
        if loss == 'hinge':
            self.loss = HingeLoss(regularizer)
        elif loss == 'squared':
            self.loss = SquaredLoss(regularizer)
        else:
            raise ValueError('Loss function {} is not defined'.format(loss))

        self.model = None
        self.lossdata = []
        self.accuracydata = []
        self.it = 0

    def fit(self, features, targets, batch_size=None, max_iter=1000):
        """
        Fits a gradient descent learner to the features and targets. The
        pseudocode for the fitting algorithm is as follow:
          - Initialize the model parameters to uniform random values in the
            interval [-0.1, +0.1].
          - While not converged:
            - Compute the gradient of the loss with respect to the current
              batch.
            - Update the model parameters by moving them in the direction
              opposite to the current gradient. Use the learning rate as the
              step size.
        For the convergence criteria, compute the loss over all examples. If
        this loss changes by less than 1e-4 during an update, assume that the
        model has converged. If this convergence criteria has not been met
        after max_iter iterations, also assume convergence and terminate.

        You should include a bias term by APPENDING a column of 1s to your
        feature matrix. The bias term is then the last value in self.model.

        Arguments:
            features - (np.array) An Nxd array of features, where N is the
                number of examples and d is the number of features.
            targets - (np.array) A 1D array of targets of length N.
            batch_size - (int or None) The number of examples used in each
                iteration. If None, use all of the examples in each update.
            max_iter - (int) The maximum number of updates to perform.
        Modifies:
            self.model - (np.array) A 1D array of model parameters of length
                d+1. The +1 refers to the bias term.
        """
        self.model = np.random.uniform(-0.1, 0.1, features.shape[1] + 1)
        ones = np.ones((1,features.shape[0]))
        f_bias = np.append(features, ones.T, axis=1)
        prev_loss = self.loss.forward(f_bias, self.model, targets)
        lossdifference = 100 #arbitrary high number to pass first 'if' statement
        it = 0

        epoch_ct = 0
        batch_idx = 0

        while it != max_iter:
            if batch_size:
                if lossdifference < 1e-4 and epoch_ct == 0:
                    break
                if epoch_ct == 0:
                    np.random.seed()
                    randomized = np.random.permutation(features.shape[0])
                    targets_shuf = targets[randomized]
                    f_bias_shuf = f_bias[randomized]
                epoch_ct += 1
                batch = f_bias_shuf[batch_idx: batch_idx + batch_size]
                batch_targets = targets_shuf[batch_idx:batch_idx + batch_size]
                batch_idx += batch_size
                gradient = self.loss.backward(batch, self.model, batch_targets)
                self.model -= self.learning_rate * gradient

                if batch_idx >= features.shape[0]:
                    new_loss = self.loss.forward(f_bias, self.model, targets)
                    self.lossdata.append(new_loss)
                    self.accuracydata.append(accuracy(targets, self.predict(features)))
                    epoch_ct = 0
                    lossdifference = abs(prev_loss - new_loss)
                    prev_loss = new_loss
                    batch_idx = 0
            else:
                if lossdifference < 1e-4:
                    break
                gradient = self.loss.backward(f_bias, self.model, targets)
                self.model -= self.learning_rate * gradient
                new_loss = self.loss.forward(f_bias, self.model, targets)
                self.lossdata.append(new_loss)
                self.accuracydata.append(accuracy(targets, self.predict(features)))
                lossdifference = abs(prev_loss - new_loss)
                prev_loss = new_loss
            it+=1

        self.it = it



    def predict(self, features):
        """
        Predicts the class labels of each example in features. Model output
        values at and above 0 are predicted to have label +1. Non-positive
        output values are predicted to have label -1.

        NOTE: your predict function should make use of your confidence
        function (see below).

        Arguments:
            features - (np.array) A Nxd array of features, where N is the
                number of examples and d is the number of features.
        Returns:
            predictions - (np.array) A 1D array of predictions of length N,
                where index d corresponds to the prediction of row N of
                features.
        """
        confidences = self.confidence(features)
        predictions = np.zeros(confidences.shape[0])
        for n in range(confidences.shape[0]):
            if confidences[n] < 0: predictions[n] = -1
            else: predictions[n] = 1

        return predictions

    def confidence(self, features):
        """
        Returns the raw model output of the prediction. In other words, rather
        than predicting +1 for values above 0 and -1 for other values, this
        function returns the original, unquantized value.

        Arguments:
            features - (np.array) A Nxd array of features, where N is the
                number of examples and d is the number of features.
        Returns:
            confidence - (np.array) A 1D array of confidence values of length
                N, where index d corresponds to the confidence of row N of
                features.
        """
        ones = np.ones((1, features.shape[0]))
        f_bias = np.append(features, ones.T, axis=1)
        for row in range(features.shape[0]):
            f_bias[row] *= self.model
        confidence = np.sum(f_bias, axis=1)
        return confidence

#accuracy copied from metrics
def accuracy(ground_truth, predictions):
    return np.mean(ground_truth == predictions)