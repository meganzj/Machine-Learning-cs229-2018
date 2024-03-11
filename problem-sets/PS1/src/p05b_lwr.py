import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    x_train, y_train = util.load_dataset(ds5_training_set_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(ds5_valid_set_path, add_intercept=True)
    clf = LocallyWeightedLinearRegression(tau=tau)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_valid)
    # Get MSE value on the validation set
    mse = np.mean((y_pred - y_eval)**2)
    print(f'MSE={mse}')
    # Plot validation predictions on top of training set

    # No need to save predictions
    # Plot data
    plt.figure()
    plt.plot(x_train, y_train, 'bx', linewidth=2)
    plt.plot(x_eval, y_pred, 'ro', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('output/p05b.png')
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x=x
        self.y=y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m,n = x.shape
        y_pred = np.zeros(m)
        for i in range(m):
            W = np.exp(-np.linalg.norm(self.x - x[i,:], axis = 1) / (2 * self.tau ** 2)) #self.x is the training set
                                                                                         #x is the prediction set                   
            W = np.diag(W)
            #print(W.shape)
            theta = np.linalg.inv(self.x.T.dot(W).dot(self.x)).dot(self.x.T).dot(W).dot(self.y) #each prediction data have it's own theta
            y_pred[i] = x[i,:].dot(theta)
        return y_pred
        # *** END CODE HERE ***