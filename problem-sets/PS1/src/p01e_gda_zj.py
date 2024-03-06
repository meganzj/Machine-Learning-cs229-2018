import numpy as np
import util
from numpy.linalg import inv
from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    model = GDA()
    model.fit(x_train,y_train)
    model.predict(x_eval)

    # *** START CODE HERE ***
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        #ZJ's notes: when fitting GDA, we do not add intercept as we are modeling p(x|y), adding intercept would make
        #the sigma matrix singular
        # *** START CODE HERE ***
        m,n = x.shape
        self.theta = np.zeros(n+1)
        phi = np.sum(y)/m
        mu_0 = np.sum(x[y==0], axis = 0)/y[y==0].shape[0] 
        mu_1 = np.sum(x[y==1], axis = 0)/y[y==1].shape[0] 
        dist = np.concatenate((x[y==0] - mu_0,x[y==1] - mu_1), axis = 0)
        sigma = dist.T.dot(dist)/m
        sigma_inv=inv(sigma)
        
        self.theta[0] = (mu_0+mu_1).T.dot(sigma_inv).dot(mu_0-mu_1)-np.log((1-phi)/phi)
        self.theta[1:] = sigma_inv.dot(mu_1-mu_0)
        
        
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        #during prediction, we map phi, mu, sigma into theta to calculate posteria probility p(y|x) with sigmoid function,
        #therefore, when prediction, we would introduce intercept so that we can use XT.theta in the formula
        # *** START CODE HERE ***
        m,n = x.shape
        prob_pred = 1 / (1 + np.exp(-x.dot(self.theta)))
        pred = np.zeros(m)
        pred[prob_pred>0.5]=1
        # *** END CODE HERE