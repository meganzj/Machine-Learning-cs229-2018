import numpy as np
import util
from numpy.linalg import inv
from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    model = LogisticRegression(eps=1e-5)
    model.fit(x_train, y_train)
    
    util.plot(x_train, y_train, self.theta, 'output/p01b_zj_{}.png'.format(pred_path[-5]))
    
    prediction
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m = x.shape[0]
        n = x.shape[1]
        self.theta = np.zeros(n) #initialize theta
        current_theta = self.theta + 1
        
        while np.linalg.norm(current_theta - self.theta, ord = 1) >= self.eps:
            current_theta = np.copy(self.theta)
            h_x = 1/(1 + np.exp(-x.dot(self.theta))) #our hypothesis for logistics regression
            gradient_J_theta = (1/m) * (h_x - y).T.dot(x) #gradient
            H = (1/m) * (x * h_x[:, np.newaxis] * (1 - h_x)[:, np.newaxis]).T.dot(x) #Hessian matrix
            delta = inv(H).dot(gradient_J_theta) #the delta for Newton methond based on inverse of Hessian matrix and gradient vector
         
            self.theta -= delta # update theta for each iteration
        
        return self.theta
        
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***

        pred = 1/(1 + np.exp(-x.dot(self.theta)))
        return pred
        # *** END CODE HERE ***