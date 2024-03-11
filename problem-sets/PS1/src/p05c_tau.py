import matplotlib.pyplot as plt
import numpy as np
import util
import math
from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load data set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    
    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    min_mse = math.inf
    optimal_tau = tau_values[0]
    for t in tau_values:
        wlr = LocallyWeightedLinearRegression(tau = t)
        wlr.fit(x_train, y_train)
        y_pred = wlr.predict(x_valid)
        mse = np.mean((y_valid-y_pred)**2)
        #print(f'MSE with tau = {mse}')
        # Plot data
        plt.figure()
        plt.suptitle(f'MSE with tau = {mse}', fontsize=12)
        plt.plot(x_valid, y_valid, 'bx', linewidth=2)
        plt.plot(x_valid, y_pred, 'ro', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('y')
        if mse < min_mse:
            min_mse = mse
            optimal_tau = t
        

    # Fit a LWR model with the best tau value
    wlr = LocallyWeightedLinearRegression(tau = optimal_tau)
    # Run on the test set to get the MSE value
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    mse = np.mean((y_test-y_pred)**2)
    print(f'final MSE with optimal tau = {mse}')
    # Save predictions to pred_path

    # *** END CODE HERE ***
