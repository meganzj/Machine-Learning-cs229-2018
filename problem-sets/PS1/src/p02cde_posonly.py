import numpy as np
import util

from p01b_logreg_zj import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    x_train, y_train = util.load_dataset(ds3_training_set_path, add_intercept=True)
    _, t_train = util.load_dataset(ds3_training_set_path, label_col='t', add_intercept=True)
    
    x_eval, y_eval = util.load_dataset(ds3_valid_set_path, add_intercept=True)
    _, t_eval = util.load_dataset(ds3_valid_set_path, label_col='t', add_intercept=True)

    x_test, y_test = util.load_dataset(ds3_test_set_path, add_intercept=True)
    _, t_test = util.load_dataset(ds3_test_set_path, label_col='t', add_intercept=True)
    
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    model = LogisticRegression(eps=1e-5)
    model.fit(x_train, t_train)
    np.savetxt(pred_path_c, model.predict(x_train) >0.5, fmt='%d')
    

    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    model_y = LogisticRegression(eps=1e-5)
    model_y.fit(x_train, y_train)
    np.savetxt(pred_path_d, model_y.predict(x_train) >0.5, fmt='%d')    
    
    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    alpha = np.average(model_y.predict(x_eval)[y_eval==1])
    eval_pred_updated = model_y.predict(x_eval)/alpha
    np.savetxt(pred_path_e, eval_pred_updated >0.5, fmt='%d') 
    #calaulate updated theta for plot updates
    theta_0_adj_factor = np.log(2/alpha-1)
    theta_0_adj_factor * np.array([1,0,0])
    theta_update = model_y.theta + theta_0_adj_factor * np.array([1,0,0])
    
    #plot c,d,e
    util.plot(x_test, y_test, model.theta, pred_path_c)

    
    util.plot(x_test, y_test, model_y.theta, pred_path_d)

    
    util.plot(x_test, y_test, theta_update, pred_path_e)

    # *** END CODER HERE
