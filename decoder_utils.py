## decoding functions

## dependencies
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from one.api import ONE
import copy
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification


def NN(x, y0, shuf=False):

    acs = []
    probabilities = []

    startTime = datetime.now()
    y = copy.deepcopy(y0)
    print('x.shape:', x.shape, 'y.shape:', y.shape, "Logistic Regression")

    # cross validation (interleaved is shuf=True)
    folds = 6
    kf = StratifiedKFold(n_splits=folds, random_state=None, shuffle=False)
    splits = kf.split(x, y)

    for train_index, test_index in splits:

        sc = StandardScaler()
        train_X = sc.fit_transform( x[train_index] )
        test_X  = sc.fit_transform( x[test_index] )
        train_y = y[train_index]
        test_y  = y[test_index]

        clf = LogisticRegression(C = 1.0, random_state=0, n_jobs=-1)
        clf.fit(train_X, train_y)
        
        # get probabilities
        y_prob_test = clf.predict_proba(test_X)[:, 1]  # probability of positive class
        y_prob_train = clf.predict_proba(train_X)[:, 1]
        
        # calculate accuracies using a threshold of 0.5
        y_pred_test = (y_prob_test >= 0.5).astype(int)
        y_pred_train = (y_prob_train >= 0.5).astype(int)
        
        res_test = np.mean(test_y == y_pred_test)
        res_train = np.mean(train_y == y_pred_train)
        ac_test = round(res_test, 4)
        ac_train = round(res_train, 4)
        acs.append([ac_train, ac_test])
        
        # store probabilities
        probabilities.append(y_prob_test)
        
        print('train:', ac_train, 'test:', ac_test)

    return np.array(acs), np.concatenate(probabilities)

