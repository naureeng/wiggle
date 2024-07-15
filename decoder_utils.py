## decoding functions

## dependencies
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from one.api import ONE
import copy
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification


def NN( x, y0, decoder, CC=1.0, shuf=True):
    """Use LR or LDA decoder
    author: michael schartner
    modified by: naureen ghani

    :param x (arr): input data [#rows x #cols] 
    :param y0 (arr): output data [#rows x 1]
    :param decoder (str): "LR" or "LDA"
    :param CC (float): regularization parameter for Logistic Regression
    :param shuf (bool): whether to shuffle data before splitting into batches

    note: x and y0 must have the same #rows for decoding

    :return np.transpose(fold_outputs) (arr): decoder accuracies in each trial for each fold [6 x #rows]

    """

    startTime = datetime.now()
    y = copy.deepcopy(y0)
    print('x.shape:', x.shape, 'y.shape:', y.shape, decoder)

    # cross-validation set-up
    folds = 6
    kf = StratifiedKFold(n_splits=folds, random_state=None, shuffle=shuf)
    splits = kf.split(x, y)

    accuracies = []
    fold_outputs = []

    for train_index, test_index in splits:

        scalar = StandardScaler()
        train_X = scalar.fit_transform( x[train_index] ) ## fit on training set
        test_X  = scalar.transform( x[test_index] ) ## apply fit to test set

        train_y = y[train_index]
        test_y  = y[test_index]

        if decoder == 'LR':
            clf = LogisticRegression(C = CC, random_state=0, n_jobs=-1)
        elif decoder == 'LDA':
            clf = LinearDiscriminantAnalysis()
        else:
            raise ValueError("Unsupported decoder type. Use 'LR' or 'LDA'.")

        ## decoding
        clf.fit(train_X, train_y)
        y_pred_test  = clf.predict( test_X )
        y_pred_train = clf.predict( train_X )
        
        ## sort decoder result by trial number
        trial_indices = np.concatenate( (test_index, train_index) )
        output = np.concatenate( (y_pred_test, y_pred_train) )
        
        sorted_indices = trial_indices.argsort()
        sorted_output = output[sorted_indices]
        
        fold_outputs.append(sorted_output)

        ## decoder result across trials
        res_test  = np.mean(test_y == y_pred_test)
        res_train = np.mean(train_y == y_pred_train)

        accuracies.append( [res_train, res_test] )

        print('train:', round(res_train,3), 'test:', round(res_test,3))

    mean_accuracies = np.mean(accuracies, axis=0)
    r_train, r_test = round(mean_accuracies[0], 3), round(mean_accuracies[1], 3)

    print('train:', r_train)
    print('test:', r_test)
    print(f'time to compute:', datetime.now() - startTime )

    return np.transpose(fold_outputs)

