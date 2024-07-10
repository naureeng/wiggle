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


def NN( x, y0, decoder, CC=1.0, shuf=False, d=None ):

    startTime = datetime.now()
    y = copy.deepcopy(y0)
    print('x.shape:', x.shape, 'y.shape:', y.shape, decoder)

    # cross validation (interleaved is shuf=True)
    folds = 6
    kf = StratifiedKFold(n_splits=folds, random_state=None, shuffle=False)
    splits = kf.split(x, y)

    acs = []
    fold_outputs = []

    for train_index, test_index in splits:

        sc = StandardScaler()
        train_X = sc.fit_transform( x[train_index] )
        test_X  = sc.fit_transform( x[test_index] )

        train_y = y[train_index]
        test_y  = y[test_index]

        if decoder == 'LR':
            clf = LogisticRegression(C = CC, random_state=0, n_jobs=-1)
            clf.fit( train_X, train_y )

            y_pred_test  = clf.predict( test_X )
            y_pred_train = clf.predict( train_X )

        elif decoder == 'LDA':
            clf = LinearDiscriminantAnalysis()
            clf.fit( train_X, train_y )

            y_pred_test  = clf.predict( test_X )
            y_pred_train = clf.predict( train_X )

        qq = test_y == y_pred_test
        
        trial_ind = np.concatenate( (test_index, train_index) )
        output = np.concatenate( (y_pred_test, y_pred_train) )

        arr1inds = trial_ind.argsort()
        sorted_trial_ind = trial_ind[arr1inds]
        sorted_output = output[arr1inds]

        fold_outputs.append(sorted_output)

        res_test  = np.mean(test_y == y_pred_test)
        res_train = np.mean(train_y == y_pred_train)

        ac_test  = round( np.mean(res_test), 4 )
        ac_train = round( np.mean(res_train), 4 )
        acs.append( [ac_train,ac_test] )

        print('train:', ac_train, 'test:', ac_test)

    r_train = round( np.mean(np.array(acs)[:,0]), 3 )
    r_test  = round( np.mean(np.array(acs)[:,1]), 3 )

    print('train:', r_train)
    print('test:', r_test)
    print(f'time to compute:', datetime.now() - startTime )

    return np.transpose(fold_outputs)
