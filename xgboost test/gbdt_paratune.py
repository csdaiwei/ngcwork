

# csdaiwei@foxmail.com

import pdb

import pickle
import numpy as np
import time

import matplotlib.pyplot as plt

from sklearn import metrics

from collections import Counter

import xgboost as xgb

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from imblearn.ensemble import EasyEnsemble

warnings.simplefilter("error")




def roc(y_true, y_score, pos_label):

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    assert y_true.shape == y_score.shape

    fpr, tpr = [], []

    order = np.argsort(-y_score)
    oyt, oys = y_true[order], y_score[order]    #ordered_y_true, ordered_y_score
    for s in oys:
        tp = ((oys >= s)*(oyt==pos_label)).sum()
        fn = ((oys < s)*(oyt==pos_label)).sum()
        fp = ((oys >= s)*(oyt!=pos_label)).sum()    #performance?
        tn = ((oys < s)*(oyt!=pos_label)).sum()
        tpr.append(float(tp)/(tp+fn))
        fpr.append(float(fp)/(tn+fp))

    return fpr, tpr


def roc_plot(fpr, tpr, auc):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)'%auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


# divide data into trainset and testset, note that
# cut postive examples and negative examples respectively

# f is factor of train/full, usually 0.7
def cut_data(X, Y, f, rand=True):

    permut = np.array(range(0, X.shape[0]))
    if rand!=None:
        np.random.seed(0)
        permut = np.random.permutation(X.shape[0])
    X, Y, indice = X[permut], Y[permut], permut      #side effect? no.

    pos_X, pos_Y, pos_indice = X[Y == 1], Y[Y == 1], indice[Y == 1]
    neg_X, neg_Y, neg_indice = X[Y == 0], Y[Y == 0], indice[Y == 0]

    pos_num = pos_X.shape[0]
    neg_num = neg_X.shape[0]

    train_pos_X, train_pos_Y, train_pos_indice = pos_X[:int(pos_num*f)], pos_Y[:int(pos_num*f)], pos_indice[:int(pos_num*f)]
    train_neg_X, train_neg_Y, train_neg_indice = neg_X[:int(neg_num*f)], neg_Y[:int(neg_num*f)], neg_indice[:int(neg_num*f)]
    test_pos_X, test_pos_Y, test_pos_indice = pos_X[int(pos_num*f):], pos_Y[int(pos_num*f):], pos_indice[int(pos_num*f):]
    test_neg_X, test_neg_Y, test_neg_indice = neg_X[int(neg_num*f):], neg_Y[int(neg_num*f):], neg_indice[int(neg_num*f):]

    train_X, train_Y, train_indice = np.concatenate([train_pos_X, train_neg_X]), np.concatenate([train_pos_Y, train_neg_Y]), np.concatenate([train_pos_indice, train_neg_indice])
    test_X, test_Y, test_indice = np.concatenate([test_pos_X, test_neg_X]), np.concatenate([test_pos_Y, test_neg_Y]), np.concatenate([test_pos_indice, test_neg_indice])


    return train_X, train_Y, train_indice, test_X, test_Y, test_indice


def ee_resampling(X, Y, num, rand=None):

    ee = EasyEnsemble(return_indices=True, n_subsets=num, random_state=rand)
    X_res, Y_res, indice_res = ee.fit_sample(X, Y)
    X, Y, indice = np.concatenate(X_res), np.concatenate(Y_res), np.concatenate(indice_res)

    return X, Y, indice





####
#
#  script runs from here
#
####


#### global constants

REPEAT = 20     # num of repeat experiments

####


# step 1, load data and construct X Y using s35-s38, as 'geno ds'

datasets = pickle.load(open('hhl_geno_6498.pickle', 'rb'))    # this line of code take times

keys = ['s%d'%i for i in range(35, 44)]      # keys of datasets in a sorted manner
group = keys[0:4]
assert set(keys) == set(datasets.keys())

datanames = reduce(lambda x, y:x&y, [set([name[0:-3] for name in datasets[k]['name']]) for k in group])  # common data point names of all signal tunnels in target group

# retrive corresponding features and labels
feats = []
labels = []
for k in group:
    names = [ns[0:-3] for ns in datasets[k]['name']]
    indice = [names.index(dn) for dn in datanames]

    feats.append(np.array(datasets[k]['feature'])[indice][:, :98])
    labels.append(np.array(datasets[k]['label'])[indice])

    if len(labels) >= 2:
        assert (labels[-1] == labels[-2]).all()


X = np.concatenate(feats, axis=1)
Y = labels[0]
Y = (Y<=2).astype(int)          # label reduction
                                # before reduction, label set is [1, 2, 3, 4, 5], smaller is better (windturbine state)
                                # after reduction, [0, 1], 1 is better state than 0


# step 2, train/test sampling 'REPEAT' times

trainsets_X, testsets_X = [], []
trainsets_Y, testsets_Y = [], []
traininds, testinds = [], []

for i in range(0, REPEAT):

    train_X, train_Y, train_indice, test_X, test_Y, test_indice = cut_data(X, Y, f=0.7, rand=i)

    # easyensemble to balance major/minor class examples
    # do ee with training set only
    train_X, train_Y, resindice = ee_resampling(train_X, train_Y, num=5, rand=i)
    train_indice = train_indice[resindice]

    trainsets_X.append(train_X)
    trainsets_Y.append(train_Y)
    traininds.append(train_indice)

    testsets_X.append(test_X)
    testsets_Y.append(test_Y)
    testinds.append(test_indice)


# step 3, generate xgboost parameters to be tested






'''

param = []

# fixed params
param['silent'] = 1
param['eval_metric'] = 'error'
param['objective'] = 'binary:logistic'

# params to be tuned
param['eta'] = 0.01             # learning rate, default 0.3
param['max_depth'] = 2          # maximum depth of a tree, default 6
param['gamma'] = 0              # min_split_loss, default 0
param['lambda'] = 1             # l2-reg of w, default 1
param['alpha'] = 0              # l1-reg of w, default 0

# default params
param['min_child_weight'] = 1   # minimum sum of instance weight needed in a child, default 1
param['max_delta_step'] = 0     # 0 means no limit. might help when extremely class-imbalance
param['subsample'] = 1          #
param['colsample_bytree'] = 1   #
param['colsample_bylevel'] = 1  #
'''

'''
eta [default=0.3, alias: learning_rate]
step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features. and eta actually shrinks the feature weights to make the boosting process more conservative.
range: [0,1]

max_depth [default=6]
maximum depth of a tree, increase this value will make the model more complex / likely to be overfitting.
range: [1, ~]

gamma [default=0, alias: min_split_loss]
minimum loss reduction required to make a further partition on a leaf node of the tree. The larger, the more conservative the algorithm will be.
range: [0, ~]

lambda [default=1, alias: reg_lambda]
L2 regularization term on weights, increase this value will make model more conservative.

alpha [default=0, alias: reg_alpha]
L1 regularization term on weights, increase this value will make model more conservative.
'''

paramsets = []

for max_depth in [4, 5, 6, 7, 8, 10]:      #[8, 10, 12, 15]
    for gamma in [0, 0.01, 0.02, 0.05, 0.2, 0.5]:
        for alpha in [0, 0.01, 0.02, 0.05, 0.2, 0.5]:
            for lamda in [0, 0.01, 0.02, 0.05, 0.2, 0.5]:

                param = {}

                # fixed params
                param['silent'] = 1
                param['eval_metric'] = 'error'
                param['objective'] = 'binary:logistic'
                param['eta'] = 0.1
                
                param['max_depth'] = max_depth
                param['gamma'] = gamma
                param['lambda'] = alpha
                param['alpha'] = lamda


                paramsets.append(param)




# step 4, test parameters and log results


auclist = []
errorlist = []

for n in range(0, len(paramsets)):


    param = paramsets[n]

    plst = list(param.items())+[('eval_metric', 'auc')]

    print n
    print plst

    esr = 100            # early stopping rounds
    num_round = 500

    aucs, errors = [], []

    for i in range(0, REPEAT):

        train_X, train_Y = trainsets_X[i], trainsets_Y[i]
        test_X, test_Y = testsets_X[i], testsets_Y[i]

        res = {}
        dtrain = xgb.DMatrix( train_X, label=train_Y)
        dtest = xgb.DMatrix( test_X, label=test_Y)
        evallist  = [(dtrain,'train'), (dtest,'test')]

        bst = xgb.train( plst, dtrain, num_round, evallist, verbose_eval=False, early_stopping_rounds=esr, evals_result=res)
        auc = res['test']['auc'][bst.best_iteration]
        error = res['test']['error'][bst.best_iteration]

        aucs.append(auc)
        errors.append(error)
        #print i, bst.best_iteration, auc, error
        #pdb.set_trace()


    print time.asctime(), np.array(aucs).mean(), np.array(errors).mean()
    print ' '

    auclist.append(np.array(aucs).mean())
    errorlist.append(np.array(errors).mean())




pickle.dump([auclist, errorlist], open('gbdt_log_depth456.pickle', 'wb'))


pdb.set_trace()



'''

55
[('silent', 1), ('eval_metric', 'error'), ('eta', 0.1), ('objective', 'binary:logistic'), ('alpha', 0.01), ('max_depth', 4), ('gamma', 0.01), ('lambda', 0.05), ('eval_metric', 'auc')]
Fri Feb 17 20:04:12 2017 0.8535014 0.2367348

'''