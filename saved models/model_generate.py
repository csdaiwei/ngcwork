

# csdaiwei@foxmail.com

import pdb
import sys

import pickle
import numpy as np

import matplotlib.pyplot as plt

from sklearn import metrics

from collections import Counter

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from imblearn.ensemble import EasyEnsemble 


# models
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def err_handler(type, flag):
    pdb.set_trace()
np.seterrcall(err_handler)
np.seterr(all='ignore')



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



if len(sys.argv) != 2:
    print 'usage: $ python model_generate.py [mb/gear/geno]'
    print 'as     $ python model_generate.py mb\n'
    sys.exit()



####
#
#  script runs from here 
#
####

LR = 2               # label reduction
TRAIN_FACTOR = 0.9   # training set factor
#REPEAT = 10         # repeat nums
#TN_THRES = 0.9      # true negative thresholds


picklefilename, picklefilename2 = '', ''

if sys.argv[1] == 'mb':
    picklefilename = 'hhl_mb_6498.pickle'
    picklefilename2 = 'by_mb_6498.pickle'

elif sys.argv[1] == 'gear':
    picklefilename = 'hhl_gear_complex.pickle'
    picklefilename2 = 'by_gear_complex.pickle'

elif sys.argv[1] == 'geno':
    picklefilename = 'hhl_geno_6498.pickle'
    picklefilename2 = 'by_geno_6498.pickle'


print ' '
print 'PARAMETERS:'
print '  pickle finename %s %s'%(picklefilename, picklefilename2)
print '  label reduction <=%d'%LR
print '  training set factor %.4f'%TRAIN_FACTOR
#print '  repeat experiment num %d'%REPEAT
#print '  true negative threshold %.4f'%TN_THRES
print ' '



# load data
datasets = pickle.load(open(picklefilename, 'rb'))
datasets2 = pickle.load(open(picklefilename2, 'rb'))

keys, groups, groupnames, groupdims = [], [], [], []

if 'mb' in picklefilename:

    keys = ['s%d'%i for i in range(0, 6)]      # keys of datasets in a sorted manner
    groups = [keys[0:6]]      
    groupnames = ['MB']
    groupdims = [98]

elif 'gear' in picklefilename:
    
    keys = ['s%d'%i for i in range(6, 35)]      # keys of datasets in a sorted manner
    groups = [keys[0:6], keys[6:12], keys[12:19], keys[19:24], keys[24:29]]        
    groupnames = ['Gear Input', 'Pl. Stage', 'LSS', 'IMS', 'HSS']
    groupdims = [13, 182, 182, 287, 287]

elif 'geno' in picklefilename:
    
    keys = ['s%d'%i for i in range(35, 44)]      # keys of datasets in a sorted manner
    groups = [keys[0:4], keys[4:]]        
    groupnames = ['Geno DS', 'Geno NDS']
    groupdims = [98, 98]

else:
    print 'picklefilename error'
    pdb.set_trace()



# For each signal tunnel group, feed data to all candidate classification models
# Log and print time & accuracy of each model
for (g, gn, dim) in zip(groups, groupnames, groupdims):

    print 'On signal group %s %s'%(gn, g)
    print ' '
    

    # step 1, construct data matrix
    
    # find common data point names of all signal tunnels in target group
    datanames = reduce(lambda x, y:x&y, [set([name.split('.s')[0] for name in datasets[k]['name']]) for k in g])

    # retrive corresponding features and labels
    feats = []
    labels = []
    for k in g:
        names = [ns.split('.s')[0] for ns in datasets[k]['name']]
        indice = [names.index(dn) for dn in datanames]

        feats.append(np.array(datasets[k]['feature'])[indice][:, :dim])
        labels.append(np.array(datasets[k]['label'])[indice])

        #feat = np.array(datasets[k]['feature'])[indice][:, :98]
        #pdb.set_trace()


        if len(labels) >= 2:
            assert (labels[-1] == labels[-2]).all()


    X1 = np.concatenate(feats, axis=1)
    Y1 = labels[0]


    # find common data point names of all signal tunnels in target group
    datanames = reduce(lambda x, y:x&y, [set([name.split('.s')[0] for name in datasets2[k]['name']]) for k in g])

    # retrive corresponding features and labels
    feats = []
    labels = []
    for k in g:
        names = [ns.split('.s')[0] for ns in datasets2[k]['name']]
        indice = [names.index(dn) for dn in datanames]

        feats.append(np.array(datasets2[k]['feature'])[indice][:, :dim])
        labels.append(np.array(datasets2[k]['label'])[indice])

        #feat = np.array(datasets2[k]['feature'])[indice][:, :98]
        #pdb.set_trace()


        if len(labels) >= 2:
            assert (labels[-1] == labels[-2]).all()


    X2 = np.concatenate(feats, axis=1)
    Y2 = labels[0]

    X = np.concatenate([X1, X2])
    Y = np.concatenate([Y1, Y2])


    print('  Original dataset shape {}'.format(Counter(Y)))

    Y = (Y<=LR).astype(int)         # label reduction
                                    # before reduction, label set is [1, 2, 3, 4, 5], smaller is better (windturbine state)
                                    # after reduction, [0, 1], 1 is better state than 0


    # step 2, resampling X and Y

    train_X, train_Y, train_indice, test_X, test_Y, test_indice = cut_data(X, Y, f=TRAIN_FACTOR, rand=0)
    res_train_X, res_train_Y, resindice = ee_resampling(train_X, train_Y, num=10, rand=0)
    res_train_indice = train_indice[resindice]
                                   

    if len(set(Y)) <= 1:
        print '  only 1 unique class after label reduction, break\n'
        continue


    print('  After label reduction, {}'.format(Counter(Y)))
    print('  ')
    print('  Cutting as training/testing sets ')
    print('    training dataset shape {}'.format(Counter(train_Y)))
    print('    testing  dataset shape {}'.format(Counter(test_Y)))
    print('  EasyEnsemble resampling')
    print('    training dataset shape {}'.format(Counter(res_train_Y)))
    print('    testing  dataset shape {}'.format(Counter(test_Y)))
    print('  ')


    # step 3, train model and dump

    models = {}

    lr = LogisticRegression()
    lr.fit(res_train_X, res_train_Y)

    rf = RandomForestClassifier()
    rf.fit(res_train_X, res_train_Y)

    params = {}
    params['max_depth'] = 4
    params['gamma'] = 0.01
    params['alpha'] = 0.01
    params['lambda'] = 0.05
    params['eta'] = 0.1
    params['silent'] = 1
    params['eval_metric'] = 'error'
    params['objective'] = 'binary:logistic'

    esr = 100            # early stopping rounds
    num_round = 500
    plst = list(params.items())+[('eval_metric', 'auc')]

    dtrain, dtest = xgb.DMatrix( res_train_X, label=res_train_Y), xgb.DMatrix( test_X, label=test_Y)
    evallist  = [(dtrain,'train'), (dtest,'test')]

    bst = xgb.train( plst, dtrain, num_round, evallist, verbose_eval=False, early_stopping_rounds=esr)

    '''
    scores = lr.predict_proba(test_X)[:, 1]
    scores = rf.predict_proba(test_X)[:, 1]
    scores = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
    fpr, tpr = roc(test_Y, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print auc
    '''

    models['lr'] = lr
    models['rf'] = rf
    models['bst'] = bst

    pickle.dump(models, open('%s.models'%gn, 'wb'))

pdb.set_trace()



