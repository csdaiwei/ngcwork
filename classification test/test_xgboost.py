

# csdaiwei@foxmail.com

import pdb
import sys

import pickle
import numpy as np
from time import time

import matplotlib.pyplot as plt

from sklearn import metrics

from collections import Counter

import xgboost as xgb

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from imblearn.ensemble import EasyEnsemble 


def err_handler(type, flag):
    pdb.set_trace()
np.seterrcall(err_handler)
np.seterr(all='ignore')



# compared models
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# deserted models
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier


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
    print 'usage: $ python test_xgboost.py [picklefilename]'
    print 'as     $ python test_xgboost.py hhl_mb_6498.pickle\n'
    sys.exit()


   

####
#
#  script runs from here 
#
####

LR = 2              # label reduction
REPEAT = 10         # repeat nums
TN_THRES = 0.9      # true negative thresholds
TRAIN_FACTOR = 0.7  # training set factor

#picklefilename = 'hhl_mb_6498.pickle'
picklefilename = sys.argv[1]


print ' '
print 'PARAMETERS:'
print '  pickle finename %s'%picklefilename
print '  label reduction <=%d'%LR
print '  repeat experiment num %d'%REPEAT
print '  training set factor %.4f'%TRAIN_FACTOR
print '  true negative threshold %.4f'%TN_THRES
print ' '



# load data
datasets = pickle.load(open(picklefilename, 'rb'))      # this line takes about 10s
                                                        # close ?

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


assert set(keys) == set(datasets.keys())

resultlog = {}
for (g, gn) in zip(groups, groupnames):
    resultlog[gn] = []


# For each signal tunnel group, feed data to all candidate classification models
# Log and print time & accuracy of each model
for (g, gn, dim) in zip(groups, groupnames, groupdims):

    print 'On signal group %s %s'%(gn, g)
    print ' '
    

    # construct data matrix
    
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


    X = np.concatenate(feats, axis=1)
    Y = labels[0]

    print('  Original dataset shape {}'.format(Counter(Y)))

    Y = (Y<=LR).astype(int)         # label reduction
                                    # before reduction, label set is [1, 2, 3, 4, 5], smaller is better (windturbine state)
                                    # after reduction, [0, 1], 1 is better state than 0

    if len(set(Y)) <= 1:
        print '  only 1 unique class after label reduction, break\n'
        continue


    # train/test sampling 'REPEAT' times

    trainsets_X, trainsets_Y = [], []
    testsets_X, testsets_Y = [], []
    traininds, testinds = [], []

    for epoch in range(0, REPEAT):
    
        train_X, train_Y, train_indice, test_X, test_Y, test_indice = cut_data(X, Y, f=TRAIN_FACTOR, rand=epoch)

        # easyensemble to balance major/minor class examples
        # do ee with training set only
        res_train_X, res_train_Y, resindice = ee_resampling(train_X, train_Y, num=5, rand=epoch)
        res_train_indice = train_indice[resindice]

        trainsets_X.append(res_train_X)
        trainsets_Y.append(res_train_Y)
        traininds.append(train_indice)

        testsets_X.append(test_X)
        testsets_Y.append(test_Y)
        testinds.append(test_indice)
    
    
    print('  After label reduction, {}'.format(Counter(Y)))
    print('  ')
    print('  Cutting as training/testing sets ')
    print('    training dataset shape {}'.format(Counter(train_Y)))
    print('    testing  dataset shape {}'.format(Counter(test_Y)))
    print('  EasyEnsemble resampling')
    print('    training dataset shape {}'.format(Counter(res_train_Y)))
    print('    testing  dataset shape {}'.format(Counter(test_Y)))
    print('  ')


    for epoch in range(0, REPEAT):
        
        train_X, train_Y = trainsets_X[epoch], trainsets_Y[epoch]
        test_X, test_Y = testsets_X[epoch], testsets_Y[epoch]



        start = time()


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

        dtrain, dtest = xgb.DMatrix( train_X, label=train_Y), xgb.DMatrix( test_X, label=test_Y)
        evallist  = [(dtrain,'train'), (dtest,'test')]

        res = {}
        bst = xgb.train( plst, dtrain, num_round, evallist, verbose_eval=False, early_stopping_rounds=esr, evals_result=res)
        auc = res['test']['auc'][bst.best_iteration]
        error = res['test']['error'][bst.best_iteration]

        #print epoch, bst.best_iteration, auc, error


        scores = bst.predict(dtest)


        # auc
        fpr, tpr = roc(test_Y, scores, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        
        
        # acc under threshold tn==n

        test_neg_num = sum(test_Y == 0)

        thresholds = list(set(scores))
        thresholds.sort()

        for th in thresholds:
            predicts = (scores > th).astype('int')

            tp = ((predicts==test_Y)*(predicts == 1)).sum() 
            fp = ((predicts!=test_Y)*(predicts == 1)).sum() 
            tn = ((predicts==test_Y)*(predicts == 0)).sum() 
            fn = ((predicts!=test_Y)*(predicts == 0)).sum() 
                


            if tn >= min(test_neg_num-1, test_neg_num*TN_THRES):
                break
        
        '''
        predicts = model.predict(test_X)
        tp = ((predicts==test_Y)*(predicts == 1)).sum() 
        fp = ((predicts!=test_Y)*(predicts == 1)).sum() 
        tn = ((predicts==test_Y)*(predicts == 0)).sum() 
        fn = ((predicts!=test_Y)*(predicts == 0)).sum() 
        '''

        acc = (predicts == test_Y).mean()


        resultlog[gn].append([auc, acc, tp, fp, tn, fn])

        print '      auc:%.4f accu:%.4f tp:%d fp:%d tn:%d fn:%d, time:%4f, %s'%(auc, acc, tp, fp, tn, fn, time() - start, 'xgb')

        pdb.set_trace()

    
    #print ' '


print 'result on %d average:\n'%(REPEAT)

for (g, gn) in zip(groups, groupnames):

    print '  signal group %s %s'%(gn, g)
    print '        auc     accu    tp      fp      tn      fn'

    
    print '  %s '%'xgbt' ,
    r = np.array(resultlog[gn])
    for i in range(0, r.shape[1]):
        s = '%.4f'%r[:, i].mean()
        print '%s '%s[0:6] ,
    print ' '
    print ' '



pdb.set_trace()

'''
# Save results as a sheet(xls) , optional

import xlwt
wb = xlwt.Workbook()
ws = wb.add_sheet('Sheet1')

# create sheet head
for i in range(0, len(keys)):
    ws.write(i+2, 0, keys[i])
for j in range(0, len(models)):
    ws.write(0, 2*j+1, models[j].__name__)
    ws.write(1, 2*j+1, 'accu')
    ws.write(1, 2*j+2, 'auc')

# insert result value
for i, k in zip(range(0, len(keys)), keys):
    acc_r = acc_results[k]
    auc_r = auc_results[k]
    for j, acc, auc in zip(range(0, len(acc_r)), acc_r, auc_r):
        ws.write(i+2, 2*j+1, acc)
        ws.write(i+2, 2*j+2, auc)

wb.save('resultlog_resample0.xls')
'''

