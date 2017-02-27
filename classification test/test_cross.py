

# csdaiwei@foxmail.com

import pdb
import sys

import pickle
import numpy as np
from time import time

import matplotlib.pyplot as plt

from sklearn import metrics

from collections import Counter

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
    print 'usage: $ python test_cross.py [mb/gear/geno]'
    print 'as     $ python test_cross.py mb\n'
    sys.exit()
   

####
#
#  script runs from here 
#
####

LR = 1              # label reduction
TN_THRES = 0.8      # true negative thresholds

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
print '  true negative threshold %.4f'%TN_THRES
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


# load classification prototype
model = GradientBoostingClassifier()

# For each signal tunnel group, feed data to all candidate classification models
# Log and print time & accuracy of each model
for (g, gn, dim) in zip(groups, groupnames, groupdims):

    print ' '
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


    X1 = np.concatenate(feats, axis=1)
    Y1 = labels[0]


    # find common data point names of all signal tunnels in target group
    datanames2 = reduce(lambda x, y:x&y, [set([name.split('.s')[0] for name in datasets2[k]['name']]) for k in g])

    # retrive corresponding features and labels
    feats = []
    labels = []
    for k in g:
        names = [ns.split('.s')[0] for ns in datasets2[k]['name']]
        indice = [names.index(dn) for dn in datanames2]

        feats.append(np.array(datasets2[k]['feature'])[indice][:, :dim])
        labels.append(np.array(datasets2[k]['label'])[indice])

        #feat = np.array(datasets2[k]['feature'])[indice][:, :98]
        #pdb.set_trace()


        if len(labels) >= 2:
            assert (labels[-1] == labels[-2]).all()


    X2 = np.concatenate(feats, axis=1)
    Y2 = labels[0]

    Y1r, Y2r = (Y1<=LR).astype(int), (Y2<=LR).astype(int)  

    res_X1, res_Y1r, res_indice = ee_resampling(X1, Y1r, num=10, rand=1)
    
    print('  Original dataset1 shape {}'.format(Counter(Y1)))
    print('  Original dataset2 shape {}'.format(Counter(Y2)))
    print('  After label reduction, dataset1 {}'.format(Counter(Y1r)))
    print('  After label reduction, dataset2 {}'.format(Counter(Y2r)))
    print('  EasyEnsemble resampling dataset1 {}'.format(Counter(res_Y1r)))
    print(' ')

    model.fit(res_X1, res_Y1r)
    assert model.classes_[1] == 1
    scores = model.predict_proba(X2)[:, 1]

    fpr, tpr = roc(Y2r, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    test_neg_num = sum(Y2r == 0)

    thresholds = list(set(scores))
    thresholds.sort()

    for th in thresholds:
        predicts = (scores > th).astype('int')

        tp = ((predicts==Y2r)*(predicts == 1)).sum() 
        fp = ((predicts!=Y2r)*(predicts == 1)).sum() 
        tn = ((predicts==Y2r)*(predicts == 0)).sum() 
        fn = ((predicts!=Y2r)*(predicts == 0)).sum() 
                

        if tn >= min(test_neg_num-1, test_neg_num*TN_THRES):
            break
    
    acc = (predicts == Y2r).mean()



    print '  training by dataset1, test by dataset2:'
    print '    auc:%.4f accu:%.4f tp:%d fp:%d tn:%d fn:%d %s'%(auc, acc, tp, fp, tn, fn, 'sklearn.gbdt')
    print ' '

    print '%-8s\t%-8s\t%-8s\t%-8s'%('turbine', 'predict', 'rlabel', 'label')

    for (nn, pp, yyr, yy) in zip(datanames2, predicts, Y2r, Y2):

        print '%-8s\t%-8s\t%-8s\t%-8s'%(nn, pp, yyr, yy)




pdb.set_trace()