

# csdaiwei@foxmail.com

import pdb
import sys

import pickle
import numpy as np

# models
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

groupnames = ['MB', 'Gear Input', 'Pl. Stage', 'LSS', 'IMS', 'HSS', 'Geno DS', 'Geno NDS']


if len(sys.argv) != 2:
    print 'usage: $ python predict.py [npzfilename]'
    print 'as     $ python model_generate.py MB.npz\n'
    sys.exit()

testfile = sys.argv[1]

print 'test matrix file: %s\n'%testfile

####
#
#  script runs from here 
#
####


npzfile  = np.load(testfile)
test_X = npzfile['arr_0']

print 'test matrix shape: ', test_X.shape
print ' '

for gn in groupnames:
    if gn in testfile:
        break

modelfile = './models/%s.models'%(gn)

print 'use pretrained model %s\n'%modelfile

models = pickle.load(open(modelfile, 'rb')) 

lr = models['lr']
rf = models['rf']
bst = models['bst']


lr_scores = lr.predict_proba(test_X)[:, 1]
rf_scores = rf.predict_proba(test_X)[:, 1]
bst_scores = bst.predict(xgb.DMatrix( test_X), ntree_limit=bst.best_ntree_limit)
test_Y = npzfile['arr_1']
test_Y_r = (test_Y <=2).astype(int)

print 'LogisticRegression scores:'
print lr_scores
print ' '

print 'RandomForestClassifier scores:'
print rf_scores
print ' '

print 'xgboost scores'
print bst_scores
print ' '

pdb.set_trace()



