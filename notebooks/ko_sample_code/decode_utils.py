#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 01:17:31 2020

@author: kohitij
"""

import numpy as np
#import h5py
#from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from scipy.stats import zscore, norm
from sklearn import preprocessing

#train, test = get_train_test_indices(640)

# lb = ['bear', 'elephant', 'faces', 'car', 'dog', 'apple', 'chair', 'plane']
# labels = np.repeat(lb, 80,axis=0)
#f = h5py.File("many_monkeys.h5",'r')
#r=np.transpose(np.array(f['nano']['left']['rates']))



def decode(features,labels,nrfolds=2,seed=0):
 
    classes=np.unique(labels)
    nrImages = features.shape[1]
    _,ind = np.unique(classes, return_inverse=True)   
    #scale data
    features = zscore(features,axis=0)
    #features = preprocessing.scale(features)
    num_classes = len(classes)
    prob = np.zeros((nrImages,len(classes)))
    prob[:]=np.NAN
    
    for i in range(nrfolds):
        train, test = get_train_test_indices(nrImages,nrfolds=nrfolds, foldnumber=i, seed=seed)
        XTrain = features[:,train]
        #XTrain = preprocessing.scale(XTrain)
        XTest = features[:,test]
        #XTest = preprocessing.scale(XTest)
        YTrain = labels[train]
        #clf = OneVsRestClassifier(SVC(C=5*10e4,kernel='linear',probability=True)).fit(XTrain.T, YTrain)
        clf = LogisticRegression(penalty='l2',C=5*10e4,multi_class='ovr', max_iter=1000, class_weight='balanced').fit(XTrain.T, YTrain)
        pred=clf.predict_proba(XTest.T)
        prob[test,0:num_classes]=pred
    return prob


def get_percent_correct_from_proba(prob, labels,class_order):
    nrImages = prob.shape[0]
    class_order=np.unique(labels)
    pc = np.zeros((nrImages,len(class_order)))
    pc[:]=np.NAN
    _,ind = np.unique(labels, return_inverse=True)
    for i in range(nrImages):
        loc_target = labels[i]==class_order
        pc[i,:] = np.divide(prob[i,labels[i]==class_order],prob[i,:]+prob[i,loc_target])
        pc[i,loc_target]=np.NAN
    return pc

def get_fa(pc, labels):
    #nrImages = pc.shape[0]
    #classes=np.unique(labels)
    #num_classes = len(classes)
    _,ind = np.unique(labels, return_inverse=True)
    full_fa = 1-pc
    pfa = np.nanmean(full_fa,axis=0)
    fa = pfa[ind]    
    return fa, full_fa
    
def get_dprime(pc,fa):
    zHit = norm.ppf(pc)
    zFA = norm.ppf(fa)
    # controll for infinite values
    zHit[np.isposinf(zHit)] = 5
    zFA[np.isneginf(zFA)] = -5
    # Calculate d'
    dp = zHit - zFA
    dp[dp>5]=5
    dp[dp<-5]=-5
    
    return dp
    
    
def get_train_test_indices(totalIndices, nrfolds=10,foldnumber=0, seed=1):
    """
    

    Parameters
    ----------
    totalIndices : TYPE
        DESCRIPTION.
    nrfolds : TYPE, optional
        DESCRIPTION. The default is 10.
    foldnumber : TYPE, optional
        DESCRIPTION. The default is 0.
    seed : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    train_indices : TYPE
        DESCRIPTION.
    test_indices : TYPE
        DESCRIPTION.

    """
    
    np.random.seed(seed)
    inds = np.arange(totalIndices)
    np.random.shuffle(inds)
    splits = np.array_split(inds,nrfolds)
    test_indices = inds[np.isin(inds,splits[foldnumber])]
    train_indices = inds[np.logical_not(np.isin(inds, test_indices))]
    return train_indices, test_indices

    
      
    
    

#W=clf.coef_
#bias = clf.intercept_


