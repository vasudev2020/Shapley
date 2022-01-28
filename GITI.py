#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:56:26 2021

@author: vasu
"""

#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,roc_auc_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from scipy import stats
from statistics import mean,stdev
import warnings
from tqdm import tqdm
from warnings import filterwarnings
from sklearn.exceptions import ConvergenceWarning

import random
#from collections import defaultdict
from statistics import mean


from Embedding import Embedding


class GITI:   
    def __init__(self,Data,Labels,allembs=False):
        self.num_folds = 5
        self.emb = Embedding(allembs)        
        self.Embs = {}
        self.CreateFolds(Data,Labels)
        self.mlp = MLPClassifier()
        self.exps = list(Data.keys())
        self.t = 2
         
    def CreateFolds(self, Data,Labels):
        skf = StratifiedKFold(n_splits=self.num_folds)
        
        self.XFolds,self.YFolds = {},{}
        
        for exp in Data:
            if exp not in self.XFolds:  self.XFolds[exp] = [[],[],[],[],[]] 
            if exp not in self.YFolds:  self.YFolds[exp] = [[],[],[],[],[]] 
            
            for fi, [train_index, test_index] in enumerate(skf.split(Data[exp], Labels[exp])):
                for ti in test_index:
                    self.XFolds[exp][fi].append(Data[exp][ti])
                    self.YFolds[exp][fi].append(Labels[exp][ti])         
                       
    def Evaluate(self,tr_exps,ts_exps):
        auc = []
        for fold in range(5):
            trainX = [self.getEmb(sent,'BERT11') for f in range(5) for exp in tr_exps for sent in self.XFolds[exp][f]   if f!=fold]
            trainY = [l for f in range(5) for exp in tr_exps for l in self.YFolds[exp][f] if f!=fold]
                
            testX = [self.getEmb(sent,'BERT11') for exp in ts_exps for sent in self.XFolds[exp][fold]]
            testY = [l for exp in ts_exps for l in self.YFolds[exp][fold]]
               
            with warnings.catch_warnings():
                filterwarnings("ignore", category=ConvergenceWarning)
                self.mlp.fit(trainX,trainY)
                
            pred_proba = [proba[1] for proba in self.mlp.predict_proba(testX)]
            auc.append(roc_auc_score(testY,pred_proba))
        return mean(auc)
            
        
    def getEmb(self,sent,embtype):
        if sent not in self.Embs:   self.Embs[sent] = self.emb.getMean(sent)
        return self.Embs[sent][embtype]
    
    def getSE(self):
        print('SE')
        for exp in self.exps:   print(exp,self.Evaluate([exp],self.exps))
        
    def getSeenSE(self):
        print('SeenSE')
        for exp in self.exps:   print(exp,self.Evaluate([exp],[exp]))
        
    def getUnseenSE(self):
        print('UnseenSE')
        for exp in self.exps:   print(exp,self.Evaluate([exp],list(set(self.exps)-set([exp]))))
            
    def getLOO(self):
        print('LOO')
        for exp in self.exps:   print(exp,self.Evaluate(list(set(self.exps)-set([exp])),self.exps))
        
    def getDifficulty(self):
        print('Difficulty')
        for exp in self.exps:   print(exp,self.Evaluate(self.exps,[exp]))
        
    '''return shapley values of idiomatic expressions '''
    def getShapleyValues(self):
        SV = {}
        for exp in tqdm(range(len(self.exps))):
            V = 0.0
            for i in range(len(self.exps)):
                for _ in range(self.t):
                    p = list(range(len(self.exps)))
                    random.shuffle(p)
                    j = p.index(exp)
                    p[i],p[j] = p[j],p[i]
                    
                    V += self.Evaluate([self.exps[j] for j in p[:i+1]],self.exps)
                    if i==0:    continue
                    
                    V -= self.Evaluate([self.exps[j] for j in p[:i]],self.exps)
            SV[self.exps[exp]] = V / (len(self.exps)*self.t)
        return SV
    
    def UnseenEvaluate(self,tr_exps,ts_exps):
        assert len(set(tr_exps)&set(ts_exps))==0
        trainX = [self.getEmb(sent,'BERT11') for f in range(5) for exp in tr_exps for sent in self.XFolds[exp][f]]
        trainY = [l for f in range(5) for exp in tr_exps for l in self.YFolds[exp][f]]
    
        testX = [self.getEmb(sent,'BERT11') for f in range(5) for exp in ts_exps for sent in self.XFolds[exp][f]]
        testY = [l for f in range(5) for exp in ts_exps for l in self.YFolds[exp][f]]
        
        with warnings.catch_warnings():
             filterwarnings("ignore", category=ConvergenceWarning)
             self.mlp.fit(trainX,trainY)
                
        pred_proba = [proba[1] for proba in self.mlp.predict_proba(testX)]
        return roc_auc_score(testY,pred_proba)
    
    #def getUnseenExpResults(self,LO,SE,SV):
    def getUnseenExpResults(self,Orders):
        Scores = [[[] for _ in range(20)] for _ in range(len(Orders))]
    
        RO_score = [[] for _ in range(20)]
        
        trials = 10
        p = list(range(len(self.exps)))
        for _ in range(trials):
            random.shuffle(p)
            for fold in range(5):
                test_exps = [self.exps[i] for i in p[fold*5:(fold+1)*5]]
                train_exps = [self.exps[i] for i in p[:fold*5]+p[(fold+1)*5:]]
                #for j in range(20): RO_score[j].append(self.Evaluate(train_exps[:j+1],test_exps))
                for j in range(20): RO_score[j].append(self.UnseenEvaluate(train_exps[:j+1],test_exps))
                
                for i,order in enumerate(Orders):
                    train_exps = [exp for exp in order if exp in train_exps]
                    #for j in range(20): Scores[i][j].append(self.Evaluate(train_exps[:j+1],test_exps))
                    for j in range(20): Scores[i][j].append(self.UnseenEvaluate(train_exps[:j+1],test_exps))
                
        for i,sc in enumerate(Scores):
            print(str(i)+',', ','.join([str(mean(a)) for a in sc]))
        print('RO,',','.join([str(mean(a)) for a in RO_score]))

        for i,sc in enumerate(Scores):
            print(str(i)+'-std,', ','.join([str(stdev(a)) for a in sc]))
        print('RO-std,',','.join([str(stdev(a)) for a in RO_score]))


    
        
        