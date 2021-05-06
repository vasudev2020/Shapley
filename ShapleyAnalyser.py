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
import warnings
from tqdm import tqdm
from warnings import filterwarnings
from sklearn.exceptions import ConvergenceWarning

import random
#from collections import defaultdict
from statistics import mean



from Embedding import Embedding


class ShapleyAnalyser:   
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
        for exp in self.exps:   print('exp',self.Evaluate([exp],self.exps))
            
    def getLOO(self):
        print('LOO')
        for exp in self.exps:   print('exp',self.Evaluate(list(set(self.exps)-set([exp])),self.exps))
        
    def getDifficulty(self):
        print('Difficulty')
        for exp in self.exps:   print('exp',self.Evaluate(self.exps),[exp])
        
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

                    #self.train([self.idioms[j] for j in p[:i+1]])
                    #V += self.test(self.idioms)[2]
                    
                    V += self.Evaluate([self.exps[j] for j in p[:i+1]],self.exps)
                    if i==0:    continue
                    #self.train([self.idioms[j] for j in p[:i]])
                    #V -= self.test(self.idioms)[2]
                    V -= self.Evaluate([self.exps[j] for j in p[:i]],self.exps)
            SV[self.exps[exp]] = V / (len(self.exps)*self.t)
            #sh.append(V / (len(self.exps)*self.t))
        return SV
                
    
        
        