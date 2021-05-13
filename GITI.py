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
    
    def getUnseenExpResults(self,LO,SE,SV):
        #LO = ['kick heel','have word','pull leg','make pile','blow whistle','get wind','make hay','blow trumpet','make face','hit road','pull punch','blow top','cut figure','get sack','get nod','see star','hit wall','hold fire','make hit','make scene','hit roof','pull plug','take heart','pull weight','make mark','lose thread','find foot','lose head']
        #SE = ['pull plug','get wind','blow trumpet','hit road','kick heel','pull weight','blow whistle','hit roof','take heart','make pile','make hay','make mark','get sack','pull punch','blow top','cut figure','have word','lose thread','find foot','get nod','pull leg','make scene','make face','hold fire','lose head','hit wall','make hit','see star']
        #SV = ['make mark','blow whistle','pull plug','blow trumpet','pull weight','make hay','hit roof','get wind','take heart','hit road','make pile','get sack','find foot','kick heel','make face','make scene','pull punch','blow top','lose thread','get nod','cut figure','pull leg','have word','hold fire','lose head','hit wall','make hit','see star']

        #LO_score = [0.0]*20
        #SE_score = [0.0]*20
        #SV_score = [0.0]*20
        #RO_score = [0.0]*20
        LO_score = [[] for _ in range(20)]
        SE_score = [[] for _ in range(20)]
        SV_score = [[] for _ in range(20)]
        RO_score = [[] for _ in range(20)]
        
        trials = 2
        p = list(range(len(self.exps)))
        for _ in range(trials):
            random.shuffle(p)
            for fold in range(5):
                test_exps = [self.exps[i] for i in p[fold*5:(fold+1)*5]]
                train_exps = [self.exps[i] for i in p[:fold*5]+p[(fold+1)*5:]]
                lo_train_exps = [exp for exp in LO if exp in train_exps]
                se_train_exps = [exp for exp in SE if exp in train_exps]
                sv_train_exps = [exp for exp in SV if exp in train_exps]
                '''
                for i in range(20): LO_score[i]+=self.Evaluate(lo_train_exps[:i+1],test_exps)
                for i in range(20): SE_score[i]+=self.Evaluate(se_train_exps[:i+1],test_exps)
                for i in range(20): SV_score[i]+=self.Evaluate(sv_train_exps[:i+1],test_exps)
                for i in range(20): RO_score[i]+=self.Evaluate(train_exps[:i+1],test_exps)
                '''
                for i in range(20): LO_score[i].append(self.Evaluate(lo_train_exps[:i+1],test_exps))
                for i in range(20): SE_score[i].append(self.Evaluate(se_train_exps[:i+1],test_exps))
                for i in range(20): SV_score[i].append(self.Evaluate(sv_train_exps[:i+1],test_exps))
                for i in range(20): RO_score[i].append(self.Evaluate(train_exps[:i+1],test_exps))
         
        '''
        for i in range(20):
            LO_score[i]/=trials*5
            SE_score[i]/=trials*5
            SV_score[i]/=trials*5
            RO_score[i]/=trials*5
            
        print('LO:',LO_score)
        print('SE:',SE_score)
        print('SV:',SV_score)
        print('RO:',RO_score)
        '''
        
        print('LO',','.join([str(mean(a)) for a in LO_score]))
        print('LO-std',','.join([str(stdev(a)) for a in LO_score]))
        
        print('SE',','.join([str(mean(a)) for a in SE_score]))
        print('SE-std',','.join([str(stdev(a)) for a in SE_score]))
        
        print('SV',','.join([str(mean(a)) for a in SV_score]))
        print('SV-std',','.join([str(stdev(a)) for a in SV_score]))
        
        print('RO',','.join([str(mean(a)) for a in RO_score]))
        print('RO-std',','.join([str(stdev(a)) for a in RO_score]))
    
        
        