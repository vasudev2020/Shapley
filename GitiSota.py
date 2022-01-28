#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 18:43:31 2021

@author: vasu
"""

import pickle, random, time
import statistics as stat
import warnings

import torch
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier


from pytorch_pretrained_bert import BertTokenizer, BertModel

''' Class to train and evaluate general idiom token idenitification model 
    on dataset used to evaluate state of the art model'''
class GITISota:
    def __init__(self,data_file,idiom_file,emb='cmb_skip',model='MLP'):        
        dataset = pickle.load(open(data_file, "rb"), encoding='latin1')
        if model=='MLP':   self.model = MLPClassifier()
        if model=='SVM':   self.model = LinearSVC()
        if model=='SGD':   self.model = SGDClassifier(loss='hinge', penalty='l1', alpha=0.001, tol=0.0001, max_iter=15)
        #self.model = SGDClassifier()
        self.bt = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert.eval()
       
        with open(idiom_file,'r') as fp:    idioms = [l.strip().split('\t') for l in fp.readlines()]
        
        idioms = sorted(idioms,key=lambda x: x[1])
        self.idioms = [x[0] for x in idioms]
                
        train_data = dataset["train_sample"]
        test_data = dataset["test_sample"]
        
        print(len(train_data),len(test_data))
        
        self.x_train, self.y_train, self.x_test, self.y_test = {},{},{},{}
        for idiom in self.idioms:
            self.x_train[idiom],self.y_train[idiom],self.x_test[idiom],self.y_test[idiom],=[],[],[],[]
            
        for train in train_data:
            self.x_train[train['verb']+' '+train['noun']].append(self.embedding(train,emb))
            self.y_train[train['verb']+' '+train['noun']].append(train["lab_int"])
            
        for test in test_data:
            self.x_test[test['verb']+' '+test['noun']].append(self.embedding(test,emb))
            self.y_test[test['verb']+' '+test['noun']].append(test["lab_int"])    
        
    def embedding(self,p,emb):
        if emb == 'cmb_skip':   return p['cmb_skip']
        if emb == 'bert':   return self.BERT(p['sent'])
        print("no associated embedding for",emb)
       
    def BERT(self,sent):
        tt = self.bt.tokenize("[CLS] "+sent+" [SEP]")
        it = self.bt.convert_tokens_to_ids(tt)

        with torch.no_grad():   encoded_layers, _ = self.bert(torch.tensor([it]), torch.tensor([[1]*len(tt)]))
        return torch.mean(encoded_layers[11], 1)[0].numpy()
    

    def calculate_metrics(self,y_test, predictions):
        matrix = confusion_matrix(y_test, predictions)
        tp = int(matrix[1][1])
        fn = int(matrix[1][0])
        fp = int(matrix[0][1])
        tn = int(matrix[0][0])
        
        reversed_matrix = np.array([[tp, fn], [fp, tn]]) 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            acc = accuracy_score(y_test, predictions)
            prec = precision_score(y_test, predictions, average='binary')
            rec = recall_score(y_test, predictions, average='binary')
            f1 = f1_score(y_test, predictions, average='binary')
        
        return acc, prec, rec, f1, reversed_matrix
    
    def train(self,idioms):
        X,Y = [],[]
        for idiom in idioms:
            X += self.x_train[idiom]
            Y += self.y_train[idiom]
        self.model.fit(X, Y)
        
    def test(self,idioms):
        X,Y = [],[]
        for idiom in idioms:
            X += self.x_test[idiom]
            Y += self.y_test[idiom]
        Y_pred = self.model.predict(X)
        _, prec, rec, f1, _ = self.calculate_metrics(Y, Y_pred)
        return [round(prec,2),round(rec,2),round(f1,2)]
    
    def evaluate(self,seen_idioms):
        seen_macro = round(stat.mean([self.test([idiom])[2] for idiom in seen_idioms]),2)
        seen_micro = self.test(seen_idioms)[2]
        unseen_idioms = [idiom for idiom in self.idioms if idiom not in seen_idioms]
        
        if len(unseen_idioms)==0: unseen_macro,unseen_micro = '-','-'
        else:
            unseen_macro = round(stat.mean([self.test([idiom])[2] for idiom in unseen_idioms]),2)
            unseen_micro = self.test(unseen_idioms)[2]
    
        net_macro = round(stat.mean([self.test([idiom])[2] for idiom in self.idioms]),2)
        net_micro = self.test(self.idioms)[2]
        return [seen_macro,seen_micro,unseen_macro,unseen_micro,net_macro,net_micro]

    def full(self):       
        self.train(self.idioms)
        #print('Parameters:',sum([a.size for a in self.model.coefs_]) +  sum([a.size for a in self.model.intercepts_]))

        print(self.evaluate(self.idioms))
        
    def loo(self):
       for i in range(len(self.idioms)):
           train_idioms = self.idioms[:i]+self.idioms[i+1:]
           self.train(train_idioms)
           print(self.idioms[i],", ",", ".join([str(a) for a in self.evaluate(train_idioms)]))
           #print(self.idioms[i], self.evaluate(train_idioms))
           
    def se(self):
        print('model, seen_macro, seen_micro, unseen_macro, unseen_micro, net_macro, net_micro')
        for i in range(len(self.idioms)):
           train_idioms = [self.idioms[i]] 
           self.train(train_idioms)
           print(self.idioms[i],", ",", ".join([str(a) for a in self.evaluate(train_idioms)]))
        
    def Unseen6_old(self):
        p = list(range(len(self.idioms)))
        Result = [0]*6
        count = 50
        #count=2
        for _ in range(count):
            random.shuffle(p)
            self.train([self.idioms[j] for j in p[:22]])
            result = self.evaluate([self.idioms[j] for j in p[:22]])
            for i,v in enumerate(result):   Result[i] += v
        for i in range(6):  Result[i] /= count
        print(Result)
        
    def Unseen6(self):
        p = list(range(len(self.idioms)))
        Macro,Micro = [],[]
        count = 50
        #count=2
        for _ in range(count):
            random.shuffle(p)
            self.train([self.idioms[j] for j in p[:22]])
            result = self.evaluate([self.idioms[j] for j in p[:22]])
            Macro.append(result[2])
            Micro.append(result[3])
        print('Micro',stat.mean(Micro),stat.stdev(Micro))
        print('Macro',stat.mean(Macro),stat.stdev(Macro))

        #for i in range(6):  Result[i] /= count
        #print(Result)

'''To print seen scores'''
def SeenScores():
    t0=time.time()   
    print('BERT-MlP')    
    gid = GITISota("../Data/ID/vnics_dataset_full_ratio-split.pkl",'idioms.txt',emb='bert')
    gid.full()
    print('Execution time:', time.time()-t0)  
    
    t0=time.time() 
    print('Skipthought-MLP')          
    gid = GITISota("../Data/ID/vnics_dataset_full_ratio-split.pkl",'idioms.txt',emb='cmb_skip')
    gid.full()
    print('Execution time:', time.time()-t0)      
        
    t0=time.time() 
    print('Skipthought-MLP')          
    gid = GITISota("../Data/ID/vnics_dataset_full_ratio-split.pkl",'idioms.txt',emb='cmb_skip',model='SVM')
    gid.full()
    print('Execution time:', time.time()-t0) 

'''To print unseen scores'''
def UnSeenScores():
    t0=time.time()   
    print('BERT-MlP')    
    gid = GITISota("../Data/ID/vnics_dataset_full_ratio-split.pkl",'idioms.txt',emb='bert')
    gid.Unseen6()
    print('Execution time:', time.time()-t0)  
    
    t0=time.time() 
    print('Skipthought-MLP')          
    gid = GITISota("../Data/ID/vnics_dataset_full_ratio-split.pkl",'idioms.txt',emb='cmb_skip')
    gid.Unseen6()
    print('Execution time:', time.time()-t0)      
        
    t0=time.time() 
    print('Skipthought-MLP')          
    gid = GITISota("../Data/ID/vnics_dataset_full_ratio-split.pkl",'idioms.txt',emb='cmb_skip',model='SVM')
    gid.Unseen6()
    print('Execution time:', time.time()-t0)      