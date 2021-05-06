#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 17:54:58 2021

@author: vasu
"""
from collections import defaultdict
import spacy
from ShapleyAnalyser import ShapleyAnalyser
from TopicModel import TopicModel
from statistics import mean

def LoadVNC(cutoff):        
    with open("../Data/ID/cook_dataset.tok", "r") as fp:
        dataset = fp.readlines()
    dataset = [d.strip().split('||') for d in dataset]
    dataset = [[1 if d[0]=='I' else 0, d[1]+' '+d[2], d[3].replace(' &apos;','\'').lower()] for d in dataset if d[0]!='Q']
    
    Datadict = defaultdict(lambda: defaultdict(list))
    
    Data = defaultdict(list)
    Labels = defaultdict(list)
    
    for label,exp,sent in dataset:  Datadict[exp][label].append(sent)

    print('Total number of expressions:',len(Datadict))
    
    for exp in Datadict.keys():    
        if len(Datadict[exp][0])>=cutoff and len(Datadict[exp][1])>=cutoff:
            for sent in Datadict[exp][0]:   
                Data[exp].append(sent)
                Labels[exp].append(0)
            for sent in Datadict[exp][1]:   
                Data[exp].append(sent)
                Labels[exp].append(1)
                  
    print('Number of expressions after filtering:',len(Data))
    
    assert len(Data)==len(Labels)

    return Data,Labels

def LoadVNCTexts():
    with open("../Data/ID/cook_dataset.tok", "r") as fp:
        dataset = fp.readlines()
    dataset = [d.strip().split('||')[3].replace(' &apos;','\'').lower() for d in dataset]
    return dataset
    

Data,Labels = LoadVNC(5) 
text = LoadVNCTexts() 

TM = TopicModel()  
TC=defaultdict(list)  
for T in range(5,50,5):
    TM.train(text,T)
    for exp in Data:  
        topics = TM.topicModel(Data[exp])
        TC[exp].append(len(set(topics))/T)

SA = ShapleyAnalyser(Data,Labels)

SA.getSE()
SA.getLOO()
SA.getDifficulty()

sv = SA.getShapleyValues()

#TM.train([sent for exp in Data for sent in Data[exp]],T)
for exp in Data:  
    topics = TM.topicModel(Data[exp])
    print(exp,sv[exp],len(Labels[exp]),sum(Labels[exp]),len(Labels[exp])-sum(Labels[exp]),mean(TC[exp]))

