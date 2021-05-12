#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 15:38:01 2021

@author: vasu
"""
from GITI import GITI
from collections import defaultdict

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
        
Data,Labels = LoadVNC(5) 
giti=GITI(Data,Labels)

sv_order = ['blow whistle','pull plug','hit wall','pull weight','make mark','blow trumpet','hit roof','hold fire','find foot','take heart','hit road','take root','pull leg','kick heel','cut figure','make pile','make hay','get wind','make hit','have word','blow top','make face','get sack','make scene','lose head','see star']
se_order = ['blow whistle','hold fire','pull plug','pull weight','make mark','hit wall','blow trumpet','make pile','hit roof','kick heel','hit road','make hay','pull leg','take heart','get wind','make hit','take root','cut figure','find foot','get sack','blow top','have word','make scene','lose head','make face','see star']
lo_order = ['see star','blow whistle','hit wall','make face','have word','make scene','make mark','blow top','get wind','pull leg','make pile','make hay','hold fire','make hit','take root','get sack','pull plug','lose head','take heart','hit road','hit roof','cut figure','blow trumpet','pull weight','find foot','kick heel']
giti.getUnseenExpResults(lo_order,se_order,sv_order)

'''
giti.getSE()

giti.getLOO()

giti.getDifficulty()

sv = giti.getShapleyValues()   
for exp in Data:  
    print(exp,sv[exp],len(Labels[exp]),sum(Labels[exp]),len(Labels[exp])-sum(Labels[exp]))
'''