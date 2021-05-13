#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 13:20:17 2021

@author: vasu
"""

from collections import defaultdict
#import spacy
#from ShapleyAnalyser import ShapleyAnalyser
from TopicModel import TopicModel
from statistics import mean
from scipy.stats import pearsonr,spearmanr,entropy
from tqdm import tqdm
from scipy.special import rel_entr


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


traintext = LoadVNCTexts()
Data,Labels = LoadVNC(5) 
TM = TopicModel()
exps = [exp for exp in Data]


def getTopicDistSim(T,opt):
    TM.train(traintext,T)
    
    testtext = [sent for exp in Data for sent in Data[exp]]
    topics = TM.topicModel(testtext)
    topic_dist = [0]*T
    for t in topics:    topic_dist[t]+=1
    topic_dist = [a/sum(topic_dist) for a in topic_dist]

    dist_sim = []
    for exp in exps:
        if opt=='all':  exp_topics = TM.topicModel([sent for sent in Data[exp]])
        if opt=='idiom':    exp_topics = TM.topicModel([sent for sent,lb in zip(Data[exp],Labels[exp]) if lb==1])
        if opt=='literal':    exp_topics = TM.topicModel([sent for sent,lb in zip(Data[exp],Labels[exp]) if lb==0])

        exp_topic_dist = [0]*T
        for t in exp_topics:    exp_topic_dist[t]+=1
        exp_topic_dist = [a/sum(exp_topic_dist) for a in exp_topic_dist]

        kld = sum(rel_entr(exp_topic_dist,topic_dist))
        dist_sim.append(kld)
    return dist_sim

def getCorr(a,b):
    assert len(a)==len(b)
    return round(pearsonr(a,b)[0],4), round(spearmanr(a,b)[0],4)


def getMeanTopicDistSim(opt):
    td =[getTopicDistSim(T,opt) for T in range(10,100,10)]
    mtd = [mean(a) for a in zip(*td)]
    return mtd
    
sv = [0.0293,0.0419,0.0534,0.0346,0.0374,0.0283,0.0342,0.0335,0.0365,0.0418,0.0458,0.0401,0.0349,0.0209,0.0288,0.0345,0.0338,0.0436,0.0346,0.0280,0.0353,0.0466,0.0449,0.0201,0.0368,0.0359]

s = [getMeanTopicDistSim('all') for _ in range(5)]
ms = [mean(a) for a in zip(*s)]
print('All')
print(ms)
print(getCorr(sv,ms))

s = [getMeanTopicDistSim('idiom') for _ in range(5)]
ms = [mean(a) for a in zip(*s)]
print('Idiom')
print(ms)
print(getCorr(sv,ms))

s = [getMeanTopicDistSim('literal') for _ in range(5)]
ms = [mean(a) for a in zip(*s)]
print('Literal')
print(ms)
print(getCorr(sv,ms))