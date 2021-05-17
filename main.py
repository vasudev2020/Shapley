#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 15:38:01 2021

@author: vasu
"""
from newGITI import GITI
from TopicModel import TopicModel

from collections import defaultdict
from scipy.special import rel_entr
from statistics import mean
import argparse
from scipy.stats import pearsonr,spearmanr


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

def getILTopicDistSim(T):
    TM.train(traintext,T)

    dist_sim = []
    for exp in exps:
        idiom_exp_topics = TM.topicModel([sent for sent,lb in zip(Data[exp],Labels[exp]) if lb==1])
        literal_exp_topics = TM.topicModel([sent for sent,lb in zip(Data[exp],Labels[exp]) if lb==0])

        idiom_exp_topic_dist = [0.000001]*T
        for t in idiom_exp_topics:    idiom_exp_topic_dist[t]+=1
        idiom_exp_topic_dist = [a/sum(idiom_exp_topic_dist) for a in idiom_exp_topic_dist]

        literal_exp_topic_dist = [0.000001]*T
        for t in literal_exp_topics:    literal_exp_topic_dist[t]+=1
        literal_exp_topic_dist = [a/sum(literal_exp_topic_dist) for a in literal_exp_topic_dist]

        kld = sum(rel_entr(idiom_exp_topic_dist,literal_exp_topic_dist))
        dist_sim.append(kld)
    return dist_sim

def getMeanILTopicDistSim():
    td =[getILTopicDistSim(T) for T in range(50,100,10)]
    mtd = [mean(a) for a in zip(*td)]
    return mtd

def getTopicDistSim(T):
    TM.train(traintext,T)
    
    testtext = [sent for exp in Data for sent in Data[exp]]
    topics = TM.topicModel(testtext)
    topic_dist = [0]*T
    for t in topics:    topic_dist[t]+=1
    topic_dist = [a/sum(topic_dist) for a in topic_dist]

    dist_sim = []
    for exp in exps:
        exp_topics = TM.topicModel([sent for sent in Data[exp]])
        
        exp_topic_dist = [0]*T
        for t in exp_topics:    exp_topic_dist[t]+=1
        exp_topic_dist = [a/sum(exp_topic_dist) for a in exp_topic_dist]

        kld = sum(rel_entr(exp_topic_dist,topic_dist))
        dist_sim.append(kld)
    return dist_sim

def getMeanTopicDistSim():
    td =[getTopicDistSim(T) for T in range(10,100,10)]
    mtd = [mean(a) for a in zip(*td)]
    return mtd

def getCorr(a,b):
    assert len(a)==len(b)
    return round(pearsonr(a,b)[0],4), round(spearmanr(a,b)[0],4)
        
Data,Labels = LoadVNC(5) 
sv = [0.0293,0.0419,0.0534,0.0346,0.0374,0.0283,0.0342,0.0335,0.0365,0.0418,0.0458,0.0401,0.0349,0.0209,0.0288,0.0345,0.0338,0.0436,0.0346,0.0280,0.0353,0.0466,0.0449,0.0201,0.0368,0.0359]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='iltopic', help='options:topic/unseen/SE/LOO/Difficulty/Shapley')
    args=parser.parse_args() 

    if args.opt=='topic':
        traintext = LoadVNCTexts()
        TM = TopicModel()
        exps = [exp for exp in Data]
        s = [getMeanTopicDistSim() for _ in range(10)]
        ms = [mean(a) for a in zip(*s)]
        print(ms)

    if args.opt=='iltopic':
        traintext = LoadVNCTexts()
        TM = TopicModel()
        exps = [exp for exp in Data]
        s = [getMeanILTopicDistSim() for _ in range(10)]
        ms = [mean(a) for a in zip(*s)]
        print(ms)
        print(getCorr(sv,ms))
        
    if args.opt=='unseen':
        giti=GITI(Data,Labels)
        sv_order = ['blow whistle','pull plug','hit wall','pull weight','make mark','blow trumpet','hit roof','hold fire','find foot','take heart','hit road','take root','pull leg','kick heel','cut figure','make pile','make hay','get wind','make hit','have word','blow top','make face','get sack','make scene','lose head','see star']
        fx_order = ['blow whistle', 'take root', 'kick heel', 'blow trumpet', 'blow top', 'find foot', 'pull weight', 'get sack', 'pull plug', 'hit road', 'hit roof', 'make hay', 'make mark', 'lose head', 'cut figure', 'hit wall', 'see star', 'make hit', 'pull leg', 'take heart', 'get wind', 'hold fire', 'make scene', 'make pile', 'make face', 'have word']
        size_order = ['take root', 'have word', 'make mark', 'take heart', 'blow whistle', 'pull plug', 'hit wall', 'see star', 'find foot', 'pull leg', 'get sack', 'make scene', 'cut figure', 'make face', 'lose head', 'kick heel', 'pull weight', 'hit road', 'blow trumpet', 'blow top', 'get wind', 'make pile', 'hold fire', 'hit roof', 'make hay', 'make hit']
        iltopicdiv_order = ['take root', 'get sack', 'make hay', 'make pile', 'cut figure', 'make hit', 'hit wall', 'hit roof', 'hold fire', 'lose head', 'kick heel', 'blow trumpet', 'pull weight', 'blow top', 'make mark', 'make scene', 'see star', 'pull leg', 'pull plug', 'have word', 'hit road', 'take heart', 'find foot', 'get wind', 'blow whistle', 'make face']
        topicdiv_order = ['take root', 'pull plug', 'make mark', 'make pile', 'hit roof', 'make scene', 'have word', 'hit wall', 'take heart', 'blow whistle', 'blow top', 'blow trumpet', 'make hay', 'hit road', 'make hit', 'pull leg', 'cut figure', 'get sack', 'pull weight', 'see star', 'make face', 'kick heel', 'lose head', 'get wind', 'hold fire', 'find foot']
        giti.getUnseenExpResults([sv_order,fx_order,size_order,iltopicdiv_order,topicdiv_order])
    
    if args.opt=='SE':
        giti=GITI(Data,Labels)
        giti.getSE()
        
    if args.opt=='LOO':
        giti.getLOO()
        
    if args.opt=='Difficulty':
        giti.getDifficulty()
        
    if args.opt=='Shapley':
        sv = giti.getShapleyValues()   
        for exp in Data:  
            print(exp,sv[exp],len(Labels[exp]),sum(Labels[exp]),len(Labels[exp])-sum(Labels[exp]))
