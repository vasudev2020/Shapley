#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 09:02:37 2021

@author: vasu
"""
import torch
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel

class Embedding:
    def __init__(self,allembs=False):
        self.allembs = allembs
        
        self.bt = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert.eval()
         
        '''
        f = open('../Data/glove.840B.300d.txt','r')
        self.gloveModel = {}
        for line in f:
            splitLines = line.split()
            if len(splitLines)!=301:    continue
            word = splitLines[0]
            wordEmbedding = np.array([float(value) for value in splitLines[1:]])
            self.gloveModel[word] = wordEmbedding
        print(len(self.gloveModel)," words loaded!")
        '''
            
    
    def getMean(self,sent):
        emb = {}
        #embs = [self.gloveModel[w] for w in sent.split() if w in self.gloveModel]
        #emb['Glove'] = np.mean(embs,axis=0)
        
        tt = self.bt.tokenize("[CLS] "+sent+" [SEP]")
        if len(tt)>512: tt=tt[:512]
        it = self.bt.convert_tokens_to_ids(tt)

        with torch.no_grad():   encoded_layers, _ = self.bert(torch.tensor([it]), torch.tensor([[1]*len(tt)]))

        if self.allembs:
            for layer in range(12):
                emb['BERT'+str(layer)] = torch.mean(encoded_layers[layer], 1)[0].numpy()
        else:   emb['BERT11'] = torch.mean(encoded_layers[11], 1)[0].numpy()
            

        emb['Rand']=np.random.randn(768)
        
        return emb