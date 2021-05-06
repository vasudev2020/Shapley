#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 17:50:43 2021

@author: vasu
"""



import pandas as pd
#from functools import reduce

import matplotlib.pyplot as plt

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from collections import defaultdict

import spacy

from nltk.corpus import stopwords
path = './topicmodel/'


class TopicModel:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        #spacy.load('en', disable=['parser', 'ner'])
        #self.nlp = nlp
        self.stop_words = stopwords.words('english')
        self.stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
        
        
    '''Tokenize words and Clean-up text'''
    def sent_to_words(self,sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
    
    def lemmatization(self,texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = self.nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out
        
    def createDictCorpus(self,data):
        '''Tokenize words and Clean-up text'''
        data_words = list(self.sent_to_words(data))
        
        ''' Remove Stop Words from data_words'''
        data_words_nostops = [[word for word in simple_preprocess(str(doc)) if word not in self.stop_words] for doc in data_words]
        
        ''' Form Bigrams'''
        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
        self.bigram_mod = gensim.models.phrases.Phraser(bigram)
        data_words_bigrams = [self.bigram_mod[doc] for doc in data_words_nostops]
        
        '''Lemmatize bigrams'''
        data_lemmatized = self.lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        
        '''create dictionary and corpus'''
        self.id2word = corpora.Dictionary(data_lemmatized)
        corpus = [self.id2word.doc2bow(text) for text in data_lemmatized]

        self.tfidfmodel = gensim.models.TfidfModel(corpus)
        
        corpus = self.tfidfmodel[corpus]
        
        return corpus,data_lemmatized
        
    def train(self,data,num_topics):
        corpus,data_lemmatized = self.createDictCorpus(data)
        self.lsi_model = gensim.models.LsiModel(corpus=corpus, num_topics=num_topics, id2word=self.id2word)
        
    def topicModel(self,data):
                
        '''Tokenize words and Clean-up text'''
        data_words = list(self.sent_to_words(data))
        #print(data_words[:1])
        
        ''' Remove Stop Words from data_words'''
        data_words_nostops = [[word for word in simple_preprocess(str(doc)) if word not in self.stop_words] for doc in data_words]
        data_words_bigrams = [self.bigram_mod[doc] for doc in data_words_nostops]
        data_lemmatized = self.lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        corpus = [self.id2word.doc2bow(text) for text in data_lemmatized]
        
        #TODO: confirm this
        corpus = self.tfidfmodel[corpus]


        Topics = []
        #for row in self.lda_model[corpus]:
        for row in self.lsi_model[corpus]:
            row = sorted(row, key=lambda x: (x[1]), reverse=True)  
            if len(row)==0: continue
            topic_num, prop_topic = row[0] # Get the Dominant topic, Perc Contribution and Keywords for each document
            Topics.append(topic_num)
    
        return Topics
    
    def save(self):
        #self.lda_model.save(path+'lda.model')
        self.lsi_model.save(path+'lsi.model')
        self.tfidfmodel.save(path+'tfidf.model')
        self.bigram_mod.save(path+'bigram.model')
        self.id2word.save(path+'id2word.dat')
        
    def load(self):
        self.id2word = corpora.Dictionary.load(path+'id2word.dat')
        self.bigram_mod = gensim.models.phrases.Phraser.load(path+'bigram.model')
        self.tfidfmodel = gensim.models.TfidfModel.load(path+'tfidf.model')
        #self.lda_model = gensim.models.LdaMulticore.load(path+'lda.model')
        self.lsi_model = gensim.models.LsiModel.load(path+'lsi.model')
        

