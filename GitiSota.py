#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 18:07:29 2021

@author: vasu
"""
import torch
import pickle

from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from pytorch_pretrained_bert import BertTokenizer, BertModel

def Evaluate(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, average='binary')
    rec = recall_score(y_test, predictions, average='binary')
    f1 = f1_score(y_test, predictions, average='binary')
    
    return [acc,prec,rec,f1]

def BERT(sent):
    tt = bt.tokenize("[CLS] "+sent+" [SEP]")
    it = bt.convert_tokens_to_ids(tt)

    with torch.no_grad():   encoded_layers, _ = bert(torch.tensor([it]), torch.tensor([[1]*len(tt)]))
    return torch.mean(encoded_layers[11], 1)[0].numpy()

svm = LinearSVC()
mlp = MLPClassifier()

dataset = pickle.load(open("../Data/ID/vnics_dataset_full_ratio-split.pkl", "rb"), encoding='latin1')
train_data = dataset["train_sample"]
test_data = dataset["test_sample"]

X_train = [p['cmb_skip'] for p in dataset["train_sample"]]
Y_train = [p['lab_int'] for p in dataset["train_sample"]]
X_test = [p['cmb_skip'] for p in dataset["test_sample"]]
Y_test = [p['lab_int'] for p in dataset["test_sample"]]

print('Skipthought-SVM',' '.join(Evaluate(svm, X_train, Y_train, X_test, Y_test)))
print('Skipthought-MLP',' '.join(Evaluate(mlp, X_train, Y_train, X_test, Y_test)))

bt = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')
bert.eval()
       
X_train = [BERT(p['sent']) for p in dataset["train_sample"]]
Y_train = [p['lab_int'] for p in dataset["train_sample"]]
X_test = [BERT(p['sent']) for p in dataset["test_sample"]]
Y_test = [p['lab_int'] for p in dataset["test_sample"]]


print('BERT-SVM',' '.join(Evaluate(svm, X_train, Y_train, X_test, Y_test)))
print('BERT-MLP',' '.join(Evaluate(mlp, X_train, Y_train, X_test, Y_test)))

