# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:44:05 2019

@author: WT
"""
import os
import copy
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DebertaTokenizer
from tqdm import tqdm
import logging

tqdm.pandas()

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def load_pickle(filename):
    completeName = filename
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def save_as_pickle(filename, data):
    completeName = filename
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)
        
### remove stopwords and non-words from tokens list
def filter_tokens(tokens, stopwords):
    tokens1 = []
    for token in tokens:
        token = token.lower()
        if (token not in stopwords) and (token not in [".",",",";","&","'s", ":", "?", "!","(",")",\
            "'","'m","'no","***","--","...","[","]"]):
            tokens1.append(token)
    return tokens1

def dummy_fun(doc):
    return doc

def preprocess(args, lower_case=True):
    logger.info("Preprocessing data...")
    counter = 0
    for file in os.listdir(args.train_data):
        if '.csv' in file:
            print("Reading file %s" % file)
            if counter == 0:
                df_train = pd.read_csv(os.path.join(args.train_data, file))
            else:
                df_train = df_train.append(pd.read_csv(os.path.join(args.train_data, file)), ignore_index=True)
            counter += 1
    
    print("Loaded %d rows of data." % len(df_train))
    tokens_length = args.tokens_length # max tokens length

    # limit to max examples per class
    print("Limiting to max %d examples per class..." % args.max_examples_per_class)
    for i, label in enumerate(df_train['label'].unique()):
        if i == 0:
            df_dum = df_train[df_train['label'] == label]
            df_train_ = df_dum.sample(n=min(len(df_dum), args.max_examples_per_class), random_state=1)
        else:
            df_dum = df_train[df_train['label'] == label]
            df_train_ = df_train_.append(df_dum.sample(n=min(len(df_dum), args.max_examples_per_class), random_state=1), ignore_index=True)
    
    df_train = copy.deepcopy(df_train_); del df_train_
    print("Total number of classes: %d" % len(df_train['label'].unique()))
    print("Classes: ", df_train['label'].unique())
    labels_dict = {'l2idx': {}}
    for idx, c in enumerate(df_train['label'].unique()):
        labels_dict['l2idx'][c] = idx
    labels_dict['idx2l'] = {v:k for k,v in labels_dict['l2idx'].items()}
    save_as_pickle(os.path.join(args.train_data, "labels.pkl"), labels_dict)
    
    tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
    tokens_length = args.tokens_length # max tokens length
    
    logger.info("Tokenizing data...")
    ### tokenize data for DEBERTA
    df_train.loc[:, "text"] = df_train["text"].progress_apply(lambda x: tokenizer(x, return_tensors="pt")['input_ids'].tolist()[0])
    df_train.loc[:, "tokens_length"] = df_train["text"].progress_apply(lambda x: len(x))
    df_train = df_train[df_train["tokens_length"] <= args.tokens_length][["text", "label"]]
    df_train.loc[:, "label"] = df_train["label"].progress_apply(lambda x: labels_dict['l2idx'][x])
    
    ### fill up reviews with [PAD] if word length less than tokens_length
    def filler(x, pad=0, length=tokens_length):
        dum = x
        while (len(dum) < length):
            dum.append(pad)
        return dum
    
    logger.info("Padding sequences...")
    df_train.loc[:, "text"] = df_train["text"].apply(lambda x: filler(x))
    df_train.loc[:, "fills"] = df_train["text"].apply(lambda x: x.count(0))
    
    print("Train test split...")
    df_train, df_test = train_test_split(df_train, test_size=0.1, random_state=0, stratify=df_train[['label']])
    print("Train data: %d" % len(df_train))
    print("Test data: %d" % len(df_test))
    print(df_train.head())
    print(df_test.head())
    
    logger.info("Saving..")
    df_train.to_pickle(os.path.join(args.train_data, "train_processed.pkl"))
    df_test.to_pickle(os.path.join(args.train_data, "infer_processed.pkl"))
    save_as_pickle(os.path.join(args.train_data, 'tokenizer.pkl'), tokenizer)

    try:
        with open("/opt/ml/model/tokenizer.pkl", 'wb') as output:
            pickle.dump(tokenizer, output)
    except Exception as e:
        print("ERROR SAVING TOKENIZER!! ", e)
    logger.info("Done!")