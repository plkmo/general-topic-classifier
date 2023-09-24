# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:44:05 2019

@author: WT
"""
import os
import re
import copy
from unicodedata import name
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
import logging

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

def tokenize(sent, lowercase=True):
    sent = re.sub("([\"'?!])(\d+)([\"'?!])", r"\2", sent) #replace certain chars behind & infront of a number with the number itself
    sent = re.sub(r"( *['’])", "'", sent) # remove extra spaces from ' s          
    sent = re.sub('<[A-Z]+/*>', '', sent) # remove special tokens eg. <FIL/>, <S>
    sent = re.sub(r"[\*\"\\…\+\-\/\=\(\)‘•€\[\]\|♫:;—”“~`#\\]", " ", sent)
    sent = re.sub(' {2,}', ' ', sent) # remove extra spaces > 1
    sent = re.sub("^ +", "", sent) # remove space in front
    sent = re.sub(r"([\.\?,!]){1,}", " ", sent) # remove multiple puncs
    sent = sent.strip().split(' ')

    if lowercase:
        sent = [q.lower() for q in sent if (q not in ['', ' ', 'nil'])]
    return sent
        
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

def load_glove(file='../../data/topic_classification/glove.840B.300d.txt'):
    if not os.path.isfile('../../data/topic_classification/glove.pkl'):
        print("Extracting glove embeddings...")
        embeddings_dict = {}
        with open(file, 'r', encoding="utf-8") as f:
            for line in f:
                try:
                    values = line.split()
                    word = values[0].strip()
                    vector = np.asarray(values[1:], "float32")
                    assert len(vector) == 300
                    embeddings_dict[word] = vector
                except Exception as e:
                    print("Raw: ", values)
                    print("Unknown Word: ", values[0].strip())
                    print(values[1:])
                    continue
        
        print("Vocab: %d" % len(embeddings_dict))
        save_as_pickle('../../data/topic_classification/glove.pkl', embeddings_dict)
        print("Saved.")
    else:
        embeddings_dict = load_pickle('../../data/topic_classification/glove.pkl')
        print("Loaded glove embeddings")
    return embeddings_dict

def compile_vocab(args, df):
    vocab_counter = {}
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        tokens = row['text'] # list of word tokens
        for token in tokens:
            if token not in vocab_counter.keys():
                vocab_counter[token] = 1
            else:
                vocab_counter[token] += 1

    vocab_counter = {k: v for k, v in sorted(vocab_counter.items(), key=lambda item: item[1], reverse=True)}
    top_vocab = []
    for idx, (k, v) in enumerate(vocab_counter.items()):
        top_vocab.append(k)
        if (idx + 1) == args.max_vocab_len:
            break
    
    print("TOP VOCAB: ", top_vocab[:100])
    print("VOCAB LENGTH: ", len(top_vocab))
    return top_vocab

def convert_to_ids(x, vocab_dict):
    ids = []
    for t in x:
        if t.strip() in vocab_dict['v2idx'].keys():
            ids.append(vocab_dict['v2idx'][t.strip()])
        else:
            ids.append(1) # unknown oov token
    return ids

def convert_to_vocabs(x, vocab_dict):
    ids = []
    for t in x:
        if t in vocab_dict['idx2v'].keys():
            ids.append(vocab_dict['idx2v'][t])
        else:
            ids.append('<unk>') # unknown oov token
    return ids

def preprocess(args):
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

    logger.info("Tokenizing data...")
    ### tokenize data for BERT
    df_train.loc[:, "text"] = df_train["text"].apply(lambda x: tokenize(x))
    top_vocab = compile_vocab(args, df_train)
    vocab_dict = {}
    vocab_dict['v2idx'] = {word:idx for idx, word in enumerate(top_vocab, start=2)}
    del top_vocab
    vocab_dict['v2idx']['<pad>'] = 0
    vocab_dict['v2idx']['<unk>'] = 1
    vocab_dict['idx2v'] = {v:k for k, v in vocab_dict['v2idx'].items()}
    save_as_pickle(os.path.join(args.train_data, "vocab.pkl"), vocab_dict)
    print("Saved vocab to %s." % os.path.join(args.train_data, "vocab.pkl"))
    
    ### fill up reviews with [PAD] if word length less than tokens_length
    def filler(x, pad=0, length=tokens_length):
        dum = x
        while (len(dum) < length):
            dum.append(pad)
        return dum

    print("Converting tokens to idxs...")
    df_train.loc[:, "text"] = df_train["text"].apply(lambda x: convert_to_ids(x, vocab_dict))
    df_train.loc[:, "tokens_length"] = df_train["text"].apply(lambda x: len(x))
    df_train = df_train[df_train["tokens_length"] <= args.tokens_length][["text", "label"]]
    df_train.loc[:, "label"] = df_train["label"].apply(lambda x: labels_dict['l2idx'][x])
    
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
    logger.info("Done!")

    print("Testing integrity...")
    test = df_train.head()[['text', 'label']]
    test_text = test['text'].apply(lambda x: convert_to_vocabs(x, vocab_dict))
    test_labels = test['label'].apply(lambda x: labels_dict['idx2l'][x])
    print("Test text: ", test_text)
    print("Test labels: ", test_labels)

if __name__ == "__main__":
    embeddings_dict = load_glove(file='../../data/topic_classification/glove.840B.300d.txt')