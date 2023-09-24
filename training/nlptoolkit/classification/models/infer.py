# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:33:49 2019

@author: plkmo
"""
import re
import pickle
import os
import pandas as pd
import torch
from tqdm import tqdm
import logging

tqdm.pandas(desc="prog-bar")
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def load_pickle(filename):
    completeName = filename
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def clean(sent):
    sent = re.sub('\d+[)]', '', sent)
    sent = re.sub(r"( *['’])", "'", sent) # remove extra spaces from ' s          
    sent = re.sub('<[A-Z]+/*>', '', sent) # remove special tokens eg. <FIL/>, <S>
    sent = re.sub(r"[\*\"\\…\+\-\/\=\(\)‘•€\[\]\|♫:;—”“~`#\\]", " ", sent)
    sent = re.sub(' {2,}', ' ', sent) # remove extra spaces > 1
    sent = re.sub("^ +", "", sent) # remove space in front
    sent = re.sub(r"([\.\?,!]){1,}", r"\1", sent).strip().strip('\n') # remove multiple puncs
    return sent

def filler(x, pad=0, length=256):
    dum = x
    while (len(dum) < length):
        dum.append(pad)
    return dum

class infer_from_trained(object):
    def __init__(self, args=None):
        if args is None:
            self.args = load_pickle("args.pkl")
        else:
            self.args = args
        self.cuda = torch.cuda.is_available()
        logger.info("Loading tokenizer and model...")
        if self.args.model_no == 1:
            from transformers import BertTokenizer as model_tokenizer
            from transformers import BertForSequenceClassification as net
            from .BERT.train_funcs import load_state
            model_type = 'bert-base-uncased' if (len(self.args.model_size) == 0) else self.args.model_size
            self.lower_case = True
            
        elif self.args.model_no == 2:
            from .XLNet.tokenization_xlnet import XLNetTokenizer as model_tokenizer
            from .XLNet.XLNet import XLNetForSequenceClassification as net
            from .XLNet.train_funcs import load_state
            model_type = 'xlnet-base-cased' if (len(self.args.model_size) == 0) else self.args.model_size
            self.lower_case = False
            
        elif self.args.model_no == 4:
            from .ALBERT.tokenization_albert import AlbertTokenizer as model_tokenizer
            from .ALBERT.ALBERT import AlbertForSequenceClassification as net
            from .ALBERT.train_funcs import load_state
            model_type = 'albert-base-v2' if (len(self.args.model_size) == 0) else self.args.model_size
            self.lower_case = False
            
        elif self.args.model_no == 5:
            from .XLMRoBERTa.tokenization_xlm_roberta import XLMRobertaTokenizer as model_tokenizer
            from .XLMRoBERTa.XLMRoBERTa import XLMRobertaForSequenceClassification as net
            from .XLMRoBERTa.train_funcs import load_state
            model_type = 'xlm-roberta-base' if (len(self.args.model_size) == 0) else self.args.model_size
            self.lower_case = False

        elif self.args.model_no == 7:
            from transformers import DebertaTokenizer as model_tokenizer
            from transformers import DebertaForSequenceClassification as net
            from .DEBERTA.train_funcs import load_state
            model_type = 'microsoft/deberta-base'
            self.lower_case = True

        elif self.args.model_no == 8:
            from transformers import ConvBertTokenizer as model_tokenizer
            from transformers import ConvBertForSequenceClassification as net
            from .ConvBERT.train_funcs import load_state
            model_type = 'YituTech/conv-bert-base'
            self.lower_case = True
        
        elif self.args.model_no == 10:
            from transformers import DistilBertTokenizer as model_tokenizer
            from transformers import DistilBertForSequenceClassification as net
            from .DistilBERT.train_funcs import load_state
            model_type = 'distilbert-base-uncased'
            self.lower_case = True
        
        if os.path.isfile(os.path.join(self.args.train_data, 'tokenizer.pkl')):
            self.tokenizer = load_pickle(os.path.join(self.args.train_data, 'tokenizer.pkl'))
        elif os.path.isfile("/opt/ml/model/tokenizer.pkl"):
            print("Loading tokenizer from /opt/ ")
            with open("/opt/ml/model/tokenizer.pkl", 'rb') as pkl_file:
                self.tokenizer = pickle.load(pkl_file)
        else:
            print("Using default tokenizer.")
            self.tokenizer = model_tokenizer.from_pretrained(model_type)
        self.tokens_length = args.tokens_length # max tokens length
        
        self.labels_dict = load_pickle(os.path.join(self.args.train_data, "labels.pkl"))
        self.num_classes = len(self.labels_dict['l2idx'].keys())
        print("NUM CLASSES: %d\n%s" % (self.num_classes, str(self.labels_dict['l2idx'].keys())))
        
        self.net = net.from_pretrained(model_type, num_labels=self.num_classes)
        
        if os.path.isfile(os.path.join(args.train_data, 'tokenizer.pkl')) or os.path.isfile("/opt/ml/model/tokenizer.pkl"):
            if self.args.model_no not in [8, 10]:
                self.net.resize_token_embeddings(len(self.tokenizer))

        if self.cuda:
            self.net.cuda()
        _, _ = load_state(self.net, None, None, args, load_best=True) 
        logger.info("Done!")

    def infer_sentence(self, sentence):
        self.net.eval()
        if self.lower_case:
            sentence = sentence.lower()
        
        if self.args.model_no in [8, 10]:
            sentence = self.tokenizer(sentence, return_tensors="pt")['input_ids'].tolist()[0][:(self.args.tokens_length-1)]
        else:
            sentence = self.tokenizer.tokenize("[CLS] " + sentence)
            sentence = self.tokenizer.convert_tokens_to_ids(sentence[:(self.args.tokens_length-1)] + ["[SEP]"])
        sentence = torch.tensor(sentence).unsqueeze(0)
        type_ids = torch.zeros([sentence.shape[0], sentence.shape[1]], requires_grad=False).long()
        src_mask = (sentence != 0).long()
        if self.cuda:
            sentence = sentence.cuda()
            type_ids = type_ids.cuda()
            src_mask = src_mask.cuda()
        if self.args.model_no == 1:
            outputs = self.net(sentence, token_type_ids=type_ids, attention_mask=src_mask)
        elif self.args.model_no in [4, 5]:
            outputs = self.net(sentence, token_type_ids=type_ids, attention_mask=src_mask)
            outputs = outputs[0]
        elif self.args.model_no in [8]:
            outputs = self.net(sentence, token_type_ids=type_ids, attention_mask=src_mask)
        elif self.args.model_no == 10:
             outputs = self.net(sentence, attention_mask=src_mask)
        else:
            outputs, _ = self.net(sentence, token_type_ids=type_ids, attention_mask=src_mask)
            
        confidence = torch.softmax(outputs.logits.data, 1)
        confidence, predicted = torch.max(confidence, 1)
        predicted = predicted.cpu().item() if self.cuda else predicted.item()
        confidence = confidence.cpu().item() if self.cuda else confidence.item()
        predicted = self.labels_dict['idx2l'][predicted]
        #print("Predicted class: %d" % predicted)
        return confidence, predicted

    def infer_sentences(self, sentences, batch_size=32):
        self.net.eval()
        processed_sentences = []
        processed_type_ids = []
        processed_src_masks = []
        preds, confidences = [], []
        print("Collecting sentences...")
        for sentence in tqdm(sentences, total=len(sentences)):
            if self.lower_case:
                sentence = sentence.lower()

            if self.args.model_no in [8, 10]:
                sentence = self.tokenizer(sentence, return_tensors="pt")['input_ids'].tolist()[0][:(self.args.tokens_length-1)]
            else:
                sentence = self.tokenizer.tokenize("[CLS] " + sentence)
                sentence = self.tokenizer.convert_tokens_to_ids(sentence[:(self.args.tokens_length-1)] + ["[SEP]"])
            sentence = filler(sentence, pad=0, length=self.args.tokens_length)
            sentence = torch.tensor(sentence)
            type_ids = torch.zeros([sentence.shape[0]], requires_grad=False).long()
            src_mask = (sentence != 0).long()
            processed_sentences.append(sentence)
            processed_type_ids.append(type_ids)
            processed_src_masks.append(src_mask)

        sentence_ = torch.stack(processed_sentences, dim=0)
        type_ids_ = torch.stack(processed_type_ids, dim=0)
        src_mask_ = torch.stack(processed_src_masks, dim=0)
        
        print("Inferring...")
        with torch.no_grad():
            for batch_idx in tqdm(range(src_mask_.shape[0]//batch_size + 1), total=src_mask_.shape[0]//batch_size + 1):
                if self.cuda:
                    sentence = sentence_[batch_idx*batch_size:(batch_idx + 1)*batch_size].cuda()
                    type_ids = type_ids_[batch_idx*batch_size:(batch_idx + 1)*batch_size].cuda()
                    src_mask = src_mask_[batch_idx*batch_size:(batch_idx + 1)*batch_size].cuda()
                else:
                    sentence = sentence_[batch_idx*batch_size:(batch_idx + 1)*batch_size]
                    type_ids = type_ids_[batch_idx*batch_size:(batch_idx + 1)*batch_size]
                    src_mask = src_mask_[batch_idx*batch_size:(batch_idx + 1)*batch_size]
                
                if sentence.shape[0] > 0:
                    if self.args.model_no == 1:
                        outputs = self.net(sentence, token_type_ids=type_ids, attention_mask=src_mask)
                    elif self.args.model_no in [4, 5]:
                        outputs = self.net(sentence, token_type_ids=type_ids, attention_mask=src_mask)
                        outputs = outputs[0]
                    elif self.args.model_no in [8]:
                        outputs = self.net(sentence, token_type_ids=type_ids, attention_mask=src_mask)
                    else:
                        outputs, _ = self.net(sentence, token_type_ids=type_ids, attention_mask=src_mask)
                    confidence = torch.softmax(outputs.logits.data, 1)
                    confidence, predicted = torch.max(confidence, 1)
                    predicted = list(predicted.cpu().numpy()) if self.cuda else list(predicted.numpy())
                    confidence = list(confidence.cpu().numpy()) if self.cuda else list(confidence.numpy())
                    predicted = [self.labels_dict['idx2l'][pred] for pred in predicted]
                    preds.extend(predicted)
                    confidences.extend(confidence)
        return confidences, preds
    
    def infer_from_input(self):
        self.net.eval()
        while True:
            user_input = input("Type input sentence (Type \'exit' or \'quit' to quit):\n")
            if user_input in ["exit", "quit"]:
                break
            confidence, predicted = self.infer_sentence(user_input)
        return confidence, predicted
    
    def infer_from_file(self, in_file="./data/input.txt", out_file="./data/output.txt"):
        if in_file is None:
            df = pd.read_csv(self.args.infer_data)
            df['labels'] = df.progress_apply(lambda x: self.infer_sentence(x['text']), axis=1)
        else:
            df = pd.read_csv(in_file, header=None, names=["sents"])
            df['labels'] = df.progress_apply(lambda x: self.infer_sentence(x['sents']), axis=1)
        df.to_csv(out_file, index=False)
        logger.info("Done and saved as %s!" % out_file)
        return
            
    def infer_from_dir(self, infer_path = "../../data/reddit/classification/for_inference", batch_size=32):
        if not os.path.isdir(os.path.join(infer_path, 'results')):
            os.mkdir(os.path.join(infer_path, 'results'))

        logger.info("Preparing data for inference...")
        for file in os.listdir(infer_path):
            file_name = infer_path + '/' + file
            print(file_name)
            if '.csv' in file_name:
                df = pd.read_csv(file_name)
                convos = []
                for idx, row in df.iterrows():
                    convo = re.split(r'[\d+]\)', re.sub('(\(.+\))', '', str(row['conversation'])))
                    convo = [clean(s) for s in convo if (len(s) > 1)]
                    
                    assert len(convo) > 1
                    convo = ' [SEP] '.join(convo).strip()
                    convos.append(convo)
                
                confidences, preds = self.infer_sentences(sentences=convos, batch_size=batch_size)

                print("LENGTH DATA, PREDS: %d, %d" % (len(df), len(preds)))
                df.loc[:, "predicted_label"] = preds
                df.loc[:, "confidence"] = confidences
                df.to_csv(infer_path + '/results/result_%s' % file, columns=df.columns, index=False)
                print("Saved to %s" % infer_path + '/results/result_%s' % file)
        return