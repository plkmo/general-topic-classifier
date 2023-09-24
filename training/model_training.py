from argparse import ArgumentParser
import copy
import pickle
import numpy as np
import pandas as pd
import torch
import os
import json
from transformers import Trainer, TrainingArguments
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import ConvBertTokenizer, ConvBertForSequenceClassification
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

try:
    print("Setting GPU no. as default...")
    torch.cuda.set_device(3)
    print("GPU: %d" % torch.cuda.current_device())
except Exception as e:
    print("Unable to set GPU. ", e)
    if torch.cuda.is_available():
        print("GPU: %d" % torch.cuda.current_device())

def compute_metrics(pred):
    # 'politics & government': 12
    # 'religion': 10
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    politics_precision, politics_recall, politics_f1, _ = precision_recall_fscore_support([int(i == 12) for i in labels], \
                                                        [int(i == 12) for i in preds], average='binary')
    religion_precision, religion_recall, religion_f1, _ = precision_recall_fscore_support([int(i == 10) for i in labels], \
                                                        [int(i == 10) for i in preds], average='binary')
    business_n_finance_precision, business_n_finance_recall, business_n_finance_f1, _ = precision_recall_fscore_support([int(i == 13) for i in labels], \
                                                        [int(i == 13) for i in preds], average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'politics_f1': politics_f1,
        'politics_precision': politics_precision,
        'politics_recall': politics_recall,
        'religion_f1': religion_f1,
        'religion_precision': religion_precision,
        'religion_recall': religion_recall,
        'business_n_finance_f1': business_n_finance_f1,
        'business_n_finance_precision': business_n_finance_precision,
        'business_n_finance_recall': business_n_finance_recall
    }

def str2bool(obj):
    if obj is None:
        return False
    elif isinstance(obj, bool):
        return obj
    elif isinstance(obj, str):
        if obj.lower() in ['true', 't', 'yes', 'y', '1']:
            return True
        elif obj.lower() in ['false', 'f', 'no', 'n', '0']:
            return False
    else:
        return True

def save_as_pickle(filename, data):
    completeName = filename
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

def load_data(data_folder, args=None, pad=True, max_len=256):
    counter = 0
    for file in os.listdir(data_folder):
        if '.csv' in file:
            print("Reading file %s" % file)
            if counter == 0:
                df_train = pd.read_csv(os.path.join(data_folder, file))
            else:
                df_train = df_train.append(pd.read_csv(os.path.join(data_folder, file)), ignore_index=True)
            counter += 1
    
    df_train.dropna(inplace=True)
    print("Loaded %d valid rows of data." % len(df_train))
    
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
    print("Class distribution: ", df_train['label'].value_counts())
    labels_dict = {'l2idx': {'entertainment & music': 0, 'sports': 1, 'travel': 2, \
                            'computers & internet': 3, 'mundane': 4, 'society & culture': 5,
                            'food': 6, 'pets & animals': 7, 'health': 8,
                            'family & relationships': 9, 'religion': 10,
                            'education & reference': 11, 'politics & government': 12,
                            'business & finance': 13, 'science & mathematics': 14, 'others': 15}}

    labels_dict['idx2l'] = {v:k for k,v in labels_dict['l2idx'].items()}
    print("Labels mapping: ", labels_dict['l2idx'])
    df_train.loc[:, "label"] = df_train["label"].apply(lambda x: labels_dict['l2idx'][x])
    print("Unique labels: ", df_train['label'].unique())
    save_as_pickle(os.path.join(data_folder, "labels.pkl"), labels_dict)

    print("Train test split...")
    df_train, df_test = train_test_split(df_train, test_size=0.03, random_state=0, stratify=df_train[['label']])
    df_test, df_val = train_test_split(df_test, test_size=0.5, random_state=0, stratify=df_test[['label']])
    print("Train data: %d" % len(df_train))
    print("Test data: %d" % len(df_test))
    print("Val data: %d" % len(df_val))
    
    print("Train encodings...")
    train_text = df_train['text'].values.tolist()
    train_labels = df_train['label'].values.tolist()
    train_encodings = tokenizer(train_text, padding=pad, truncation=True, max_length=500)
    del train_text

    print("Test encodings...")
    test_text = df_test['text'].values.tolist()
    test_labels = df_test['label'].values.tolist()
    test_encodings = tokenizer(test_text, padding=pad, truncation=True, max_length=500)
    del test_text

    print("Validation encodings...")
    val_text = df_val['text'].values.tolist()
    val_labels = df_val['label'].values.tolist()
    val_encodings = tokenizer(val_text, padding=pad, truncation=True, max_length=500)
    del val_text
    
    return {
        'train_encodings': train_encodings,
        'train_labels': train_labels,
        'test_encodings': test_encodings,
        'test_labels': test_labels,
        'val_encodings': val_encodings,
        'val_labels': val_labels
    }

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self,encodings,labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self,idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

def get_tokenizer_model(model_selection='distilbert', num_classes=16,
                        load_model='output/mpnet_topicclassifier/best_model/model'):
    if model_selection.lower() == 'distilbert':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_classes)
    elif model_selection.lower() == 'convbert':
        tokenizer = ConvBertTokenizer.from_pretrained('YituTech/conv-bert-base')
        if len(load_model) > 2:
            model = ConvBertForSequenceClassification.from_pretrained(load_model, num_labels=num_classes)
        else:
            model = ConvBertForSequenceClassification.from_pretrained('YituTech/conv-bert-base', num_labels=num_classes)
    elif model_selection.lower() == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=num_classes)
    elif model_selection.lower() == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
    elif model_selection.lower() == 'dehatebert':
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/dehatebert-mono-english")
        model = AutoModelForSequenceClassification.from_pretrained("Hate-speech-CNERG/dehatebert-mono-english")
    elif model_selection.lower() == 'hatebert':
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT")
        model = AutoModelForSequenceClassification.from_pretrained("GroNLP/hateBERT")
    elif model_selection.lower() == 'mpnet':
        from transformers import MPNetTokenizer, MPNetForSequenceClassification
        tokenizer = MPNetTokenizer.from_pretrained('microsoft/mpnet-base')
        if len(load_model) > 2:
            model = MPNetForSequenceClassification.from_pretrained(load_model, num_labels=num_classes)
        else:
            model = MPNetForSequenceClassification.from_pretrained('microsoft/mpnet-base', num_labels=num_classes)
    else:
        raise ValueError(f'{model_selection} not recognized')

    return model, tokenizer

if __name__ == '__main__':

    parser = ArgumentParser()
    # General Parameters
    parser.add_argument('--job_name', type=str, default='topicclassifier', help='Name of the job')
    parser.add_argument('--job_id', type=str, default='1', help='ID of the job')
    parser.add_argument('--model_selection', type=str, default='gpt2', help='ID of the job')
    parser.add_argument('--load_model', type=str, default='./input/mpnet_topicclassifier/best_model/model', help='Path to trained data')
    parser.add_argument('--data_path_train', type=str, default="./data/topic_classification/processed/", help='Path to train data')
    parser.add_argument('--data_path_val', type=str, default='data/clean_val.csv', help='Path to validation data')
    parser.add_argument('--data_path_test', type=str, default='data/clean_test.csv', help='Path to test data')
    parser.add_argument('--data_sample_size', type=int, default=None, help='Sample size for data')
    parser.add_argument('--pad', type=str2bool, default=True, help='Padding in tokenizer')
    parser.add_argument("--max_examples_per_class", type=int, default=330000, help="Max number of training + test data points per class")

    # Training Parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epoch', type=int, default=5, help='Number of epoch')
    parser.add_argument('--es_patience', type=int, default=2, help='Patience for early stopping')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation')
    parser.add_argument('--use_cuda', type=str2bool, default=True, help='Use GPU for training, if available')
    
    args = parser.parse_args()

    if args.model_selection.lower() == 'gpt2':
        args.batch_size = 1
        args.pad = False

    experiment_output_folder = f'output/{args.job_id}_{args.job_name}'
    print(f'Creating output folder: {experiment_output_folder}')
    if not os.path.exists(experiment_output_folder):
        os.makedirs(experiment_output_folder)

    argparse_dict = vars(args)
    args_save_path = os.path.join(experiment_output_folder, 'argparse_parameters.json')
    print(f'Saving argument parameters: {args_save_path}')
    with open(args_save_path, 'w') as j_file:
        json.dump(argparse_dict, j_file)

    print(f'Getting tokenizer and model: {args.model_selection}')
    model, tokenizer = get_tokenizer_model(model_selection=args.model_selection, num_classes=16,\
                                        load_model=args.load_model)
    data_ = load_data(args.data_path_train, args=args, pad=args.pad)
    print('Processing Train Dataset')
    train_dataset = ClassificationDataset(data_['train_encodings'], data_['train_labels'])
    print('Processing Val Dataset')
    val_dataset = ClassificationDataset(data_['val_encodings'], data_['val_labels'])
    print('Processing Test Dataset')
    test_dataset = ClassificationDataset(data_['test_encodings'], data_['test_labels'])
    del data_

    if args.model_selection.lower() == 'convbert':
        print("FREEZING MOST HIDDEN LAYERS FOR CONVBERT...")
        unfrozen_layers = ["classifier", "pooler", "convbert.encoder.layer.11",\
                            "convbert.encoder.layer.10",
                            ]
    elif args.model_selection.lower() in ['hatebert', 'bert']:
        print("FREEZING MOST HIDDEN LAYERS FOR BERT...")
        unfrozen_layers = ["classifier", "pooler", "bert.encoder.layer.11",\
                            "bert.encoder.layer.10",
                            ]
    elif args.model_selection.lower() == 'dehatebert':
        print("FREEZING MOST HIDDEN LAYERS FOR DEHATEBERT...")
        unfrozen_layers = ["classifier", "pooler", "bert.encoder.layer.11",\
                            "bert.encoder.layer.10",
                            ]
    elif args.model_selection.lower() == 'mpnet':
        print("FREEZING MOST HIDDEN LAYERS FOR DEHATEBERT...")
        unfrozen_layers = ["classifier", "pooler", "mpnet.encoder.layer.11",\
                            "mpnet.encoder.layer.10", "mpnet.encoder.relative_attention_bias"
                            ]

    for name, param in model.named_parameters():
        if not any([layer in name for layer in unfrozen_layers]):
            print("[FROZE]: %s" % name)
            param.requires_grad = False
        else:
            print("[FREE]: %s" % name)
            param.requires_grad = True

    print('Init Trainer')
    print("USE CUDA: ", str(args.use_cuda))
    training_args = TrainingArguments(
        output_dir=experiment_output_folder,          # output directory
        evaluation_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=3,
        num_train_epochs=args.num_epoch,              # total number of training epochs
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.batch_size,   # batch size for evaluation
        warmup_steps=12,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        no_cuda=not args.use_cuda,
        load_best_model_at_end=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_strategy="epoch",
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,            # evaluation dataset
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.es_patience)] if val_dataset is not None else None
        )

    print('Training in progress...')
    try:
        trainer.train(resume_from_checkpoint=True)
    except Exception as e:
        print(e)
        trainer.train(resume_from_checkpoint=False)

    print('Evaluate on test set')
    print(trainer.evaluate(test_dataset))

    print('Saving best model...')
    best_model_path = os.path.join(experiment_output_folder, 'best_model')
    model.cpu()
    model.save_pretrained(os.path.join(best_model_path,'model'))
    tokenizer.save_pretrained(os.path.join(best_model_path,'tokenizer'))
