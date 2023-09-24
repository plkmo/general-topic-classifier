# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:04:38 2019

@author: WT
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from nlptoolkit.utils.misc import save_as_pickle
from nlptoolkit.classification.models.GCN.trainer import train_and_fit as GCN
from nlptoolkit.classification.models.BERT.trainer import train_and_fit as BERT
from nlptoolkit.classification.models.XLNet.trainer import train_and_fit as XLNet
from nlptoolkit.classification.models.GAT.trainer import train_and_fit as GAT
from nlptoolkit.classification.models.ALBERT.trainer import train_and_fit as ALBERT
from nlptoolkit.classification.models.XLMRoBERTa.trainer import train_and_fit as XLMRoBERTa
from nlptoolkit.classification.models.GIN.trainer import train_and_fit as GIN
from nlptoolkit.classification.models.DEBERTA.trainer import train_and_fit as DEBERTA
from nlptoolkit.classification.models.ConvBERT.trainer import train_and_fit as ConvBERT
from nlptoolkit.classification.models.ADAN.trainer import train_and_fit as ADAN
from nlptoolkit.classification.models.DistilBERT.trainer import train_and_fit as DistilBERT
from nlptoolkit.classification.models.infer import infer_from_trained
import logging
from argparse import ArgumentParser

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

'''
Classification input data files:
train.csv - columns: text, labels
infer.csv - columns: text, optional labels

Output file:
results.csv - columns: index, predicted labels
'''

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env", type=str, default="aws", help="aws or nscc")
    parser.add_argument("--train_data", type=str, default="../../data/topic_classification/processed/", \
                        help="training data csv file path")
    parser.add_argument("--infer_data", type=str, default="./data/infer.csv", help="infer data csv file path")
    parser.add_argument("--max_vocab_len", type=int, default=60000, \
                        help="GCN, GAT, GIN: Max vocab size to consider based on top frequency tokens")
    parser.add_argument("--hidden_size_1", type=int, default=330, \
                        help="Size of first GCN hidden weights")
    parser.add_argument("--hidden_size_2", type=int, default=130, \
                        help="Size of second GCN hidden weights")
    parser.add_argument('--batched', type=int, default=0,\
                        help= 'For GCN, GIN - 0: no batch training ; 1: do batch training')
    parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units for GAT')
    parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions for GAT')
    parser.add_argument("--tokens_length", type=int, default=256, help="Max tokens length for BERT")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of prediction classes (starts from integer 0)")
    parser.add_argument("--max_examples_per_class", type=int, default=200000, help="Max number of training + test data points per class")
    parser.add_argument("--train_test_split", type=int, default=1, help="0: No, 1: Yes (Only activate if infer.csv contains labelled data)")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="GCN: Ratio of test to training nodes")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--gradient_acc_steps", type=int, default=1, help="No. of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--num_epochs", type=int, default=11, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.0007, help="learning rate")
    parser.add_argument("--use_cuda", type=int, default=0, help="Use cuda for GAT (0: No , 1: Yes)")
    parser.add_argument("--use_sagemaker", type=int, default=0, help="Use Sagemaker (0: No , 1: Yes)")
    parser.add_argument("--model_no", type=int, default=8, help='''Model ID: (0: Graph Convolution Network (GCN), 
                                                                            \n1: BERT, 
                                                                            \n2: XLNet, 
                                                                            \n3: Graph Attention Network (GAT))
                                                                            \n4: ALBERT
                                                                            \n5: XLMRoBERTa
                                                                            \n6: GIN
                                                                            \n7: DeBERTa
                                                                            \n8: ConvBERT
                                                                            \n9: ADAN
                                                                            \n10: DistilBERT''')
    parser.add_argument("--model_size", type=str, default="bert-base-uncased", \
                            help="Model size")
    
    parser.add_argument("--train", type=int, default=1, help="Train model on dataset")
    parser.add_argument("--infer", type=int, default=1, help="Infer input sentence labels from trained model")
    
    args = parser.parse_args()
    save_as_pickle("args.pkl", args)
    
    if args.train:
        if args.model_no == 0:
            net = GCN(args)
        elif args.model_no == 1:
            net = BERT(args)
        elif args.model_no == 2:
            XLNet(args)
        elif args.model_no == 3:
            net = GAT(args)
        elif args.model_no == 4:
            net = ALBERT(args)
        elif args.model_no == 5:
            net = XLMRoBERTa(args)
        elif args.model_no == 6:
            net = GIN(args)
        elif args.model_no == 7:
            net = DEBERTA(args)
        elif args.model_no == 8:
            net = ConvBERT(args)
        elif args.model_no == 9:
            net = ADAN(args)
        elif args.model_no == 10:
            net = DistilBERT(args)
        else:
            print("Model selection not found.")
    
    if args.infer:
        if args.model_no in [0, 3, 6]:
            logger.info("Infer function not compatible with GCN, GAT, GIN!")
        else:
            inferer = infer_from_trained(args)
            while True:
                opt = '1'
                if opt == '0':
                    inferer.infer_from_input()
                    break
                elif opt == '1':
                    #inferer.infer_from_file(in_file="./data/input.txt", out_file="/opt/ml/model/output.csv")
                    inferer.infer_from_file(in_file=os.path.join(args.train_data, "test_infer.csv"), \
                                            out_file=os.path.join(args.train_data, "inferred_test_infer.csv"))
                    break
                else:
                    print("Invalid option, please try again.")
