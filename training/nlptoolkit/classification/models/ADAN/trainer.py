# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 16:11:07 2019

@author: WT
"""
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from .preprocessing_funcs import save_as_pickle, load_pickle
from .train_funcs import load_dataloaders, load_state, load_results, model_eval, infer
from transformers import BertForSequenceClassification
import matplotlib.pyplot as plt
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def train_and_fit(args):
    cuda = torch.cuda.is_available()
    
    train_loader, test_loader, train_len, test_len = load_dataloaders(args)
    logger.info("Training data points: %d" % train_len)
    logger.info("Test data points: %d" % test_len)

    labels_dict = load_pickle(os.path.join(args.train_data, "labels.pkl"))
    num_classes = len(labels_dict['l2idx'].keys())
    print("NUM CLASSES: %d\%s" % (num_classes, str(labels_dict['l2idx'].keys())))

    net = BertForSequenceClassification.from_pretrained(args.model_size, num_labels=args.num_classes)
    net.resize_token_embeddings(len(tokenizer))
    if cuda:
        net.cuda()
        
    logger.info("FREEZING MOST HIDDEN LAYERS...")
    unfrozen_layers = ["classifier", "bert.pooler", "bert.encoder.layer.11"] if ('base' in args.model_size) else \
                    ["classifier", "bert.pooler", "bert.encoder.layer.23"]
    for name, param in net.named_parameters():
        if not any([layer in name for layer in unfrozen_layers]):
            print("[FROZE]: %s" % name)
            param.requires_grad = False
        else:
            print("[FREE]: %s" % name)
            param.requires_grad = True
       
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([{"params":net.bert.parameters(),"lr": 0.0003},\
                             {"params":net.classifier.parameters(), "lr": args.lr}])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,8,12,15,18,20], gamma=0.8)
    
    start_epoch, best_pred = load_state(net, optimizer, scheduler, args, load_best=False)    
    losses_per_epoch, accuracy_per_epoch = load_results(args)
    
    f1_per_epoch = []
    
    logger.info("Starting training process...")
    update_size = len(train_loader)//10
    for epoch in range(start_epoch, args.num_epochs):
        net.train(); total_loss = 0.0; losses_per_batch = []
        start_time = time.time()
        for i, data in enumerate(train_loader, 0):
            inputs, token_type, mask, labels = data
            if cuda:
                inputs, token_type, mask, labels = inputs.cuda(), token_type.cuda(), mask.cuda(), labels.cuda()
            inputs = inputs.long(); labels = labels.long()
            outputs = net(inputs, token_type_ids=token_type, attention_mask=mask)
            loss = criterion(outputs.logits, labels)
            loss = loss/args.gradient_acc_steps
            loss.backward()
            clip_grad_norm_(net.parameters(), args.max_norm)
            if (i % args.gradient_acc_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            total_loss += loss.item()
            if (i % update_size) == (update_size - 1):    # print every 100 mini-batches of size = batch_size
                losses_per_batch.append(args.gradient_acc_steps*total_loss/update_size)
                print('[Epoch: %d, %5d/ %d points] total loss per batch: %.3f' %
                      (epoch + 1, (i + 1)*args.batch_size, train_len, losses_per_batch[-1]))
                total_loss = 0.0
        
        scheduler.step()
        losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
        if args.train_test_split == 1:
            eval_metrics = model_eval(net, test_loader, cuda=cuda)
            accuracy_per_epoch.append(eval_metrics['acc'])
            f1_per_epoch.append(eval_metrics['f1'])
        else:
            eval_metrics = model_eval(net, train_loader, cuda=cuda)
            accuracy_per_epoch.append(eval_metrics['acc'])
            f1_per_epoch.append(eval_metrics['f1'])
        
        print("Epoch completed, took %.2f seconds." % (time.time() - start_time))
        print("Losses at Epoch %d: %.7f" % (epoch + 1, losses_per_epoch[-1]))
        print("Accuracy at Epoch %d: %.7f" % (epoch + 1, accuracy_per_epoch[-1]))
        print("F1 at Epoch %d: %.7f" % (epoch + 1, f1_per_epoch[-1]))
        if f1_per_epoch[-1] > best_pred:
            best_pred = f1_per_epoch[-1]
            if args.use_sagemaker != 1:
                torch.save({
                        'epoch': epoch + 1,\
                        'state_dict': net.state_dict(),\
                        'best_acc': f1_per_epoch[-1],\
                        'optimizer' : optimizer.state_dict(),\
                        'scheduler' : scheduler.state_dict(),\
                    }, os.path.join("./data/" , "test_model_best_%d.pth.tar" % args.model_no))
            else:
                torch.save({
                        'epoch': epoch + 1,\
                        'state_dict': net.state_dict(),\
                        'best_acc': f1_per_epoch[-1],\
                        'optimizer' : optimizer.state_dict(),\
                        'scheduler' : scheduler.state_dict(),\
                    }, os.path.join("/opt/ml/model" , "test_model_best_%d.pth.tar" % args.model_no))
        if (epoch % 2) == 0:
            save_as_pickle("test_losses_per_epoch_%d.pkl" % args.model_no, losses_per_epoch)
            save_as_pickle("test_accuracy_per_epoch_%d.pkl" % args.model_no, f1_per_epoch)
            if args.use_sagemaker != 1:
                torch.save({
                        'epoch': epoch + 1,\
                        'state_dict': net.state_dict(),\
                        'best_acc': f1_per_epoch[-1],\
                        'optimizer' : optimizer.state_dict(),\
                        'scheduler' : scheduler.state_dict(),\
                    }, os.path.join("./data/" , "test_checkpoint_%d.pth.tar" % args.model_no))
            else:
                torch.save({
                        'epoch': epoch + 1,\
                        'state_dict': net.state_dict(),\
                        'best_acc': f1_per_epoch[-1],\
                        'optimizer' : optimizer.state_dict(),\
                        'scheduler' : scheduler.state_dict(),\
                    }, os.path.join("/opt/ml/model" , "test_checkpoint_%d.pth.tar" % args.model_no))
    
    logger.info("Finished Training!")
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    ax.scatter([e for e in range(len(losses_per_epoch))], losses_per_epoch)
    ax.tick_params(axis="both", length=2, width=1, labelsize=14)
    ax.set_xlabel("Epoch", fontsize=22)
    ax.set_ylabel("Training Loss per batch", fontsize=22)
    ax.set_title("Training Loss vs Epoch", fontsize=32)
    plt.savefig(os.path.join("./data/" if (args.use_sagemaker != 1) else "/opt/ml/model" ,"loss_vs_epoch_%d.png" % args.model_no))
    
    fig2 = plt.figure(figsize=(20,20))
    ax2 = fig2.add_subplot(111)
    ax2.scatter([e for e in range(len(accuracy_per_epoch))], accuracy_per_epoch)
    ax2.tick_params(axis="both", length=2, width=1, labelsize=14)
    ax2.set_xlabel("Epoch", fontsize=22)
    ax2.set_ylabel("Test Accuracy", fontsize=22)
    ax2.set_title("Test Accuracy vs Epoch", fontsize=32)
    plt.savefig(os.path.join("./data/" if (args.use_sagemaker != 1) else "/opt/ml/model" ,"accuracy_vs_epoch_%d.png" % args.model_no))

    fig3 = plt.figure(figsize=(20,20))
    ax3 = fig3.add_subplot(111)
    ax3.scatter([e for e in range(len(f1_per_epoch))], f1_per_epoch)
    ax3.tick_params(axis="both", length=2, width=1, labelsize=14)
    ax3.set_xlabel("Epoch", fontsize=22)
    ax3.set_ylabel("Test F1", fontsize=22)
    ax3.set_title("Test F1 vs Epoch", fontsize=32)
    plt.savefig(os.path.join("./data/" if (args.use_sagemaker != 1) else "/opt/ml/model" ,"F1_vs_epoch_%d.png" % args.model_no))
    
    infer(test_loader, net)