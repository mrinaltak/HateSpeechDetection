import sys
import os
import random
import shutil
import copy
import inspect
import torch
import datasets
import transformers
import sklearn.metrics
import random
import argparse
import time
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from utils import *
from soft_embeddings import SoftEmbedding
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import ElectraTokenizer, ElectraForSequenceClassification

def arg_parse():
    parser = argparse.ArgumentParser('arguments for training')   

    # load pretrained model
    parser.add_argument('--model', type=str, default='T5', choices=['T5','BERT','ELECTRA'])
    parser.add_argument('--model_path', type=str, default=None, help='absolute path to model checkpoint')
    parser.add_argument('--prefix', type=str, default='Continuous', help='prefix for saving stuff')

    # meta setting
    parser.add_argument('--data_path', type=str, default='../hateval_dataset', help='Root dataset')
    parser.add_argument('--seed', type=int, default=0, help='fix random seed')
    parser.add_argument('--batch_size', type=int, default=8, help='Size of training batch')
    parser.add_argument('--epochs', type=int, default=5, help='Number of Epochs to Run')
    parser.add_argument('--lr', type=float, default=3e-3, help='Learning Rate')
    parser.add_argument('--lr_decay', type=float, default=1, help='Learning Rate decay rate')
    parser.add_argument('--n_tokens', type=int, default=20, help='Number of Tokens in Continuous prompt')
    
    # other defaults
    parser.add_argument('--max_target_length', type=int, default=10, help='Number of tokens in generator output')
    parser.add_argument('--max_input_length', type=int, default=512, help='Number of tokens in input')
    parser.add_argument('--initialize_from_vocab', type=int, default=1, help='should the continuous prompts be initialized from vocab or not')
    parser.add_argument('--n_tasks', type=int, default=3, help='number of different tasks')

    opt = parser.parse_args()
    opt.cuda = torch.cuda.is_available()
    opt.device = 'cuda' if opt.cuda else 'cpu'
    opt.model_folder = "_".join(["HatEval", opt.prefix, opt.model, str(opt.n_tokens), str(opt.lr), str(opt.batch_size), str(opt.lr_decay)])
    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)
    opt.start_epoch = 0

    return opt

def create_model(args):
    model_name = args.model
    if model_name == 'T5':
        tokenizer = T5Tokenizer.from_pretrained('t5-small', cache_dir='../t5_cache')
        model = T5ForConditionalGeneration.from_pretrained('t5-small', cache_dir='../t5_cache')
    elif model_name == "BERT":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='../bert_cache')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', cache_dir='../bert_cache')
    elif model_name == "ELECTRA":
        tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator', cache_dir='../electra_cache')
        model = ElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator', cache_dir='../electra_cache')
    else:
        raise NotImplementedError('Model not supported: {}'.format(model_name))
    
    print("Freezing model params")
    if model_name in ['BERT', 'ELECTRA']:
        print("unfreeze classification head")
        for name, param in model.named_parameters():
            if 'classifier' not in name: # classifier layer
                param.requires_grad = False
    else:
        for name, param in model.named_parameters():
            param.requires_grad = False

    #attach soft embeddings layer
    s_wte = SoftEmbedding(model.get_input_embeddings(), 
                      n_tokens=args.n_tokens, 
                      initialize_from_vocab=args.initialize_from_vocab,
                      n_tasks = args.n_tasks,
                      device = args.device)
    # model.set_input_embeddings(s_wte)
    optimizer = AdamW([s_wte.learned_embedding], lr=args.lr, eps=1e-8)

    if args.cuda:
        model = model.cuda()

    if args.model_path != None:
        if os.path.isfile(args.model_path):
            map_location = 'cuda' if args.cuda else 'cpu'
            print("=> loading checkpoint '{}'".format(args.model_path))
            checkpoint = torch.load(args.model_path, map_location=map_location)
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            s_wte.load_state_dict(checkpoint['s_wte'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded successfully '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.model_path))

    return model, tokenizer, optimizer, s_wte, args

def main():
    args = arg_parse()

    set_seed(args.seed)
    # eval_batch_size = 1 if args.model in ['T5'] else args.batch_size
    eval_batch_size = args.batch_size

    train_dataset, val_dataset, test_dataset = load_hateval_data(args.data_path)
    train_dataloader = torch.utils.data.DataLoader(train_dataset['train'], shuffle=True, batch_size=eval_batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset['val'], shuffle=True, batch_size=eval_batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset['test'], shuffle=True, batch_size=eval_batch_size)

    print("Length of Datasets - ")
    print("Train = {}\nVal = {}\nTest = {}".format(len(train_dataset['train']),len(val_dataset['val']), len(test_dataset['test'])))
    
    model, tokenizer, optimizer, s_wte, args = create_model(args)
    train_model(args, model, optimizer, tokenizer, s_wte, train_dataloader, val_dataloader)
    # validate(args, model, optimizer, tokenizer, s_wte, val_dataloader)
    tester(args, model, tokenizer, s_wte, test_dataloader)

def save_model(args, epoch, model, optimizer, s_wte):
    # save model
    print('==> Saving...')
    state = {
        'opt': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        's_wte': s_wte.state_dict(),
        'epoch': epoch,
    }
    save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
    torch.save(state, save_file)
    # help release GPU memory
    del state
    
def get_output(model_name, label):
    if model_name in ['T5']:
        return "yes" if label else "no"
    else:
        return 1 if label else 0
    return 0

def create_input(model_name, batch):
    input_sequences=[]
    task_number=[]
    output_sequences = []
    for idx, text in enumerate(batch['text']):
        input_sequences.append(text)
        task_number.append(0)
        output_sequences.append(get_output(model_name, batch['HS'][idx]))

        if batch['HS'][idx]:
            input_sequences.append(text)
            task_number.append(1)
            output_sequences.append(get_output(model_name, batch['TR'][idx]))

            if batch['TR'][idx]:
                input_sequences.append(text)
                task_number.append(2)
                output_sequences.append(get_output(model_name, batch['AG'][idx]))

    return input_sequences, task_number, output_sequences

def train_model(args, model, optimizer, tokenizer, s_wte, train_dataloader, val_dataloader):
    losses = []
    for epoch in tqdm(range(args.start_epoch,args.epochs)):
        print("Training epoch %d" % epoch)
        print()
        model.train()
        t0 = time.time()
        for batch in tqdm(train_dataloader):
            input_sequences, task_number, output_sequences = create_input(args.model, batch)
            
            encoding = tokenizer(input_sequences,
                                padding='longest',
                                max_length=args.max_input_length,
                                truncation=True,
                                return_tensors="pt")
            input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
            
            #add prompt tokens
            input_ids = torch.cat([torch.tensor(task_number).repeat(args.n_tokens,1).transpose(0,1), input_ids], 1)
            attention_mask = torch.cat([torch.full((attention_mask.shape[0],args.n_tokens), 1), attention_mask], 1)

            # encode the targets
            if args.model in ['T5']:
                target_encoding = tokenizer(output_sequences,
                                            padding='longest',
                                            max_length=args.max_target_length,
                                            truncation=True)
                labels = target_encoding.input_ids

                # replace padding token id's of the labels by -100
                labels = [
                        [(label if label != tokenizer.pad_token_id else -100) for label in labels_example] for labels_example in labels
                ]
                labels = torch.tensor(labels)
            else:
                labels = torch.tensor(output_sequences)

            # forward pass
            if args.cuda:
                input_ids, attention_mask, labels = input_ids.cuda(), attention_mask.cuda(), labels.cuda()

            loss = model(inputs_embeds=s_wte(input_ids), attention_mask=attention_mask, labels=labels).loss
            optimizer.zero_grad()
            loss.backward()
            model.zero_grad()
            optimizer.step()
            losses.append(loss.detach().cpu().item())

        t1 = time.time()
        print("Time taken ", t1-t0, "\n")
        save_model(args, epoch, model, optimizer, s_wte)
        plt.plot(losses)
        plt.savefig(os.path.join(args.model_folder, "train_loss.png"))

        validate(args, model, optimizer, tokenizer, s_wte, val_dataloader)

        lr = args.lr * (args.lr_decay ** (epoch+1))  

        ##evaluate on validation set

def validate(args, model, optimizer, tokenizer, s_wte, dataloader):

    model.eval()
    task_accs = [[],[],[]]

    if args.model in ['T5']:
        pos = tokenizer("yes").input_ids[0]
        neg = tokenizer("no").input_ids[0]
        decoder_input_ids = model._shift_right(tokenizer([''],return_tensors="pt").input_ids)
        if args.cuda:
            decoder_input_ids = decoder_input_ids.cuda()

    for batch in tqdm(dataloader):
        if args.model in ['T5']:
            input_sequences, task_number, output_sequences_ = create_input('BERT', batch)
        else:
            input_sequences, task_number, output_sequences_ = create_input(args.model, batch)
        encoding = tokenizer(input_sequences,
                            padding='longest',
                            max_length=args.max_input_length,
                            truncation=True,
                            return_tensors="pt")
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
        input_ids = torch.cat([torch.tensor([task_number]).repeat(args.n_tokens,1).transpose(0,1), input_ids], 1)
        attention_mask = torch.cat([torch.full((attention_mask.shape[0],args.n_tokens), 1), attention_mask], 1)
        
        if args.cuda:
            input_ids, attention_mask = input_ids.cuda(), attention_mask.cuda()
        with torch.no_grad():
            if args.model in ['T5']:    
                output_dist = model(inputs_embeds=s_wte(input_ids), attention_mask=attention_mask, decoder_input_ids = decoder_input_ids.repeat(input_ids.shape[0],1)).logits
                output_dist = output_dist[:,-1,[neg, pos]]
                preds = torch.argmax(output_dist, dim=-1).cpu().numpy()
            else:
                preds = model(inputs_embeds=s_wte(input_ids),attention_mask=attention_mask).logits
                preds = torch.argmax(preds,dim=1).cpu().numpy()

        for i,pred in enumerate(preds):
            task_accs[task_number[i]].append(pred==output_sequences_[i])
    print("Validation Task Accs\ntask-HS %.4f, task-TR %.4f, task-AG %.4f"%(np.mean(task_accs[0]), np.mean(task_accs[1]), np.mean(task_accs[2])))
    return task_accs

def tester(args, model, tokenizer, s_wte, test_dataloader):
    print()
    print("Testing")
    print()

    task_preds = [[],[],[]]
    task_gt = [[],[],[]]
    emr = []
    input_text = []

    model.eval()
    if args.model in ['T5']:
        pos = tokenizer("yes").input_ids[0]
        neg = tokenizer("no").input_ids[0]
        decoder_input_ids = model._shift_right(tokenizer([''],return_tensors="pt").input_ids)
        if args.cuda:
            decoder_input_ids = decoder_input_ids.cuda()

    for batch in tqdm(test_dataloader):
        sentences = batch['text'] # use different length sentences to test batching
        inputs = tokenizer(sentences, return_tensors="pt", padding=True)
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask

        bs = input_ids.shape[0]
        task_number = [0] * bs + [1] *bs + [2] * bs
        input_ids = input_ids.repeat(3,1)
        attention_mask = attention_mask.repeat(3,1)

        #add prompt tokens
        input_ids = torch.cat([torch.tensor(task_number).repeat(args.n_tokens,1).transpose(0,1), input_ids], 1)
        attention_mask = torch.cat([torch.full((attention_mask.shape[0],args.n_tokens), 1), attention_mask], 1)

        if args.cuda:
            input_ids, attention_mask = input_ids.cuda(), attention_mask.cuda()

        with torch.no_grad():
            if args.model in ['T5']:
                output_dist = model(inputs_embeds=s_wte(input_ids), attention_mask=attention_mask, decoder_input_ids = decoder_input_ids.repeat(input_ids.shape[0],1)).logits
                output_dist = output_dist[:,-1,[neg, pos]]
                preds = torch.argmax(output_dist, dim=-1).cpu().numpy()
            else:
                preds = model(inputs_embeds=s_wte(input_ids),attention_mask=attention_mask).logits
                preds = torch.argmax(preds,dim=1).cpu().numpy()

        for i in range(bs):
            task_gt[0].append(batch['HS'][i].cpu().numpy().item())
            task_gt[1].append(batch['TR'][i].cpu().numpy().item())
            task_gt[2].append(batch['AG'][i].cpu().numpy().item())
            input_text.append(sentences[i])

            
            hs_pred = preds[i]
            tr_pred = preds[bs+i]
            ag_pred = preds[2*bs+i]

            task_preds[0].append(hs_pred)
            emr.append(task_preds[0][-1]==task_gt[0][-1])
            if hs_pred == 0:
                task_preds[1].append(0)
                task_preds[2].append(0)
            else:
                task_preds[1].append(tr_pred)
                emr[-1] *= (task_preds[1][-1]==task_gt[1][-1])
                if tr_pred == 0:
                    task_preds[2].append(0)
                else:
                    task_preds[2].append(ag_pred)
                    emr[-1] *= (task_preds[2][-1]==task_gt[2][-1])


    print("HS Report-\n")
    report = classification_report(y_true=task_gt[0], y_pred=task_preds[0])
    print(report)
    print("TR Report-\n")
    gt_ = np.array(task_gt[1])[np.array(task_gt[0]).astype(bool)]
    pred_ = np.array(task_preds[1])[np.array(task_gt[0]).astype(bool)]
    report = classification_report(y_true=gt_, y_pred=pred_)
    print(report)
    print("AG Report-\n")
    gt_ = np.array(task_gt[2])[np.array(task_gt[1]).astype(bool)]
    pred_ = np.array(task_preds[2])[np.array(task_gt[1]).astype(bool)]
    report = classification_report(y_true=gt_, y_pred=pred_)
    print(report)
    print("EMR", np.mean(emr))

    metrics0 = precision_recall_fscore_support(task_gt[0], task_preds[0], average='macro')
    metrics1 = precision_recall_fscore_support(task_gt[1], task_preds[1], average='macro')
    metrics2 = precision_recall_fscore_support(task_gt[2], task_preds[2], average='macro')

    mean = (np.array(metrics0)[:-1]+np.array(metrics1)[:-1]+np.array(metrics2)[:-1])/3
    print("AVG Precision {}, Recall {}, F1 {}\n".format(mean[0],mean[1],mean[2]))

    with open("cache/HatEval_" + args.prefix + "_" + args.model + "_preds.pkl", "wb") as f:
        pickle.dump([input_text,task_gt,task_preds], f)


if __name__=='__main__':
    main()


            


