import os
import wandb
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from transformers import AdamW, get_cosine_schedule_with_warmup

from dataset_class import TextClassificationDataset
from utils.pad_collate import PadCollate
from train_operation import TrainOperation
from utils.focal_loss import FocalLoss

import random
import torch.backends.cudnn as cudnn

# seed 고정
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', default = 5e-05, type=float) 
    parser.add_argument('-bc', '--base_model_ckpt', type=str) #
    parser.add_argument('-tr', '--train_set', type=str) #
    parser.add_argument('-vl', '--valid_set', type=str) # 
    parser.add_argument('-fl', '--use_focal_loss', default = False, type=bool)
    parser.add_argument('-w_loss', '--use_loss_weight', default = False)
    parser.add_argument('-t_bs', '--train_batch_size', default=100, type=int)
    parser.add_argument('-v_bs', '--valid_batch_size', default=32, type=int)
    parser.add_argument('-drop_p', '--dropout_percent', default = 0.2)
    parser.add_argument('-sp', '--save_path', type=str) #
    parser.add_argument('-e', '--epochs', default = 10, type=int)
    parser.add_argument('-wr', '--warm_up_rate', default = 0.2, type=float)
    parser.add_argument('-de', '--device', default='cuda', type=str)
    parser.add_argument('-p_name', "--wandb_project_name", default="Project", type=str)

    parameters = parser.parse_args()

    return parameters

def get_loss_func(args, train_set):
    if args.use_focal_loss:
        loss = FocalLoss(alpha=0.25) ## loss weight 적용하지 않음
    else:
        if args.use_loss_weight:
            loss_weight = torch.tensor(calc_label_weight(train_set))
            loss_weight = loss_weight.to(args.device).float()
            loss = nn.CrossEntropyLoss(weight=loss_weight)
        else:
            loss = nn.CrossEntropyLoss()

    return loss

def calc_label_weight(train_set):
    labels = [int(l) for l in train_set.loc[:, 'label']]

    label_unique, label_cnts = np.unique(labels, return_counts=True)

    label_weight = [label_cnts[i] / max(label_cnts) for i in range(len(label_cnts))]

    return label_weight

def calc_warmup_steps(args, num_batchs):
    num_training_steps = args.epochs * num_batchs
    num_warmup_steps = int(args.warm_up_rate * num_training_steps)

    return num_warmup_steps


def wandb_init(args):
    if args.use_focal_loss:
        loss_type = "focal_loss"
    else:
        loss_type = "cross_entropy"

    wandb.init(project = args.wandb_project_name,
        config={
            'epochs' : args.epochs,
            'batch_size' : args.train_batch_size,
            'learning_rate' : args.learning_rate,
            'drop_p' : args.dropout_percent,
            'use_loss_weight' : args.use_loss_weight,
            'loss_type' : loss_type
        }
    )

def make_save_dir(parameters):
    dir_cnt = len(list(os.listdir(parameters.save_path))) + 1
    current_save_dir = os.path.join(parameters.save_path, 'hate_speech_' + str(dir_cnt))
    
    os.makedirs(current_save_dir)

    return current_save_dir
    

def main(parameters) -> None:
    train_set = pd.read_csv(parameters.train_set, sep = '\t')
    valid_set = pd.read_csv(parameters.valid_set, sep = '\t')   
    train_set.index = range(len(train_set)) # index 설정
    valid_set.index = range(len(valid_set)) # index 설정
    
    train_torch_dataset = TextClassificationDataset(train_set, parameters.device)
    valid_torch_dataset = TextClassificationDataset(valid_set, parameters.device)
    ppd = PadCollate(device = parameters.device, pad_id = 1)

    train_dataloader = DataLoader(train_torch_dataset, collate_fn=ppd.pad_collate, batch_size=parameters.train_batch_size, shuffle = True)
    valid_dataloader = DataLoader(valid_torch_dataset, collate_fn=ppd.pad_collate, batch_size=parameters.valid_batch_size, shuffle = False)

    loss_function = get_loss_func(parameters, train_set)
    train_operation = TrainOperation(parameters.base_model_ckpt, parameters.device, parameters, loss_function)

    optimizer = AdamW(params = train_operation.model.parameters(), lr = float(parameters.learning_rate))

    num_batchs = len(train_dataloader) * parameters.epochs
    num_warmup_steps = calc_warmup_steps(parameters, num_batchs)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer = optimizer, num_warmup_steps = num_warmup_steps, num_training_steps = num_batchs)

    wandb_init(parameters)
    wandb.watch(train_operation.model)

    ### 현재 save dir 생성
    current_save_dir = make_save_dir(parameters)


    # train step & valid & save
    for epoch in range(parameters.epochs):
        ## train operation
        train_operation.model.train()

        train_steps, train_examples = 0, 0
        tr_loss, tr_corrects, tr_f1_score = 0, 0, 0

        for _, batch in tqdm(enumerate(train_dataloader, 0)):
            temp_corrects, temp_loss, temp_f1_score, softmax_output, is_correct_data = train_operation(batch)

            train_steps += 1
            train_examples += batch['data_label'].size(0)

            tr_loss += temp_loss
            tr_corrects += temp_corrects
            tr_f1_score += temp_f1_score

            ## update 
            optimizer.zero_grad()
            temp_loss.backward()
            optimizer.step()
            lr_scheduler.step()

            step_loss_mean = tr_loss / train_steps
            step_correct_mean = tr_corrects / train_examples
            step_f1_score_mean = tr_f1_score / train_examples

            wandb.log({'train loss' : step_loss_mean, 'train corrects' : step_correct_mean, 'train f1_score' : step_f1_score_mean})
        torch.cuda.empty_cache()

        ## save checkpoint
        torch.save({
            'epoch' : epoch,
            'model_state_dict' : train_operation.model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict()
        }, os.path.join(current_save_dir, f'text_classifier_{epoch}.pt'))

        ## valid saved model
        valid_steps, valid_examples = 0, 0
        vl_loss, vl_corrects, vl_f1_score = 0, 0, 0

        train_operation.model.eval()
        for _, eval_batch in enumerate(valid_dataloader, 0):
            with torch.no_grad():
                temp_corrects, temp_loss, temp_f1_score, softmax_output, is_correct_data = train_operation(eval_batch)

                valid_steps += 1
                valid_examples += eval_batch['data_label'].size(0)

                vl_loss += temp_loss
                vl_corrects += temp_corrects
                vl_f1_score += temp_f1_score

        step_loss_mean = vl_loss / valid_steps
        step_correct_mean = vl_corrects / valid_examples
        step_f1_score_mean = vl_f1_score / valid_examples

        print(f'-------- {epoch} valid result --------')
        print(f'valid step_loss_mean : {step_loss_mean}')
        print(f'valid step_correct_mean : {step_correct_mean}')
        print(f'valid step_f1_score_mean : {step_f1_score_mean}\n')

        wandb.log({'valid loss' : step_loss_mean, 'valid corrects' : step_correct_mean, 'valid f1_score' : step_f1_score_mean})
        torch.cuda.empty_cache()

    print('Finetuning operation is Done.')


if __name__ == '__main__':
    parameters = get_params()
    main(parameters=parameters)