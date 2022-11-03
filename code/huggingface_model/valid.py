import os
from typing import List,Tuple
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
    parser.add_argument('-bc', '--base_model_ckpt', type=str)
    parser.add_argument('-s_fbc', '--selected_finetune_checkpoint', type=str)
    parser.add_argument('-vl', '--valid_set', type=str)
    parser.add_argument('-w_loss', '--use_loss_weight', default = False, type=bool)
    parser.add_argument('-bs', '--valid_batch_size', default=32, type=int)
    parser.add_argument('-drop_p', '--dropout_percent', default=0.2, type=float)
    parser.add_argument('-de', '--device', default="cuda", type=str)

    parameters = parser.parse_args()

    return parameters

def make_valid_result_dir(dir_path:str):
    if not os.path.isdir(dir_path):
        temp_dir = os.getcwd()
        print(f"temp dir path : {temp_dir}")
        print(f"make valid result dir -> {dir_path}")
        os.mkdir(dir_path)
    

def get_loss_func(args, dataset):
    if args.use_loss_weight:
        loss_weight = torch.tensor(calc_label_weight(dataset)).float()
        loss = nn.CrossEntropyLoss(weight=loss_weight)
    else:
        loss = nn.CrossEntropyLoss()

    return loss

def calc_label_weight(dataset):
    labels = [int(l) for l in dataset.loc[:, 'label']]

    label_unique, label_cnts = np.unique(labels, return_counts=True)

    label_weight = [label_cnts[i] / max(label_cnts) for i in range(len(label_cnts))]

    return label_weight


def valid(args):
    result_save_path = './valid_result'
    make_valid_result_dir(result_save_path)

    valid_set = pd.read_csv(args.valid_set, sep = '\t')
    ppd = PadCollate(device = args.device, pad_id = 1)

    tokenizer_ckpt = os.path.join(args.base_model_ckpt, 'vocab.txt')
    valid_torch_dataset = TextClassificationDataset(valid_set, args.device, ckpt=tokenizer_ckpt)
    valid_dataloader = DataLoader(valid_torch_dataset, collate_fn=ppd.pad_collate, batch_size=args.valid_batch_size, shuffle = False, num_workers=0)

    loss_function = get_loss_func(args, valid_set)
    
    valid_operation = TrainOperation(args.base_model_ckpt, args.device, args, loss_function)
    checkpoint = torch.load(args.selected_finetune_checkpoint)
    valid_operation.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    valid_operation.model.to(args.device)
    valid_operation.model.eval()

    valid_steps, valid_examples = 0, 0
    vl_loss, vl_corrects, vl_f1_score = 0, 0, 0
    valid_is_correct_datas = [] ## 정답 데이터, 오답 데이터 저장
    valid_softmax_datas = [] ## softmax 데이터 받아옴

    for _, eval_batch in enumerate(valid_dataloader, 0):
        with torch.no_grad():
            temp_corrects, temp_loss, temp_f1_score, softmax_output, is_correct_data = valid_operation(eval_batch)
            valid_is_correct_datas.extend(is_correct_data) ## infer 결과 체크
            valid_softmax_datas.extend(softmax_output)

            valid_steps += 1
            valid_examples += eval_batch['data_label'].size(0)

            vl_loss += temp_loss
            vl_corrects += temp_corrects
            vl_f1_score += temp_f1_score

        step_loss_mean = vl_loss / valid_steps
        step_correct_mean = vl_corrects / valid_examples
        step_f1_score_mean = vl_f1_score / valid_steps 

    print(f'-------- valid result --------')
    print(f'valid step_loss_mean : {step_loss_mean}')
    print(f'valid step_correct_mean : {step_correct_mean}')
    print(f'valid step_f1_score_mean : {step_f1_score_mean}\n')

    valid_set.insert(len(valid_set.columns), 'softmax_result', valid_softmax_datas)
    temp_dir_in_cnt = str(len(os.listdir(result_save_path)) + 1)
    
    ## original_way
    valid_result_correct_set, valid_result_not_correct_set = separate_by_true_false(valid_is_correct_datas, valid_dataset=valid_set)
    valid_result_correct_set.to_csv(result_save_path + '/valid_for_compare_correct_result_' + temp_dir_in_cnt + '.tsv', sep = '\t', index = False)
    valid_result_not_correct_set.to_csv(result_save_path + '/valid_for_compare_not_correct_result_' + temp_dir_in_cnt + '.tsv', sep = '\t', index = False)

    print(f"valid results saved.")

    return valid_is_correct_datas

def separate_by_true_false(is_correct_list:List, valid_dataset:pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    correct_true_list = [bool(i) for i in is_correct_list]
    correct_false_list = [bool(abs(i-1)) for i in is_correct_list]

    only_true_dataset = valid_dataset[correct_true_list]
    only_false_dataset = valid_dataset[correct_false_list]

    return only_true_dataset, only_false_dataset

if __name__ == '__main__':
    parameters = get_params()
    valid_is_correct_datas = valid(parameters)