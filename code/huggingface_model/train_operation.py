from turtle import forward
from typing import List, Tuple
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score

from transformers import AutoModelForSequenceClassification

class TrainOperation(nn.Module):
    def __init__(self, base_ckpt, device, args, loss_func):
        super(TrainOperation, self).__init__()

        self.model = AutoModelForSequenceClassification.from_pretrained(base_ckpt)
        self.model.to(device)

        self.device = device
        self.args = args

        self.loss_func = loss_func
        self.softmax = nn.Softmax(dim=1)

    def forward(self, toked_input) -> Tuple[int, torch.Tensor, np.float, List, List]:
        labels = toked_input['data_label']

        input_ids = toked_input['input_ids']
        attention_mask = toked_input['attention_mask']

        model_output = self.model(input_ids = input_ids, attention_mask = attention_mask)
        model_logits = model_output['logits']

        temp_corrects, is_correct_data, output_max = self.calc_corrects(model_logits, labels)
        temp_loss = self.loss_func(model_logits, labels)

        labels_numpy = labels.detach().cpu().numpy()
        output_max_numpy = output_max.detach().cpu().numpy()

        temp_f1_score = f1_score(labels_numpy, output_max_numpy)

        softmax_output = self.softmax(model_logits) # -> tensor
        softmax_output = softmax_output.tolist()

        return temp_corrects, temp_loss, temp_f1_score, softmax_output, is_correct_data

    def calc_corrects(self, model_output, labels) -> Tuple[int, List, torch.Tensor]:
        output_max_index = torch.argmax(model_output, dim=1)

        is_correct_data = [1 if output_max_index[i].item() == labels[i].item() else 0 for i in range(len(labels))]

        corrects_num = sum(is_correct_data)

        return corrects_num, is_correct_data, output_max_index