import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TextClassificationDataset(Dataset):
    def __init__(self, pd_dataset, device, ckpt):
        super(TextClassificationDataset, self).__init__()
        self.pd_dataset = pd_dataset
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self.max_length = 512

    def __len__(self):
        return len(self.pd_dataset)

    def __getitem__(self, index):
        data_index = int(self.pd_dataset.loc[index, 'index'])
        data_text = str(self.pd_dataset.loc[index, 'sentence'])
        data_label = torch.tensor(self.pd_dataset.loc[index, 'label']).int()
        data_label = data_label.to(self.device)

        ## padding 안해주는 이유는 collate_fn으로 처리할 것이기 때문.
        toked_input = self.tokenizer.encode_plus(data_text, return_tensors = 'pt', max_length = self.max_length, truncation = True)
        input_ids = torch.squeeze(toked_input['input_ids']).to(self.device)
        attention_mask = torch.squeeze(toked_input['attention_mask']).to(self.device)

        return {
            'index' : data_index,
            'sentence' : data_text,
            'input_ids' : input_ids,
            'attention_mask' : attention_mask,
            'data_label' : data_label
        }

