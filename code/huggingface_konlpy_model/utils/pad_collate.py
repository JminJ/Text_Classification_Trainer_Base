import torch
from torch.nn.utils.rnn import pad_sequence

class PadCollate():
    def __init__(self, device, pad_id=1): # pad_id가 해당 tokenizer에서 1로 세팅되어 있음.
        self.pad_id = pad_id
        self.device = device

    def pad_collate(self, batch):
        input_ids, data_labels, attention_mask = [], [], []
        for idx, obj in enumerate(batch):
            input_ids.append(obj['input_ids'])
            data_labels.append(obj['data_label'])
            attention_mask.append(obj['attention_mask'])

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=self.pad_id)
        data_labels = torch.LongTensor(data_labels) 
        data_labels = data_labels.to(self.device)

        new_batch = {
            'input_ids' : input_ids.contiguous(),
            'attention_mask' : attention_mask.contiguous(),
            'data_label' : data_labels.contiguous()
        }

        return new_batch