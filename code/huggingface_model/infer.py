import os
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from huggingface_konlpy import get_tokenizer

import numpy as np
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

class InferTextClassification(nn.Module):
    def __init__(self, cls_checkpoint, device = 'cpu', base_ckpt=None):
        super(InferTextClassification, self).__init__()

        self.base_ckpt = base_ckpt
        self.device = device

        self.tokenizer = get_tokenizer(os.path.join(base_ckpt, 'vocab.txt'), 'mecab', False)
        self.tokenizer.pad_token = "[PAD]"
        self.tokenizer.cls_token = "[CLS]"
        self.tokenizer.unk_token = "[UNK]"
        self.tokenizer.sep_token = "[SEP]"
        self.tokenizer.mask_token = "[MASK]"
        self.tokenizer.add_special_tokens({'additional_special_tokens':["[BOS]", "[EOS]", "[NXT]", "[CTX]", "[SEPT]","[USR]","[SYS]","[ATCL]","[CMT]","[REP]", "[SRC]", "[TGT]"]})

        self.text_classificaiton_model = AutoModelForSequenceClassification.from_pretrained(base_ckpt)
        checkpoint = torch.load(cls_checkpoint, map_location=torch.device(self.device))
        self.text_classificaiton_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.text_classificaiton_model.eval()
        self.text_classificaiton_model.to(device)

        self.softmax = nn.Softmax(dim = 1)

    def forward(self, input_text):
        toked_result = self.tokenizer(input_text, return_tensors = 'pt', padding=True).to(self.device)
        input_ids = toked_result['input_ids']
        attention_mask = toked_result['attention_mask']

        print(toked_result)
        model_cls_output = self.text_classificaiton_model(input_ids = input_ids, attention_mask = attention_mask)
        model_cls_logits = model_cls_output.logits
        print(model_cls_logits)

        softmax_output = self.softmax(model_cls_logits)

        return softmax_output

if __name__ == '__main__':
    cls_ckpt = "" # test에 사용 될 모델 체크포인트
    cls_model = InferTextClassification(cls_checkpoint=cls_ckpt)

    input_text = "안녕하세요 JminJ입니다."

    cls_softmax = cls_model(input_text)

    print(f'\ninput_text : {input_text}')
    print(cls_softmax)