# Text_Classification_Trainer_Base
## Intro
Boilerplate code for Text Classification task. (using Huggingface)

## Train
### how to train?
```BASH
python code/huggingface_model/train.py \
    --learning_rate=5e-05 \
    --base_model_ckpt="bert-base-uncased"
    --train_set=TRAIN_DATASET_PATH \
    --valid_set=VALID_DATASET_PATH \
    --device=DEVICE \
    --train_batch_size=64 \
    --valid_batch_size=32 \
    --epochs=5 \
    --save_path=SAVE_PATH \
    --wandb_project_name=WANDB_PROJECT_NAME
```
### parameters
| parameter | type | description | default |
| ---------- | ---------- | ---------- | --------- |
| base_model_ckpt | str | base model checkpoint's local path or huggingface hub name | - |
| learning_rate | float | decise learning rate for train | 5e-05 |
| train_set | str | train dataset path | - |
| valid_set | str | valid dataset path | - |
| use_focal_loss | bool | decising use Focal Loss | False |
| use_loss_weight | bool | decising apply loss_weight | False |
| train_batch_size | int | batch size for model train | 100 |
| valid_batch_size | int | batch size for model valid | 32 |
| dropout_percent | float | percent value using in dropout | 0.2 |
| save_path | str | base path using for save train model checkpoint | - |
| epochs | int | epochs using in training | 10 |
| warmup_rate | float | percent for learning-rate warmup | 0.2 |
| device | str | device use in training | "cuda" |
| wandb_project_name | str | this project's name for wandb initializing | "Project" |

## Valid
### how to valid?
```BASH
python code/huggingface_model/valid.py \
    --base_model_ckpt="bert-base-uncased" \
    --selected_finetune_checkpoint=SELECTED_FINETUNE_CHECKPOINT \
    --valid_set=VALID_DATASET_PATH \
    --device=DEVICE \
    --valid_batch_size=32 \
```
### parameters
| parameter | type | description | default |
| ---------- | ---------- | ---------- | --------- |
| base_model_ckpt | str | base model checkpoint's local path or huggingface hub name | - |
| selected_finetune_checkpoint | str | model checkpoint using for valid | - |
| valid_set | str | valid dataset path | - |
| use_loss_weight | bool | decising apply loss_weight | False |
| valid_batch_size | int | batch size for model valid | 32 |
| dropout_percent | float | percent value using in dropout | 0.2 |
| device | str | device use in training | "cuda" |

</br>

## Contact
* jminju254@gmail.com
## Reference
* [Huggingface](https://huggingface.co/docs/transformers/index)
