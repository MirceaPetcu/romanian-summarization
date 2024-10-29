import bitsandbytes as bnb
import os
import torch
from tqdm import tqdm
import torch.nn as nn
from transformers import BitsAndBytesConfig,Seq2SeqTrainingArguments,Seq2SeqTrainer
import torch
from peft import PeftModel
import bitsandbytes as bnb
# import evaluate
import numpy as np
from datasets import load_dataset,DatasetDict, concatenate_datasets
from transformers import (
    T5ForConditionalGeneration,
    MT5ForConditionalGeneration,
    MBartTokenizer,
    MBartForConditionalGeneration,
    T5Tokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

import logging

logging.basicConfig(filename='dpo_logs_mbart_128_128.log',   
                    filemode='w',               
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')




hyperparameters = {
                'lr':5e-6,
                'batch_size_train': 3,
                'batch_size_val': 2,
                'model_type': 'lora',
                'gradient_accumulation_steps': 8,
                'epochs':5,
                'warmup_steps':None,
                'warmup_ratio':0.1,
                'wd': None,
                'log_steps': 5,
                'eval_steps': 10,
                'save_steps': 10,
                'rank' : 128,
                'alpha': 128,
                'beta': 0.7,
                'lr_scheduler': 'cosine',
                'run_name': 'run10',
                'output_dir': 'results_dpo10'

                }



def find_all_linear_names(model):
    """
    Find all fully connected layers and add low-rank adapters to each one.
    """
    cls = bnb.nn.Linear4bit

    lora_module_names = set()

    for name, module in model.named_modules():
        if isinstance(module, cls):
            # Extract module names
            names = name.split('.')
            # Add the module's top-level name to the set
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')

    return list(lora_module_names)


def get_model_lora():
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32,
    )

    checkpoint = "facebook/mbart-large-cc25"
    tokenizer = MBartTokenizer.from_pretrained(checkpoint, 
                                            src_lang="ro_RO", tgt_lang="ro_RO",
                                            )
    model = MBartForConditionalGeneration.from_pretrained(checkpoint,
                                                        quantization_config=bnb_config,
                                                        low_cpu_mem_usage = True
                                                        )
    model.config.use_cache = False
    model.supports_gradient_checkpointing = True  
    model.enable_input_require_grads()
    prepare_model_for_kbit_training(model,True,{'use_reentrant':False})
    


    lora_modules = find_all_linear_names(model)

    config = LoraConfig(
        r=hyperparameters['rank'], 
        lora_alpha=hyperparameters['alpha'], 
        target_modules=lora_modules+["lm_head"],  
        lora_dropout=0.01, 
        bias="none", 
        task_type="SEQ_2_SEQ_LM",
        init_lora_weights = True,
        use_rslora=True,
    )
    

    for name, param in model.named_parameters():
        param.requires_grad=False
    #     print(name,param.requires_grad,param.dtype,sep=' ',end='\n')
    print(model)
    # model = get_peft_model(model, config)
    model = PeftModel.from_pretrained(model, 'mbart-128-128',is_trainable=True,adapter_name='train2')
    model.load_adapter('mbart-128-128', is_trainable=False, adapter_name="reference")

    model.lm_head.weight.requires_grad=False
    print(model.lm_head.weight.requires_grad)
    print(model.lm_head.weight.dtype)
    
    print(model)
    model.train()
    return model,tokenizer,config

    


def main(policy,peft_config,tokenizer):

    
    def get_data_from_hub():
        dataset = load_dataset('mirceaPetcu/dpo-dataset',token='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')
        def change_ceddila_to_comma(batch):
            batch['prompt'] = batch['prompt'].replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")
            batch['rejected'] = batch['rejected'].replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")
            batch['chosen'] = batch['chosen'].replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")
            batch['chosen'] = batch['chosen'].replace('..','. ')
            batch['prompt'] = batch['prompt'].strip()
            batch['rejected'] = batch['rejected'].strip()
            batch['chosen'] = batch['chosen'].strip()

            return batch
        
        dataset = dataset.map(change_ceddila_to_comma)
        dataset = dataset['train']
        return dataset
    

    def return_prompt_and_responses(samples):
        return {
            "prompt": [prompt for prompt in samples["prompt"]],
            "chosen": samples["chosen"],
            "rejected": samples["rejected"],
        }


    ################################################### main function ###################################################
    
    # deterministic
    import torch
    torch.manual_seed(0)
    import random
    random.seed(0)
    import numpy as np
    np.random.seed(0)
    torch.cuda.manual_seed_all(0)

    dataset = get_data_from_hub()

    dataset = dataset.map(
        return_prompt_and_responses,
        batched=True,
        remove_columns=['id']
    )
    dataset = dataset.shuffle(seed=0)
    dataset = dataset.train_test_split(test_size=0.15, shuffle=True, seed=10)
    
    print(dataset)
    label_pad_token_id = -100 
   
    import wandb
    wandb.init(project="mbart-rosum-dpo-align")
    wandb.log(hyperparameters)

    
    from transformers import TrainingArguments
    from trl import DPOTrainer


    trainer = DPOTrainer(
        model = policy,
        # ref_model = reference,
        model_adapter_name="train2",
        ref_adapter_name="reference",
        args =TrainingArguments(
        output_dir=hyperparameters['output_dir'],
        log_level='info',
        eval_strategy="steps",
        per_device_train_batch_size=hyperparameters['batch_size_train'],
        gradient_accumulation_steps=hyperparameters['gradient_accumulation_steps'],
        per_device_eval_batch_size=hyperparameters['batch_size_val'],
        save_steps=hyperparameters['save_steps'],
        logging_steps=hyperparameters['log_steps'],
        learning_rate=hyperparameters['lr'],
        eval_steps=hyperparameters['eval_steps'],
        num_train_epochs=hyperparameters['epochs'],
        warmup_ratio=hyperparameters['warmup_ratio'],
        lr_scheduler_type=hyperparameters['lr_scheduler'],
        remove_unused_columns=False,
        # load_best_model_at_end=True,
        run_name=hyperparameters['run_name']
        ),
        beta = hyperparameters['beta'],
        train_dataset = dataset["train"],
        eval_dataset = dataset["test"],
        tokenizer = tokenizer,
        max_length = 1024,
        max_prompt_length=1024,
        # peft_config=peft_config,
        max_target_length=256,
        is_encoder_decoder=True
        )

    trainer.train()
    trainer.evaluate(dataset['test'])
    trainer.save_model()
    wandb.finish(exit_code=0)


    

if __name__ == '__main__':

    for k,v in hyperparameters.items():
        logging.info(k + ' : ' + str(v))

    from huggingface_hub import login
    login(token='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')

    policy,tokenizer,config = get_model_lora()


    main(policy,config,tokenizer)
    
    logging.info('Done.')
