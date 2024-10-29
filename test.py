import bitsandbytes as bnb
import os
import torch
from tqdm import tqdm
import torch.nn as nn
from transformers import BitsAndBytesConfig,Seq2SeqTrainingArguments,Seq2SeqTrainer
import torch
from peft import PeftModel
import bitsandbytes as bnb
import evaluate
import numpy as np
from datasets import load_dataset,DatasetDict, concatenate_datasets
from transformers import (
    T5ForConditionalGeneration,
    MT5ForConditionalGeneration,
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

logging.basicConfig(filename='testing_logs_mbart_torch32.log',   
                    filemode='w',               
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')




hyperparameters = {
                'batch_size_train': 8,
                'batch_size_val': 8,
                'batch_size_test': 8,
                'qlora': True,
                'data_type': 'Content',
                'model' : 'mt5basero6',
                'load_states' : 'states_mt5basero6',
                'local': True,
                'load_model': r'only_model_mt5basero6\pytorch_model.bin',
                'subset': 'all',
                'metric' : None
                }

from transformers import (
    T5ForConditionalGeneration,
    MT5ForConditionalGeneration,
    MBartTokenizer,
    MBartForConditionalGeneration)
def get_model_lora():
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    )
    checkpoint = "facebook/mbart-large-cc25"
    tokenizer = MBartTokenizer.from_pretrained(checkpoint, src_lang="ro_RO", tgt_lang="ro_RO")
    model = MBartForConditionalGeneration.from_pretrained(checkpoint,
                                                        # quantization_config=bnb_config,
                                                        low_cpu_mem_usage = True
                                                        )
    model = PeftModel.from_pretrained(model, 'mbart_mbart_1024_torch32',is_trainable=False)
    model = model.merge_and_unload(safe_merge=True)
    return model,tokenizer


def get_model_qlora():
        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
        )

        checkpoint = 'dumitrescustefan/mt5-large-romanian'
        tokenizer = T5Tokenizer.from_pretrained(checkpoint,legacy=False)
        model = T5ForConditionalGeneration.from_pretrained(checkpoint,use_cache=False,
                                                     quantization_config=bnb_config,
                                                     trust_remote_code=True
                                                    )
        
   
        model.config.use_cache = False
        model.supports_gradient_checkpointing = True  
        model.gradient_checkpointing_enable({'use_reentrant':False})
        model.resize_token_embeddings(len(tokenizer))
        model = prepare_model_for_kbit_training(model)

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

        lora_modules = find_all_linear_names(model)
        config = LoraConfig(
            r=8, 
            lora_alpha=16, 
            target_modules=lora_modules+['lm_head'],  
            lora_dropout=0.03, 
            bias="none", 
            task_type="SEQ_2_SEQ_LM",
            init_lora_weights = True
        )
        model = get_peft_model(model, config)
        return model,tokenizer
    
    
def get_model():
    checkpoint = "facebook/mbart-large-cc25"
    trained_model = MBartForConditionalGeneration.from_pretrained(checkpoint)
    trained_model.load_state_dict(torch.load(r'mbart-lmhead-trained\pytorch_model.bin'))
    
    # model.resize_token_embeddings(len(tokenizer))
    
    for name, param in model.named_parameters():
        param.requires_grad = False
    model.lm_head.weight.requires_grad = False

    return model,tokenizer
        

def main(model,tokenizer):

    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        if hyperparameters['local']:
            logging.info(
                f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
            )
        print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


    
    def get_data_from_hub():
        dataset = load_dataset('mirceaPetcu/good-samples-less1800words',token='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')
        return dataset

    
    def get_tokenized_data(dataset,tokenizer):

        def preprocess_function(batch):
            model_inputs = tokenizer(batch[hyperparameters['data_type']], max_length=768, padding=False, truncation=True)

            labels = tokenizer(text_target=batch['Summary'], max_length=256, padding=False, truncation=True)

            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
            model_inputs["labels"] = labels["input_ids"]
            
            return model_inputs
        
        def change_ceddila_to_comma(batch):
            batch['Summary'] = batch['Summary'].replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")
            batch[hyperparameters['data_type']] = batch[hyperparameters['data_type']].replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")
            return batch
        
        dataset = dataset.map(change_ceddila_to_comma)

        tokenized_dataset = dataset.map(preprocess_function, batched=True,remove_columns=['Category', 'Title', 'Content', 'Summary', 'href', 'Source', '__index_level_0__'])
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'labels','attention_mask'])

        return tokenized_dataset

    # import nltk
    # def postprocess_text(preds, labels):
    #     preds = [pred.strip() for pred in preds]
    #     labels = [label.strip() for label in labels]

    #     # ROUGE expects a newline after each sentence
    #     # change to spacy
    #     preds = ["\n".join(nltk.sent_tokenize(pred,language='romanian')) for pred in preds]
    #     labels = ["\n".join(nltk.sent_tokenize(label,language='romanian')) for label in labels]

    #     return preds, labels


    def eval_model(model, dataloader):
        model.eval()  
        total = 0
        nr_samples = 0
        for batch in tqdm(dataloader):
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
                total += loss.item() * batch['input_ids'].size(0)
                nr_samples += batch['input_ids'].size(0)
                
        val_loss = total/nr_samples
        return val_loss
    
    
   
    
    
    ################################################### main function ###################################################
    
    # deterministic
    import torch
    torch.manual_seed(0)
    import random
    random.seed(0)
    import numpy as np
    np.random.seed(0)
    torch.cuda.manual_seed_all(0)

    print_trainable_parameters(model)
    dataset = get_data_from_hub()
    tokenized_dataset = get_tokenized_data(dataset,tokenizer)
    
    train_dataset =  tokenized_dataset['train']
    valid_dataset =  tokenized_dataset['validation']
   
    ignore_padding_in_loss = True
    label_pad_token_id = -100 if ignore_padding_in_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        padding='longest'
    )
    
   
    from accelerate import Accelerator
    accelerator = Accelerator()

    import torch
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hyperparameters['batch_size_train'], shuffle=False,collate_fn=data_collator,pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=hyperparameters['batch_size_val'], shuffle=False,collate_fn=data_collator,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(tokenized_dataset['test'], batch_size=hyperparameters['batch_size_test'], shuffle=False,collate_fn=data_collator,pin_memory=True)

    model, train_loader,val_loader,test_loader = accelerator.prepare(model,
                                                        train_loader,
                                                        val_loader,
                                                        test_loader,
                                                        device_placement=
                                                        [True,True,True,True])

    
    if hyperparameters['local']:
        logging.info('Device: ' + str(accelerator.device))
    print('Device: ' + str(accelerator.device))

    if hyperparameters['subset'] == 'all':
        loss = eval_model(model,val_loader)
        logging.info(f'Val loss : {loss}')
        loss = eval_model(model,test_loader)
        logging.info(f'Test loss : {loss}')
        loss = eval_model(model,train_loader)
        logging.info(f'Train loss : {loss}')
    elif hyperparameters['subset'] == 'train':
        loss = eval_model(model,train_loader)
    elif hyperparameters['subset'] == 'val':
        loss = eval_model(model,val_loader)
    else:
        loss = eval_model(model,test_loader)
    
    if hyperparameters['subset'] != 'all':
        logging.info(f"{hyperparameters['subset']} loss : {loss}")


    if hyperparameters['metric'] is not None:
        pass
    

if __name__ == '__main__':


    if hyperparameters['qlora']:
        model,tokenizer = get_model_lora()
    else:
        model,tokenizer = get_model()

    main(model,tokenizer)
    
    if hyperparameters['local']:
        logging.info('Done.')
