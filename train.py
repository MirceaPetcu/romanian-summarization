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

logging.basicConfig(filename='training_logs_mbart_1024_torch32_goodaleph_epoch2.log',   
                    filemode='w',               
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')




hyperparameters = {
                'lr':0.00001,
                'batch_size_train': 4,
                'batch_size_val': 2,
                'model_type': 'lora',
                'gradient_accumulation_steps': 4,
                'epochs':1,
                'warmup_steps':400,
                'wd': 0.01,
                'cosine_cycles': 0.5,
                'train_loss_print': 360,
                'evaluation_steps': 4000,
                'data_type':  'Content',
                'model' : 'mt5basero',
                'save_model' : 'model_flant5',
                'save_states': 'mbart_mbart_1024_torch32_goodaleph_epoch2',
                'load_states' : None,
                'is_resuming' : False,
                'load_skip_steps': 0,
                'load_epoch' : 0,
                'local': True,
                'load_model': 'only_model_mt5basero9\pytorch_model.bin',
                'context' : 1024,
                'final_model': 'final_qlora_model_mbart',
                'rank' : 32,
                'alpha': 64,
                'acc_states': 'acc_states_mbart_1024_lm_head_v2',

                }


def check_lm_head_same(model, old_head):
    new_head = model.lm_head.state_dict()["weight"]
    old_head = old_head["weight"]
    if not torch.equal(new_head, old_head):
        print("Weights are different, Good to go")
    else:
        print("Weights are same. Try to add the correct lm_head")


def copy_trained_head(model):
    checkpoint = "facebook/mbart-large-cc25"
    trained_model = MBartForConditionalGeneration.from_pretrained(checkpoint,)
    trained_model.to('cuda:0')
    trained_model.load_state_dict(torch.load(r'acc_states_mbart_1024_lm_head_v2\pytorch_model.bin'))
    trained_head_state_dict = trained_model.lm_head.state_dict()
    check_lm_head_same(model,trained_head_state_dict)
    model.lm_head.load_state_dict(trained_head_state_dict)
    del trained_head_state_dict, trained_model

    return model


def get_model_lora():
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32,
    )
    checkpoint = "facebook/mbart-large-cc25"
    tokenizer = MBartTokenizer.from_pretrained(checkpoint, 
                                            src_lang="ro_RO", tgt_lang="ro_RO"
                                            )
    model = MBartForConditionalGeneration.from_pretrained(checkpoint,
                                                        quantization_config=bnb_config,
                                                        low_cpu_mem_usage = True
                                                        )
    model.config.use_cache = False
    model.supports_gradient_checkpointing = True  
    model.enable_input_require_grads()
    prepare_model_for_kbit_training(model,True,{'use_reentrant':False})
    
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
        r=hyperparameters['rank'], 
        lora_alpha=hyperparameters['alpha'], 
        target_modules=lora_modules+["lm_head"],  
        lora_dropout=0.01, 
        bias="none", 
        task_type="SEQ_2_SEQ_LM",
        init_lora_weights = True,
        use_rslora=True,
    )
    # model = get_peft_model(model, config)
    model = PeftModel.from_pretrained(model, 'mbart_mbart_1024_torch32_goodaleph',is_trainable=True)

    for name, param in model.named_parameters():
        print(name,param.requires_grad,param.dtype,sep=' ',end='\n')
    model.lm_head.weight.requires_grad=False
    print(model.lm_head.weight.requires_grad)
    print(model.lm_head.weight.dtype)
    print(model)
    model.train()
    return model,tokenizer



def get_model_qlora():
        model = 1
        tokenizer = 2
        return model,tokenizer
    
    
def get_model():
    checkpoint = "facebook/mbart-large-cc25"
    tokenizer = T5Tokenizer.from_pretrained(checkpoint,legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint,use_cache=False
                                                     )
    
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    model.supports_gradient_checkpointing = True  
    model.gradient_checkpointing_enable({'use_reentrant':False})
    for name, param in model.named_parameters():
        param.requires_grad = True
    model.lm_head.weight.requires_grad = True


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
        dataset = load_dataset('mirceaPetcu/good-aleph',token='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')
        def change_ceddila_to_comma(batch):
            batch['Summary'] = batch['Summary'].replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")
            batch[hyperparameters['data_type']] = batch[hyperparameters['data_type']].replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")
            return batch
        
        
        
        dataset = dataset.map(change_ceddila_to_comma)

        return dataset

    
    def get_tokenized_data(dataset,tokenizer):

        def preprocess_function(batch):
            # inputs = ["summarize: " + doc for doc in batch[hyperparameters['data_type']]]
            model_inputs = tokenizer(batch[hyperparameters['data_type']], max_length=1024, padding=False, truncation=True)

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

        tokenized_dataset = dataset.map(preprocess_function, batched=True,remove_columns= ['Category', 'Title', 'Content', 'Summary', 'href', 'Source', 'nr_unk_tokens_inputs', 'nr_unk_tokens_labels', 'nr_tokens_content_mt5basero', 'nr_tokens_summary_mt5basero', 'nr_sents_content', 'nr_sents_summary', 'nr_words_content', 'nr_words_summary', 'ratio', 'ratio_tokens_content', 'ratio_token_summary', 'ratio_words_per_sent_content', 'ratio_tokens_per_sent_content', 'real_nr_words_content', 'real_nr_words_summary', 'real_ratio_tokens_words_content', 'real_ratio_tokens_words_summary', 'nr_tokens_content', 'nr_tokens_summary', 'real_ratio_tokens_content', 'real_ratio_token_summary'])
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'labels','attention_mask'])

        return tokenized_dataset


    def eval_model(model, dataloader,epoch):
        model.eval()  
        total = 0.0
        for batch in tqdm(dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            total += loss.item()
            del outputs
            del loss
            del batch
        val_loss = total/len(dataloader)

        return val_loss
    
    
    def save_model(model,epoch,accelerator,history,best_loss):
        if len(history['val_loss']) > 0:
            if history['val_loss'][-1] < best_loss:
                print('SAVED!!!')
                best_loss = history['val_loss'][-1]
                # accelerator.save_state(hyperparameters['acc_states'],safe_serialization=False)
                PeftModel.save_pretrained(
                    model,
                    hyperparameters['save_states'],
                    safe_serialization=False,
                )
        return best_loss
       
            

    
    
    ################################################### main function ###################################################
    
    # deterministic
    import torch
    torch.manual_seed(10)
    import random
    random.seed(10)
    import numpy as np
    np.random.seed(10)
    torch.cuda.manual_seed_all(10)
    model.to('cuda:0')
    print_trainable_parameters(model)
    dataset = get_data_from_hub()
    tokenized_dataset = get_tokenized_data(dataset,tokenizer)
    
    train_dataset =  tokenized_dataset['train']
    valid_dataset =  tokenized_dataset['validation']
   
    ignore_padding_in_loss = True
    label_pad_token_id = -100 
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        padding='longest'
    )
    
   
    from accelerate import Accelerator
    accelerator = Accelerator(gradient_accumulation_steps=hyperparameters['gradient_accumulation_steps']
                              ,step_scheduler_with_optimizer=True
                            #   ,mixed_precision='bf16'
                            )

    import torch
    from transformers import get_cosine_schedule_with_warmup,get_constant_schedule_with_warmup,get_linear_schedule_with_warmup
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hyperparameters['batch_size_train'], shuffle=True,collate_fn=data_collator,pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=hyperparameters['batch_size_val'], shuffle=False,collate_fn=data_collator,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(tokenized_dataset['test'], batch_size=2, shuffle=False,collate_fn=data_collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters['lr'],weight_decay=hyperparameters['wd'])
    # lr_scheduler = get_cosine_schedule_with_warmup(optimizer,
    #                                                 num_warmup_steps=hyperparameters['warmup_steps'],
    #                                                 num_cycles = hyperparameters['cosine_cycles'],
    #                                                 num_training_steps=len(train_loader)*hyperparameters['epochs']//hyperparameters['gradient_accumulation_steps'])
    lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=hyperparameters['warmup_steps'],
                                                    num_training_steps=len(train_loader)*hyperparameters['epochs']//hyperparameters['gradient_accumulation_steps'])
    # lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer,
    #                                                  num_warmup_steps=hyperparameters['warmup_steps'],
    #                                                 )
    model, optimizer,train_loader,val_loader,lr_scheduler = accelerator.prepare(model,
                                                                    optimizer,
                                                                    train_loader,
                                                                    val_loader,
                                                                    lr_scheduler,
                                                                    device_placement=
                                                                    [True,True,True,True,True])

    
    if hyperparameters['is_resuming']:
        accelerator.load_state(hyperparameters['load_states'])
        skip_loader = accelerator.skip_first_batches(train_loader, hyperparameters['load_skip_steps'])

    if hyperparameters['local']:
        logging.info('Device: ' + str(accelerator.device))
    print('Device: ' + str(accelerator.device))
    history = {'val_loss':[],'train_loss':[]}
    best_loss = 99999

    try:
        stepi = 0
        avg_loss = 0.0
        train_loss = 0.0
        for epoch in range(hyperparameters['epochs']):
            logging.info(f'Epoch: {epoch}')
            model.train()
            if hyperparameters['is_resuming']:
                if epoch == hyperparameters['load_epoch']:
                    loader = skip_loader
                else:
                    loader = train_loader
            else:
                loader = train_loader
                
            for batch in tqdm(loader):
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    avg_loss += loss.item()
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if (stepi % (hyperparameters['gradient_accumulation_steps']-1)==0 or stepi == (len(loader)-1)) and stepi % len(loader) != 0:
                    train_loss = train_loss + (avg_loss/accelerator.gradient_accumulation_steps)
                    real_batch_loss = avg_loss/accelerator.gradient_accumulation_steps
                    if stepi % hyperparameters['train_loss_print'] == 0:
                        batch_size = hyperparameters['batch_size_train']*hyperparameters['gradient_accumulation_steps']
                        if hyperparameters['local']:
                            logging.info(f'Train loss for batch of {batch_size} samples: {real_batch_loss} at step {stepi} from epoch {epoch}')
                        print(f'Train loss for batch of {batch_size} samples: {real_batch_loss} at step {stepi} from epoch {epoch}')
                        history['train_loss'].append(avg_loss)
                    avg_loss = 0.0

                if stepi % hyperparameters['evaluation_steps'] == 0:
                    train_loss = train_loss/(hyperparameters['evaluation_steps']/accelerator.gradient_accumulation_steps)
                    logging.info(f"Train loss for {hyperparameters['evaluation_steps']} : {train_loss}")
                    print(f"Train loss for {hyperparameters['evaluation_steps']} : {train_loss}")
                    train_loss = 0.0
                    val_loss=eval_model(model,val_loader,epoch)
                    history['val_loss'].append(val_loss)
                    if hyperparameters['local']:
                        logging.info(f'Val Loss:{val_loss} at step {stepi} from epoch {epoch}')
                    print(f'Val Loss:{val_loss} at step {stepi} from epoch {epoch}')
                    model.train()
                    best_loss = save_model(model,epoch,accelerator,history,best_loss)
                    model.train()

                stepi += 1
        
    except Exception as e:
        logging.error(str(e))
        print(str(e))
        save_model(model,epoch,accelerator,history,best_loss)
    finally:
        val_loss = eval_model(model,val_loader,epoch)
        history['val_loss'].append(val_loss)
        logging.info(f'Final val loss :{val_loss}')
        best_loss = save_model(model,epoch,accelerator,history,best_loss)
        for k in history:
            logging.info(k)
            for v in history[k]:
                logging.info(v)
        logging.info(f'Best_loss : {best_loss}')

    

if __name__ == '__main__':

    if hyperparameters['local']:
        for k,v in hyperparameters.items():
            logging.info(k + ' : ' + str(v))

    if hyperparameters['model_type'] == 'qlora':
        model,tokenizer = get_model_qlora()
    elif hyperparameters['model_type'] == 'lora':
        model,tokenizer = get_model_lora()
    else:
        model,tokenizer = get_model()

    main(model,tokenizer)
    
    if hyperparameters['local']:
        logging.info('Done.')
