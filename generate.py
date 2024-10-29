import os
import torch
from tqdm import tqdm
import torch.nn as nn
from transformers import BitsAndBytesConfig,Seq2SeqTrainingArguments,Seq2SeqTrainer
import torch
from peft import PeftModel
# import bitsandbytes as bnb
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

logging.basicConfig(filename='generating_logs_mbart1024torch32_30x.log',   
                    filemode='w',               
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')




hyperparameters = {
                'batch_size_train': 8,
                'batch_size_val': 8,
                'batch_size_test': 4,
                'qlora': True,
                'data_type': 'NewContent512',
                'model' : 'mt5basero2',
                'load_states' : 'states_mt5basero6',
                'local': True,
                'load_model': r'only_model_mt5basero2\pytorch_model.bin',
                'subset': 'all',
                'metric' : None,
                'outputs_file' : 'generated_references30.json'
                }

import spacy
nlp = spacy.load('ro_core_news_lg')


from transformers import (
    T5ForConditionalGeneration,
    T5ForConditionalGeneration,
    MBartTokenizer,
    MBartForConditionalGeneration)


def get_model_lora():
    checkpoint = "facebook/mbart-large-cc25"
    tokenizer = MBartTokenizer.from_pretrained(checkpoint, src_lang="ro_RO", tgt_lang="ro_RO")
    model = MBartForConditionalGeneration.from_pretrained(checkpoint,
                                                        # low_cpu_mem_usage = True
                                                        )
    
    model = PeftModel.from_pretrained(model, r'states_mbart_textrank512',is_trainable=False)
    model = model.merge_and_unload(safe_merge=True)
    model.eval()
    # print(model)
    return model,tokenizer
    
    
def get_model():
    checkpoint = "google/flan-t5-large"
    tokenizer = T5Tokenizer.from_pretrained(checkpoint, 
                                            )
    model = T5ForConditionalGeneration.from_pretrained(checkpoint,
                                                       low_cpu_mem_usage = True
                                                        )
    
    model = PeftModel.from_pretrained(model, 'archive',is_trainable=False)
    model = model.merge_and_unload(safe_merge=True)
    model.eval()
    # print(model)
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
        dataset = load_dataset('mirceaPetcu/textrank-mbart-real',token='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')
        def change_ceddila_to_comma(batch):
            batch['Summary'] = batch['Summary'].replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")
            batch[hyperparameters['data_type']] = batch[hyperparameters['data_type']].replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")
            return batch
        dataset = dataset.map(change_ceddila_to_comma)

        return dataset

    
    def get_tokenized_data(dataset,tokenizer):

        def preprocess_function(batch):
            model_inputs = tokenizer(batch[hyperparameters['data_type']], max_length=1024, padding=True, truncation=True)

            labels = tokenizer(text_target=batch['Summary'], max_length=256, padding=True, truncation=True)

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

        # tokenized_dataset = dataset.map(preprocess_function, batched=True,remove_columns=['Category', 'Title', 'Content', 'Summary', 'Source','href'])
        tokenized_dataset = dataset.map(preprocess_function, batched=True,remove_columns= ['Category', 'Title', 'Content', 'Summary', 'href', 'Source', 'nr_unk_tokens_inputs', 'nr_unk_tokens_labels', 'nr_tokens_content_mt5basero', 'nr_tokens_summary_mt5basero', 'nr_sents_content', 'nr_sents_summary', 'nr_words_content', 'nr_words_summary', 'ratio', 'ratio_tokens_content', 'ratio_token_summary', 'ratio_words_per_sent_content', 'ratio_tokens_per_sent_content', 'real_nr_words_content', 'real_nr_words_summary', 'real_ratio_tokens_words_content', 'real_ratio_tokens_words_summary', 'nr_tokens_content', 'nr_tokens_summary', 'real_ratio_tokens_content', 'real_ratio_token_summary'])
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'labels','attention_mask'])

        return tokenized_dataset

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # ROUGE expects a newline after each sentence
        # change to spacy
        preds = ["\n".join([str(s) for s in list(nlp(pred).sents)]) for pred in preds]
        labels = ["\n".join([str(s) for s in list(nlp(label).sents)]) for label in labels]

        return preds, labels

    def generate_and_eval(model,dataloader):
        # dataloader = dataloader.train_test_split(test_size=0.01110744528758386,shuffle=True, seed=0)
        # dataloader = dataloader['test']
        import evaluate
        rouge_score = evaluate.load("rouge")
        bert_score = evaluate.load("bertscore")
        result_bert = {'precision': [], 'recall': [],'f1': []}
        model.eval()
        nr = 0
        generated = []
        references = []
        for sample in tqdm(dataloader):
            with torch.no_grad():
                # if nr == 400:
                #     break
                nr += 1
                model_inputs = tokenizer(sample['NewContent512'], max_length=512, padding=True, truncation=True,return_tensors='pt').to('cuda')
                generated_tokens = accelerator.unwrap_model(model).generate(
                    model_inputs["input_ids"],
                    attention_mask= model_inputs["attention_mask"],
                    decoder_start_token_id=tokenizer.lang_code_to_id["ro_RO"],
                    min_new_tokens=35,
                    max_new_tokens=250,
                    do_sample= False,  
                    no_repeat_ngram_size=5,
                    num_beams=4,
                    # encoder_no_repeat_ngram_size=5,
                    length_penalty=0.5,
                    # early_stopping=True,
                    # encoder_repetition_penalty=1.5
#                     num_beams = 12,
#                     length_penalty=0.8,
#                     temperature = 0.95,
#                     num_return_sequences=3,
#                     no_repeat_ngram_size=10
                    )

               
                generated_tokens = generated_tokens.cpu().numpy()

                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
                generated.append(decoded_preds[0])
                references.append(sample['Summary'])
                with open('dpo_for_test7.txt','a',encoding='utf-8') as f:
                    f.write('Content:\n\n')
                    f.write(sample['Content'])
                    f.write('\n\n')
                    f.write('Generated:\n\n')
                    f.write(decoded_preds[0])
                    f.write('\n\n')
                    f.write('References:\n\n')
                    f.write(sample['Summary'])
                    f.write('\n\n')
                    f.write('------------------------------------------\n')

        print(nr)
        # metrics
        result_bert = bert_score.compute(predictions=generated, references=references, lang='ro',idf=True)
        print(sum(result_bert['precision'])/len(result_bert['precision']))
        print(sum(result_bert['recall'])/len(result_bert['recall']))
        print(sum(result_bert['f1'])/len(result_bert['f1']))
        print('bert no tfidf')
        result_bert_noidf = bert_score.compute(predictions=generated, references=references, lang='ro',idf=False)
        print(sum(result_bert_noidf['precision'])/len(result_bert_noidf['precision']))
        print(sum(result_bert_noidf['recall'])/len(result_bert_noidf['recall']))
        print(sum(result_bert_noidf['f1'])/len(result_bert_noidf['f1']))
        
        generated, references = postprocess_text(generated,references)
        # Compute metrics
        result = rouge_score.compute(predictions=generated,references=references)
        # Extract the median ROUGE scores
        result = {key: value for key, value in result.items()}
        result = {k: round(v, 4) for k, v in result.items()}
       
        print(result)
                
                
                
        # Compute metrics
        

    def eval_model(model, dataloader):
        model.eval()  
        total = 0
        nr_samples = 0
        for batch in tqdm(dataloader):
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
                total += loss.item() 
                
        val_loss = total/len(dataloader)
        print(val_loss)
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
    # tokenized_dataset = get_tokenized_data(dataset,tokenizer)
    
    # train_dataset =  tokenized_dataset['train']
    # valid_dataset =  tokenized_dataset['validation']
   
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
    
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hyperparameters['batch_size_train'], shuffle=False,collate_fn=data_collator,pin_memory=True)
    # val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=hyperparameters['batch_size_val'], shuffle=False,collate_fn=data_collator,pin_memory=True)
    # test_loader = torch.utils.data.DataLoader(tokenized_dataset['test'], batch_size=hyperparameters['batch_size_test'], shuffle=False,collate_fn=data_collator,pin_memory=True)

    # model, train_loader,val_loader,test_loader = accelerator.prepare(model,
    #                                                     train_loader,
    #                                                     val_loader,
    #                                                     test_loader,
    #                                                     device_placement=
    #                                                     [True,True,True,True])

    
    if hyperparameters['local']:
        logging.info('Device: ' + str(accelerator.device))
    model.to('cuda:0')
    print('Device: ' + str(accelerator.device))
    # eval_model(model,val_loader)
    # eval_model(model,test_loader)

    generate_and_eval(model,dataset['test'])

   
    

if __name__ == '__main__':
    
    print(hyperparameters['data_type'])
    if hyperparameters['qlora']:
        model,tokenizer = get_model_lora()
    else:
        model,tokenizer = get_model()

    main(model,tokenizer)
    
    if hyperparameters['local']:
        logging.info('Done.')