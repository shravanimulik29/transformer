import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path # from abosulte to relative path

#config - model details
#ds - datasets
#lang - language

def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds,lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    
    #if tokenizer does not exits we create it
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer
    
def get_ds(config):
    #from hugging face
    ds_raw = load_dataset('opus_books',f'{config["lang_src"]}-{config["lang_tgt"]}',split='train')

    #build tokenizers
    tokenizer_src = get_or_build_tokenizer(config,ds_raw,config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config,ds_raw,config['lang_tgt'])

    #spliting
    train_ds_size = int(0.9* len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw,[train_ds_size,val_ds_size])

    


