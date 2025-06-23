from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __ini__(self,ds,tokenizer_src,tokenizer_tgt,src_lang,tgt_lang,seq_len) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.Tensor([tokenizer_src.token_to_id(['[SOS]'])] ,dtype=torch.int64)

        self.eos_token = torch.Tensor([tokenizer_src.token_to_id(['[EOS]'])] ,dtype=torch.int64)

        self.pad_token = torch.Tensor([tokenizer_src.token_to_id(['[PAD]'])] ,dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index:Any) -> Any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        #it will give input ids
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        #padding token to reach padding len(-2 becase of eos and sos)
        enc_num_padding_toens = self.seq_len- len(enc_input_tokens) -2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) -1 #only eos

        if enc_num_padding_toens < 0 or dec_num_padding_tokens <0 :
            raise ValueError('Sentence is too large')
        
        






