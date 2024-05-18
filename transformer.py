import torch
import torch.nn as nn
from dataset import de_vocab,en_vocab,de_preprocess,en_preprocess,PAD_ID,train_dataset
from encoder import Encoder
from decoder import Decoder
from config import *
class Transformer(nn.Module):
    def __init__(self,enc_vocab_size,dec_vocab_size,emb_size,q_k_size,v_size,f_size,head,nblocks):
        super(Transformer,self).__init__()
        self.enc_vocab_size=enc_vocab_size
        self.dec_vocab_size=dec_vocab_size
        self.emb_size=emb_size
        self.q_k_size=q_k_size
        self.v_size=v_size
        self.f_size=f_size
        self.head=head
        self.nblocks=nblocks
        self.decoder=Decoder(dec_vocab_size,emb_size,q_k_size,v_size,f_size,head,nblocks)
        self.encoder=Encoder(enc_vocab_size,emb_size,q_k_size,v_size,f_size,head,nblocks)

    def forward(self,enc_x,dec_x):
        # x: (batch_size,seq_len1)
        # y: (batch_size,seq_len2)
        enc_z=self.encoder(enc_x)
        result=self.decoder(dec_x,enc_z,enc_x)
        return result
    
if __name__=='__main__':
    model=Transformer(len(de_vocab),len(en_vocab),EMBEDDING_SIZE,128,128,ATTENTION_HEAD,FFN_SIZE,BLOCK_NUM)
    de_sentence,de_ids =de_preprocess(train_dataset[0][0])
    en_sentence,en_ids =en_preprocess(train_dataset[0][1])
    
    # print(de_ids.size(),en_ids.size())

    if len(de_ids)<SEQ_MAX_LEN:
        de_ids = de_ids + [PAD_ID] * (SEQ_MAX_LEN-len(de_ids))
    if len(en_ids)<SEQ_MAX_LEN:
        en_ids = en_ids + [PAD_ID] * (SEQ_MAX_LEN-len(en_ids))
    en_ids=torch.tensor(en_ids).unsqueeze(0)
    de_ids=torch.tensor(de_ids).unsqueeze(0)
    result = model(de_ids,en_ids)
    print(result.size())
    print('transformer output:',result)