import torch
import torch.nn as nn
from embedding import EmbeddingWithPosition
from encoder import Encoder
from decoder_block import DecoderBlock
from dataset import de_preprocess,en_preprocess,de_vocab,en_vocab,train_dataset,PAD_ID
from config import *

class Decoder(nn.Module):
    def __init__(self,vocab_szie,emb_size,q_k_size,v_size,head,f_size,nblocks):
        super(Decoder,self).__init__()
        self.vocab_szie = vocab_szie
        self.emb_size = emb_size
        self.q_k_size = q_k_size
        self.v_size = v_size
        self.head = head
        self.f_size = f_size
        self.nblocks = nblocks

        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(emb_size,q_k_size,v_size,head,f_size) for _ in range(nblocks)]
        )
        self.emb = EmbeddingWithPosition(vocab_szie,emb_size)

        self.out_linear = nn.Linear(emb_size,vocab_szie)
        # self.softmax = nn.Softmax(dim=-1)
    def forward(self,dec_x,encoder_z,encoder_x):
        # x: (batch_size,seq_len)
        # encoder_z: (batch_size,seq_len,emb_size)
        # first_attn_mask: (batch_size,seq_len,seq_len)
        # second_attn_mask: (batch_size,seq_len,seq_len)
       
        
        atten_mask1= (dec_x==PAD_ID).unsqueeze(1).expand(dec_x.size()[0],dec_x.size()[1],dec_x.size()[1])
        atten_mask1=atten_mask1 | torch.triu(torch.ones(dec_x.size()[1],dec_x.size()[1]),diagonal=1).bool().unsqueeze(0).expand(dec_x.size()[0],-1,-1).to(DEVICE)
        atten_mask2= (encoder_x==PAD_ID).unsqueeze(1).expand(encoder_x.size(0),dec_x.size(1),encoder_x.size(1)).to(DEVICE)
        # print(atten_mask1.size(),atten_mask2.size())
        dec_x = self.emb(dec_x)
        for decoder_block in self.decoder_blocks:
            dec_x = decoder_block(dec_x,encoder_z,atten_mask1,atten_mask2)
        return self.out_linear(dec_x)

if __name__ == '__main__':
    _,de_ids =de_preprocess(train_dataset[0][0])
    _,en_ids =en_preprocess(train_dataset[0][1])

    if len(de_ids)<SEQ_MAX_LEN:
        de_ids = de_ids + [PAD_ID] * (SEQ_MAX_LEN-len(de_ids))
    if len(en_ids)<SEQ_MAX_LEN:
        en_ids = en_ids + [PAD_ID] * (SEQ_MAX_LEN-len(en_ids))
    en_ids_tensor=torch.tensor(en_ids).unsqueeze(0)
    de_ids_tensor=torch.tensor(de_ids).unsqueeze(0)
    
    encoder =Encoder(len(de_vocab),EMBEDDING_SIZE,128,128,ATTENTION_HEAD,FFN_SIZE,BLOCK_NUM)
    encoder_z =encoder(de_ids_tensor)
    decoder = Decoder(len(en_vocab),EMBEDDING_SIZE,128,128,ATTENTION_HEAD,FFN_SIZE,BLOCK_NUM)
    result_decoder = decoder(en_ids_tensor,encoder_z,de_ids_tensor)
    print(result_decoder.size())
    print('result_decoder:',result_decoder)