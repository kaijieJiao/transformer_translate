import torch
import torch.nn as nn
from embedding import EmbeddingWithPosition
from dataset import en_preprocess,de_preprocess,de_vocab,en_vocab,train_dataset,PAD_ID
from multiheadattention import MultiHeadAttention
from encoder_block import EncoderBlock
from config import *
class Encoder(nn.Module):

    def __init__(self,vocab_size,emb_size,q_k_size,v_size,head,f_size,nblocks):
        super(Encoder,self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.q_k_size = q_k_size
        self.v_size = v_size
        self.head = head
        self.f_size = f_size
        self.nblocks = nblocks
        self.encoder = nn.ModuleList([
            EncoderBlock(emb_size,q_k_size,v_size,head,f_size) for _ in range(nblocks)
        ])
        self.embding = EmbeddingWithPosition(vocab_size,emb_size)
    def forward(self,x):
        atten_mask = (x==PAD_ID).unsqueeze(1).expand(-1,x.size()[1],x.size()[1]).to(DEVICE)
        x = self.embding(x)
        for i in range(self.nblocks):
            x = self.encoder[i](x,atten_mask)
        return x
    
if __name__ == "__main__":
    encoder =Encoder(len(de_vocab),EMBEDDING_SIZE,128,128,ATTENTION_HEAD,FFN_SIZE,BLOCK_NUM)
    _, de_ids = de_preprocess(train_dataset[0][0])
    if len(de_ids)<SEQ_MAX_LEN:
        de_ids = de_ids + [PAD_ID]*(SEQ_MAX_LEN-len(de_ids))
    de_ids_tensor=torch.tensor(de_ids,dtype=torch.long).unsqueeze(0)
    
    result_encoder = encoder(de_ids_tensor)
    print(result_encoder.shape)
    print('result_encoder:',result_encoder)
