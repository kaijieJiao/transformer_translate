import torch
import torch.nn as nn
from multiheadattention import MultiHeadAttention
from embedding import EmbeddingWithPosition
from dataset import de_preprocess,en_preprocess,de_vocab,en_vocab,train_dataset,PAD_ID
from config import *
from encoder import Encoder
from torch.nn import TransformerDecoderLayer,TransformerDecoder
class DecoderBlock(nn.Module):

    def __init__(self,embedding_size,q_k_size,v_size,multi_head_num,ffn_size):
        super(DecoderBlock,self).__init__()
        self.embedding_size=embedding_size
        self.q_k_size=q_k_size
        self.v_size=v_size
        self.multi_head_num=multi_head_num
        self.ffn_size=ffn_size
        # self.encoder = Encoder(encoder_embedding_size,encoder_q_k_size,encoder_v_size,encoder_multi_head_num,encoder_ffn_size,encoder_nblocks)
        self.multiheadattention1 = MultiHeadAttention(embedding_size,q_k_size,v_size,multi_head_num)
        self.layer_norm1 = nn.LayerNorm(embedding_size)

        self.multiheadattention2 = MultiHeadAttention(embedding_size,q_k_size,v_size,multi_head_num)
        self.layer_norm2 = nn.LayerNorm(embedding_size)

        self.ffn = nn.Sequential(
            nn.Linear(embedding_size,embedding_size*ffn_size),
            nn.ReLU(),
            nn.Linear(embedding_size*ffn_size,embedding_size)
        )
        self.layer_norm3 = nn.LayerNorm(embedding_size)
    def open_kvcache(self):
        self.multiheadattention1.set_kvcache(kv_cache_type='self_attention')
        self.multiheadattention2.set_kvcache(kv_cache_type='cross_attention')
    def close_kvcache(self):
        self.multiheadattention1.set_kvcache(kv_cache_type='')
        self.multiheadattention2.set_kvcache(kv_cache_type='')
    def forward(self,x,encoder_z,atten_mask1,atten_mask2):
        # x: (batch_size,seq_len,embedding_size)
        x = self.layer_norm1(self.multiheadattention1(x,x,atten_mask1)+x)
        # encoder_result [batch_size,seq_len,embedding_size]
        z = self.layer_norm2(self.multiheadattention2(x,encoder_z,atten_mask2)+x)
        return self.layer_norm3(self.ffn(z)+z)
    
if __name__ == "__main__":
    _,de_ids = de_preprocess(train_dataset[0][0])
    _,en_ids = en_preprocess(train_dataset[0][1])

    
    if len(de_ids)<SEQ_MAX_LEN:
        de_ids = de_ids + [PAD_ID]*(SEQ_MAX_LEN-len(de_ids))
    if len(en_ids)<SEQ_MAX_LEN:
        en_ids = en_ids + [PAD_ID]*(SEQ_MAX_LEN-len(en_ids))
    de_ids_tensor = torch.tensor(de_ids,dtype=torch.long)
    en_ids_tensor = torch.tensor(en_ids,dtype=torch.long)
    de_ids_tensor = de_ids_tensor.unsqueeze(0)
    en_ids_tensor = en_ids_tensor.unsqueeze(0)
    atten_mask1 = (en_ids_tensor==PAD_ID).unsqueeze(1).expand(en_ids_tensor.shape[0],en_ids_tensor.shape[1],en_ids_tensor.shape[1])
    atten_mask1= atten_mask1 | torch.triu(torch.ones(en_ids_tensor.shape[1],en_ids_tensor.shape[1]),diagonal=1).bool().unsqueeze(0).expand(en_ids_tensor.shape[0],-1,-1)
    atten_mask2 =(de_ids_tensor==PAD_ID).unsqueeze(1).expand(de_ids_tensor.shape[0],de_ids_tensor.shape[1],de_ids_tensor.shape[1])
    print(atten_mask1.size(),atten_mask2.size())

    en_embedding = EmbeddingWithPosition(len(en_vocab),EMBEDDING_SIZE)

    encoder = Encoder(len(de_vocab),EMBEDDING_SIZE,128,128,ATTENTION_HEAD,FFN_SIZE,BLOCK_NUM)
    encoder_z = encoder(de_ids_tensor,atten_mask2)
    en_embeddings = en_embedding(en_ids_tensor)
    decoder = DecoderBlock(EMBEDDING_SIZE,128,128,ATTENTION_HEAD,FFN_SIZE)
    result_decoder_block = decoder(en_embeddings,encoder_z,atten_mask1,atten_mask2)
    print(result_decoder_block.size())
    print('result_decoder_block:',result_decoder_block)