from dataset import de_preprocess,en_preprocess,de_vocab,en_vocab,train_dataset
import torch
import torch.nn as nn
from embedding import EmbeddingWithPosition
from multiheadattention import MultiHeadAttention
from config import EMBEDDING_SIZE,ATTENTION_HEAD,BLOCK_NUM,FFN_SIZE
class EncoderBlock(nn.Module):
    def __init__(self,embedding_size,q_k_size,v_size,multi_head_num,ffn_size):
        super(EncoderBlock,self).__init__()
        self.embedding_size=embedding_size
        self.q_k_size=q_k_size
        self.v_size=v_size
        self.multi_head_num=multi_head_num
        self.ffn_size=ffn_size
        self.multiheadattention = MultiHeadAttention(embedding_size,q_k_size,v_size,multi_head_num)
        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_size,embedding_size*ffn_size),
            nn.ReLU(),
            nn.Linear(embedding_size*ffn_size,embedding_size)
        )
        self.layer_norm2 = nn.LayerNorm(embedding_size)
    def forward(self,x,attn_mask):
        x = self.multiheadattention(x,x,attn_mask) + x
        x = self.layer_norm1(x)
        x = self.ffn(x) + x
        x=self.layer_norm2(x)
        return x

if __name__ == '__main__':
    encoder_block = EncoderBlock(EMBEDDING_SIZE,64,64,ATTENTION_HEAD,FFN_SIZE)
    embedding = EmbeddingWithPosition(len(de_vocab),EMBEDDING_SIZE)
    de_sentence,de_ids = de_preprocess(train_dataset[0][0])
    de_ids_tensor = torch.tensor(de_ids,dtype=torch.long).unsqueeze(0)
    de_embedding = embedding(de_ids_tensor)
    attention_mask = torch.zeros(de_embedding.size()[0],de_embedding.size()[1],de_embedding.size()[1])
    result_encoder_block = encoder_block(de_embedding,attention_mask)
    print(result_encoder_block.size())
    print("result_encoder_block:",result_encoder_block)