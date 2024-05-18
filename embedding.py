import torch 
import torch.nn as nn
from dataset import de_vocab,en_vocab,de_preprocess,en_preprocess,train_dataset
import math
from config import EMBEDDING_SIZE

class EmbeddingWithPosition(nn.Module):
    def __init__(self,vocab_size,embedding_size,dropout=0.1,seq_max_len=5000):
        super(EmbeddingWithPosition,self).__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_size)
        # position embedding size [seq_max_len,embedding_size]
        position_idx = torch.arange(0,seq_max_len,dtype=torch.float).unsqueeze(-1)  #[[0],[1],[2],,,[seq_max_len-1]]
        position_emb_fill = position_idx / torch.pow(10000,torch.arange(0,embedding_size,2)/embedding_size)
        position_embedding=torch.zeros(seq_max_len,embedding_size)
        position_embedding[:,::2]=torch.sin(position_emb_fill)
        position_embedding[:,1::2]=torch.cos(position_emb_fill)
        self.register_buffer('position_embedding',position_embedding)

        #防止过拟合
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        # x  [batch_size,seq_len]
        x_emb=self.embedding(x)
        seq_len=x.size(1)
        embeddingwithposition = x_emb+self.position_embedding.unsqueeze(0)[:,:seq_len,:]
        return self.dropout(embeddingwithposition)

if __name__ == "__main__":
    embdding = EmbeddingWithPosition(len(de_vocab),EMBEDDING_SIZE)

    de_sentence , en_sentence = train_dataset[0][0],train_dataset[0][1]
    de_tokens,de_ids = de_preprocess(de_sentence)
    de_ids_tensor =torch.tensor(de_ids,dtype=torch.long)

    de_emb = embdding(de_ids_tensor.unsqueeze(0))
    print(de_emb.size())
    print(de_emb)