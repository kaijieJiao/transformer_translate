import torch
import torch.nn as nn
from dataset import de_preprocess,en_preprocess,train_dataset,de_vocab,en_vocab
from embedding import EmbeddingWithPosition
from config import EMBEDDING_SIZE,ATTENTION_HEAD
class MultiHeadAttention(nn.Module):
    def __init__(self,embedding_size,qk_size,v_size,num_head):
        super(MultiHeadAttention,self).__init__()
        self.num_head=num_head
        self.q_k_size=qk_size
        self.v_size=v_size
        self.embedding_size=embedding_size
        self.w_q = nn.Linear(embedding_size,num_head*qk_size)
        self.w_k = nn.Linear(embedding_size,num_head*qk_size)
        self.w_v = nn.Linear(embedding_size,num_head*v_size)
        self.w_o = nn.Linear(num_head*v_size,embedding_size)
        self.soft_max=nn.Softmax(dim=-1)
        

    def forward(self,x,x_k_v,attention_mask=None):
        # x=y [batch_size,sequence_len,embedding_size]
        q=self.w_q(x)  #q  [batch_size,sequence_len,num_head*qk_size]
        k=self.w_k(x_k_v)  #k  [batch_size,sequence_len,num_head*qk_size]
        v=self.w_v(x_k_v)  #v  [batch_size,sequence_len,num_head*v_size]
        q=q.view(q.size()[0],q.size()[1],self.num_head,-1).transpose(1,2)   #q [batch_size,num_head,sequence_len,qk_size]
        k=k.view(k.size()[0],k.size()[1],self.num_head,-1).transpose(1,2).transpose(2,3)   #k [batch_size,num_head,sequence_len,qk_size]
        v=v.view(v.size()[0],v.size()[1],self.num_head,-1).transpose(1,2)   #v [batch_size,num_head,sequence_len,v_size]
        scores=torch.matmul(q,k)/(self.q_k_size**0.5)    #scores [batch_size,num_head,sequence_len_q,sequence_len_k]
        # attention_mask [batch_size,sequence_len,sequence_len]
        attention_mask=attention_mask.unsqueeze(1).expand(-1,self.num_head,-1,-1)
        scores=scores.masked_fill(attention_mask,-1e9) #scores [batch_size,num_head,sequence_len,sequence_len]
        atten=self.soft_max(scores)
        z=torch.matmul(atten,v)    #z  [batch_size,num_head,sequence_len,v_size]
        z=z.transpose(1,2)
        return self.w_o(z.reshape(z.size()[0],z.size()[1],-1))
if __name__=='__main__':
    embdding=EmbeddingWithPosition(len(de_vocab),EMBEDDING_SIZE)
    multiheadattention=MultiHeadAttention(EMBEDDING_SIZE,128,128,ATTENTION_HEAD)
    de_sentence = train_dataset[0][0]
    _,de_ids = de_preprocess(de_sentence)
    de_ids_tensor = torch.tensor(de_ids,dtype=torch.long).unsqueeze(0)
    de_embeddings=embdding(de_ids_tensor)
    atten_mask = torch.zeros(de_embeddings.size()[0],de_embeddings.size()[1],de_embeddings.size()[1])
    attention = multiheadattention(de_embeddings,de_embeddings,atten_mask)
    print('attention_size:',attention.size())
    print('attention_size:',attention.shape)

    print('multiheadattention:',attention)