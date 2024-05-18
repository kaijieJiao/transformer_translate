import torch
import torch.nn as nn
from dataset import *
from transformer import Transformer
from config import *
from torch.utils.data import Dataset , DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
class De2EnDataset(Dataset):
    def __init__(self,dataset):
        super(De2EnDataset,self).__init__()
        self.enc_x,self.dec_x=[],[]
        for de_sentence,en_sentence in dataset:
            _,de_ids = de_preprocess(de_sentence)
            _,en_ids = en_preprocess(en_sentence)
            if len(en_ids)>SEQ_MAX_LEN or len(de_ids)>SEQ_MAX_LEN:
                continue
            self.enc_x.append(de_ids)
            self.dec_x.append(en_ids)
            assert len(self.enc_x)==len(self.dec_x)
    def __len__(self):
        return len(self.dec_x)
    def __getitem__(self,index):
        return self.enc_x[index],self.dec_x[index]
def collate_fn(batch):
    enc_x_batch=[]
    dec_x_batch=[]
    for enc_x,dec_x in batch:
        enc_x = torch.tensor(enc_x)
        dec_x = torch.tensor(dec_x)
        enc_x_batch.append(enc_x)
        dec_x_batch.append(dec_x)
    enc_x_batch = pad_sequence(enc_x_batch,True,PAD_ID)
    dec_x_batch = pad_sequence(dec_x_batch,True,PAD_ID)
    return enc_x_batch,dec_x_batch
def evaluate(model,dataloader,loss_fn,device):
    model.eval()
    total_loss = 0
    for batch_i,(enc_x,dec_x) in tqdm(enumerate(dataloader)):
        enc_x = enc_x.to(device)
        dec_y = dec_x[:,1:].to(device)
        dec_x = dec_x[:,:-1].to(device)
        dec_y_hat = model(enc_x,dec_x)
        loss = loss_fn(dec_y_hat.view(-1,dec_y_hat.size(-1)),dec_y.view(-1))
        total_loss += loss.item()
        # print(f"batch:{batch_i} loss:{loss.item()}")
    print(f"test_avg_loss:{total_loss/len(dataloader)}")
    return total_loss/len(dataloader)
if __name__ == '__main__':
    dataset_train = De2EnDataset(train_dataset)
    dataset_test = De2EnDataset(test_dataset)

    batch_size = 512
    dataloader_train = DataLoader(dataset_train,batch_size,shuffle=True,collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset_test,batch_size,shuffle=True,collate_fn=collate_fn)

    MODEL_PATH = './model_save/model_1.pth'
    device=DEVICE
    try:
        model=torch.load(MODEL_PATH).to(device)
    except:
        print('model not found')
        model = Transformer(enc_vocab_size=len(de_vocab),dec_vocab_size=len(en_vocab),emb_size=EMBEDDING_SIZE,q_k_size=128,v_size=128,f_size=FFN_SIZE,head=ATTENTION_HEAD,nblocks=BLOCK_NUM).to(device)
    loss_fn=nn.CrossEntropyLoss(ignore_index=PAD_ID)  #PAD词不参与损失计算
    optimizer = torch.optim.SGD(model.parameters(),lr=1e-3,momentum=0.9)
    epoch = 50
    model.train()
    min_loss = 0
    for i in range(epoch):
        loss_sum = 0
        for batch_i, (enc_x,dec_x) in tqdm(enumerate(dataloader_train)):
            enc_x = enc_x.to(device)
            real_dec_output=dec_x[:,1:].to(device)
            real_dec_input = dec_x[:,:-1].to(device)
            # print(enc_x.size(),real_dec_input.size())
            pre_output = model(enc_x,real_dec_input)
            loss=loss_fn(pre_output.view(-1,pre_output.size(-1)),real_dec_output.view(-1))
            loss_sum+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_i % 10 == 0:
                print("\nepoch:{},batch:{},loss:{}".format(i,batch_i,loss.item()))
        print("*"*70)
        print("epoch:{},train_avg_loss:{}".format(i,loss_sum/len(dataloader_train)))
        test_avg_loss = evaluate(model,dataloader_test,loss_fn,device)
        print("*"*70)

        if min_loss>test_avg_loss:
            min_loss=test_avg_loss
            torch.save(model,f"{MODEL_PATH}")
            


        

