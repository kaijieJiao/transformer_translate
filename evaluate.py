import torch
import torch.nn as nn
from config import *
from dataset import *
from transformer import Transformer
def translate(model,enc_sentence,device):
    enc_tokens,enc_ids = de_preprocess(enc_sentence)
    if len(enc_tokens)>SEQ_MAX_LEN:
        enc_ids=enc_ids[:,:SEQ_MAX_LEN]
        enc_tokens=enc_tokens[:SEQ_MAX_LEN]
    enc_x = torch.tensor(enc_ids).unsqueeze(0).to(device)
    encoder_z= model.encoder(enc_x)
    dec_x = []
    dec_x.append(SOS_ID)
    MAX_TOKEN_NUM=100
    while len(dec_x) < MAX_TOKEN_NUM:
        dec_x_batch = torch.tensor(dec_x).unsqueeze(0).to(device)
        dec_y = model.decoder(dec_x_batch,encoder_z,enc_x)
        next_token_probs = dec_y[0,dec_y.size(1)-1,:]
        next_token_id = torch.argmax(next_token_probs,-1)
        if next_token_id == EOS_ID:
            break
        else:
            dec_x.append(next_token_id.item())
    dec_x_ids = [id for id in dec_x if id not  in [UNK_ID,SOS_ID,PAD_ID,EOS_ID] ]
    dec_tokens = en_vocab.lookup_tokens(dec_x_ids) 
    return ' '.join(dec_tokens)


if __name__ == '__main__':
    random_num = torch.randint(0,len(train_dataset),(1,)).item()
    print("原文：",train_dataset[random_num])
    model = torch.load("model_save/model.pth").to(DEVICE)
    model.eval()
    enc_sentence = input("请输入要翻译的德语：")
    while enc_sentence != 'q':
        dec_sentence = translate(model,enc_sentence,DEVICE)
        print(dec_sentence)
        print('*'*100)
        random_num = torch.randint(0,len(train_dataset),(1,)).item()
        print("原文：",train_dataset[random_num])

        enc_sentence = input("请继续输入要翻译的德语（q 退出）：")

