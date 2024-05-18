from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k,Multi30k
import torch
from config import SEQ_MAX_LEN
# 下载翻译数据集
multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
"""
train: 29000
valid: 1014
test: 1000
"""
train_dataset = list(Multi30k(split='train',language_pair=('de','en')))
test_dataset = list(Multi30k(split='valid',language_pair=('de','en')))
# test_dataset = list(Multi30k(split='test',language_pair=('de','en')))

# print(train_dataset)

# 创建分词器
de_tokenizer=get_tokenizer('spacy', language='de_core_news_sm')
en_tokenizer=get_tokenizer('spacy', language='en_core_web_sm')
UNK_TOKEN,PAD_TOKEN,SOS_TOKEN,EOS_TOKEN = '<UNK>','<PAD>','<SOS>','<EOS>'
UNK_ID,PAD_ID,SOS_ID,EOS_ID = 0,1,2,3
de_tokens=[]
en_tokens=[]
for de , en in train_dataset:
    de_tokens.append(de_tokenizer(de))
    en_tokens.append(en_tokenizer(en))
de_vocab = build_vocab_from_iterator(de_tokens,specials=[UNK_TOKEN,PAD_TOKEN,SOS_TOKEN,EOS_TOKEN],special_first=True)
de_vocab.set_default_index(UNK_ID)
en_vocab = build_vocab_from_iterator(en_tokens,specials=[UNK_TOKEN,PAD_TOKEN,SOS_TOKEN,EOS_TOKEN],special_first=True)
en_vocab.set_default_index(UNK_ID)
def de_preprocess(de_sentence):
    de_tokens = de_tokenizer(de_sentence)
    de_tokens = [SOS_TOKEN] + de_tokens + [EOS_TOKEN]
    de_ids=de_vocab(de_tokens)
    return de_tokens,de_ids
def en_preprocess(en_sentence):
    en_tokens = en_tokenizer(en_sentence)
    en_tokens = [SOS_TOKEN] + en_tokens + [EOS_TOKEN]
    en_ids = en_vocab(en_tokens)
    return en_tokens,en_ids

if __name__ == "__main__":
    print('de vocab size:',len(de_vocab))
    print('en_vocab:',len(en_vocab))
    print(en_preprocess("Hello"))
    print(de_preprocess("Hallo"))
