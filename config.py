import torch
EMBEDDING_SIZE = 512
ATTENTION_HEAD = 8
BLOCK_NUM = 12
FFN_SIZE = 4
SEQ_MAX_LEN = 512
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
