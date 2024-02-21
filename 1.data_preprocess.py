import os
from ckip_transformers.nlp import CkipWordSegmenter
import numpy as np
import torch
import pickle

CURRENT_PATH = os.path.dirname(__file__)
ws_driver = CkipWordSegmenter(model="albert-base", device=0)

data_path = f'{CURRENT_PATH}/data/PTT_Gossiping_more.txt'

data_word_index = f'{CURRENT_PATH}/data/QA_word_index_PTT_Gossiping_more.pkl'
data_index_word = f'{CURRENT_PATH}/data/QA_index_word_PTT_Gossiping_more.pkl'

data = []
data_Q = []
data_A = []

Q_len = 0
A_len = 0


# 資料準備
with open(data_path,'r') as fp:
    all = fp.readlines()
    for per in all:
        per = per.strip('\n')
        per_split = per.split()
        Q = per_split[0]
        A = per_split[1]
       
        data_Q.append(Q)
        data_A.append(A)
        data.append(Q)
        data.append(A)

        Q_len += len(Q)
        A_len += len(A)

# print('Q mean:', Q_len/len(data_Q)) 
# print('A mean:', A_len/len(data_A))

# 將文本轉換為數字序列
input_texts = data_Q
target_texts = data_A
input_characters = sorted(list(set(''.join(input_texts))))
target_characters = sorted(list(set(''.join(target_texts))))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

data_texts = data
data_characters = sorted(list(set(''.join(data_texts))))
max_data_seq_length = max([len(txt) for txt in data_texts])
data_token_index = {'<pad>': 0}  # 將 <pad> 標記的索引設為 0
# 建立字典對照表
for i, char in enumerate(data_characters):
    data_token_index[char] = i + 1  
data_token_index['<sos>'] = len(data_token_index) 
data_token_index['<unk>'] = len(data_token_index)
#data_token_index = dict([(char, i) for i, char in enumerate(data_characters)])


data_texts_inverse = data
data_characters_inverse = sorted(list(set(''.join(data_texts_inverse))))
data_token_index_inverse = { 0 : '<pad>'}  # 將 <pad> 標記的索引設為 0
# 建立字典對照表
for i, char in enumerate(data_characters_inverse):
    data_token_index_inverse[i + 1] = char  
data_token_index_inverse[len(data_token_index_inverse) ] ='<sos>'
data_token_index_inverse[len(data_token_index_inverse)] = '<unk>'
print(data_token_index_inverse)
print('max_data_seq_length:',max_data_seq_length)
# print(num_encoder_tokens)
# print(input_characters)
# print(input_token_index)
# print(num_decoder_tokens)
# print(target_characters)
# print(target_token_index)


# print(data_characters)
# print(data_token_index)
# print('max len:', max_data_seq_length)
# print('max Q len:', max_encoder_seq_length)
# print('max A len:', max_decoder_seq_length)

# print('Q len:',len(data_Q))
# print('A len:',len(data_A))
# with open(f"{data_Q_word_index}", "wb") as tf:
#     pickle.dump(input_token_index,tf)

# with open(f"{data_A_word_index}", "wb") as tf:
#     pickle.dump(target_token_index,tf)

with open(f"{data_word_index}", "wb") as fp:
    pickle.dump(data_token_index, fp)

with open(f"{data_index_word}", "wb") as fp:
    pickle.dump(data_token_index_inverse, fp)