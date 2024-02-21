import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, GPT2Tokenizer
import os

CURRENT_PATH = os.path.dir(__file__)
data_path = f'{CURRENT_PATH}/data/PTT_Gossiping.txt'

# 載入繁體中文BERT和GPT-2的Tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('uer/gpt2-chinese-cluecorpussmall')

# 建立Dataset物件
dataset = ChatbotDataset(data, bert_tokenizer, gpt2_tokenizer, max_seq_length=128)

# 建立DataLoader物件
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 使用DataLoader進行訓練
for batch in dataloader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    gpt2_input = batch['gpt2_input'].to(device)

    # 在此進行訓練過程
    pass
