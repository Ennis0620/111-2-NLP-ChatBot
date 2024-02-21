import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, GPT2Tokenizer

class ChatbotDataset(Dataset):
    def __init__(self, data, bert_tokenizer, gpt2_tokenizer, max_seq_length):
        self.data = data
        self.bert_tokenizer = bert_tokenizer
        self.gpt2_tokenizer = gpt2_tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        q, a = self.data[index]

        # 將Q和A轉換為BERT的輸入格式
        encoded_input = self.bert_tokenizer.encode_plus(q, a, add_special_tokens=True, max_length=self.max_seq_length, padding='max_length', truncation=True, return_tensors='pt')

        # 將GPT-2的輸入轉換為Tensor
        gpt2_input = self.gpt2_tokenizer.encode(a, add_special_tokens=True, max_length=self.max_seq_length, padding='max_length', truncation=True, return_tensors='pt').squeeze()

        return {
            'input_ids': encoded_input['input_ids'].squeeze(),
            'attention_mask': encoded_input['attention_mask'].squeeze(),
            'gpt2_input': gpt2_input
        }
