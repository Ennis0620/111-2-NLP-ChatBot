from transformers import BertTokenizerFast,BertModel
import torch
from torch.utils.data import Dataset,DataLoader
import os 
from transformers import BertConfig
import pickle

class ChatDataset(Dataset):
    def __init__(self, data, bert_tokenizer,max_len):
        CURRENT_PATH = os.path.dirname(__file__)
        data_word_index = f'{CURRENT_PATH}/data/QA_word_index_PTT_Gossiping_more.pkl'
        with open(data_word_index,'rb') as fp:
            self.word_index = pickle.load(fp)
            
        self.data = data
        self.bert_tokenizer = bert_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        question, answer = self.data[index]
        q_input = self.bert_tokenizer(question, add_special_tokens=False,return_tensors='pt',max_length=self.max_len,truncation=True,padding='max_length',)  # 问题（Q）不包含<SOS>和<END>
        a_input = [self.sentence_pad_input2(answer)]
        a_input = torch.tensor(a_input).to(torch.long)

        #print(a_input,len(a_input))
        target_ids = a_input[0][1:] # target_ids 是 a_input 去掉<SOS>
        padding_length = self.max_len - target_ids.size(0)
        target_ids = torch.cat([target_ids, torch.zeros(padding_length, dtype=torch.long)], dim=0)  # #捕到一樣長度
        target_ids = target_ids.unsqueeze(0)  # 添加 batch 维度

        return {'q_input': q_input, 'a_input': a_input, 'target':target_ids}
    
    def sentence_pad_input2(self, sentence):
        # 分割句子為單個字元
        chars = list(sentence)
        # 添加開始和結束標記
        chars = ['<sos>'] + chars + ['<end>']
        # 將句子補齊到指定的最大長度
        padded_chars = chars[:self.max_len] + ['<pad>'] * (self.max_len - len(chars))
        # 將字元轉換為索引
        indexed_chars = [self.word_index.get(char, self.word_index['<unk>']) for char in padded_chars]
        return indexed_chars


if __name__ == "__main__":
    CURRENT_PATH = os.path.dirname(__file__)
    # 載入繁體中文BERT模型作為Encoder
    bert_model_name = 'bert-base-chinese'
    bert_tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
    config = BertConfig.from_pretrained(bert_model_name)
    vocab_size = config.vocab_size
    print("Vocabulary Size:", vocab_size)

    data_path = f'H:/NCNU/class/111-2_code/3.NLP/111-2-NLP HW/final/data/PTT_Gossiping_more.txt'
    data = []
    with open(data_path,'r') as fp:
        all = fp.readlines()
        for per in all:
            per = per.strip('\n')
            per_split = per.split()
            Q = per_split[0]
            A = per_split[1]
            data.append((Q,A))

    max_len = 41
    dataset = ChatDataset(data,bert_tokenizer,max_len)
    dataloader = DataLoader(dataset,batch_size=1)
    d = next(iter(dataloader))
    ans = d
    print('In Q:',ans['q_input'])
    print('In A:',ans['a_input'])
    print('Target:',ans['target'])

