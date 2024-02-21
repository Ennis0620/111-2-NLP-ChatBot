import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import pickle

class ChatDataset(Dataset):
    def __init__(self, data, mode ,max_len, vocab_size):
        CURRENT_PATH = os.path.dirname(__file__)
        data_word_index = f'{CURRENT_PATH}/data/QA_word_index_PTT_Gossiping_more.pkl'
        with open(data_word_index,'rb') as fp:
            self.word_index = pickle.load(fp)
        # print(self.word_index)
        self.data = data
        self.max_len = max_len
        self.mode = mode
        self.vocab_size = vocab_size
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode =='test':
            input_text = self.data[index]
            input_text = self.sentence_pad_input1(input_text)
            # print(input_text)
            #test要給<sos>,0,0,0,...0
            target_text = [0] * self.max_len
            target_text[0] = self.word_index['<sos>']
            # print(target_text)
            input_text = torch.tensor(input_text).to(torch.long)
            target_text = torch.tensor(target_text).to(torch.long)

            return input_text, target_text
        else:
            input_text, target_text = self.data[index]
            input_text = self.sentence_pad_input1(input_text)
            target_text = self.sentence_pad_input2(target_text)
            
            '''
            實際上的decoder輸出要是A的往左shift
            ex:
            A = '<sos>,我,捐,最,頂,的,<end>,0,0,0' (實際上會是index)
            out = '我,捐,最,頂,的,<end>,0,0,0,0'
            '''
            output_answer = target_text
            output_answer = output_answer[1:]   #刪掉第0個index
            output_answer.append(0)             #後面補0

            input_text = torch.tensor(input_text).to(torch.long)
            target_text = torch.tensor(target_text).to(torch.long)
            output_answer = torch.tensor(output_answer).to(torch.long)
            # print('output_answer:',output_answer)
            #實際答案要轉one-hot
            #output_answer = self.one_hot(output_answer)
            # print(input_text.size())
            # print(target_text.size())
            return (input_text,target_text),output_answer 
    
    def sentence_pad_input1(self, sentence):
        # 分割句子為單個字元
        chars = list(sentence)
        # 添加開始和結束標記
        chars = chars
        # 將句子補齊到指定的最大長度
        padded_chars = chars[:self.max_len] + ['<pad>'] * (self.max_len - len(chars))
        # 將字元轉換為索引
        indexed_chars = [self.word_index.get(char, self.word_index['<unk>']) for char in padded_chars]
        return indexed_chars
    
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


    def one_hot(self,target_text):
        one_hot = torch.zeros(target_text.size(0), self.vocab_size)
        labels = torch.unsqueeze(target_text,1).to(torch.int64)#此行在訓練時要加上才能練,但要summary則需要拿掉
        one_hot.scatter_(1, labels, 1)
        return one_hot
    

if __name__ == "__main__":
    CURRENT_PATH = os.path.dirname(__file__)
    data_path = f'{CURRENT_PATH}/data/PTT_Gossiping_more.txt'
    data = []
    with open(data_path,'r') as fp:
        all = fp.readlines()
        for per in all:
            per = per.strip('\n')
            per_split = per.split()
            Q = per_split[0]
            A = per_split[1]
            data.append((Q,A))

    max_len = 38
    vocab_size = 4771
    dataset = ChatDataset(data=data,
                          max_len=max_len,
                          mode='train',
                          vocab_size = vocab_size)
    dataloader = DataLoader(dataset,batch_size=1)
    d = next(iter(dataloader))
    (in1,in2),ans = d
    print('In:',in1)
    print('In2:',in2)
    print('In3:',ans)

    # dataset = ChatDataset(data=data,
    #                       max_len=max_len,
    #                       mode='test',
    #                       vocab_size = vocab_size)
    # dataloader = DataLoader(dataset,batch_size=10)
    # d = next(iter(dataloader))
    # in1,in2 = d
    # print('In:',in1)
    # print('In2:',in2)
   