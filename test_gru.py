import torch
import torch.nn as nn
import os, shutil, tqdm
from model.model_gru import Seq2Seq
from dataset_gru import ChatDataset
from utils.early_stop import early_stop
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle


def test(model,dataloader,index_word,word_index):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generate_ans = ''
    with torch.no_grad():
        for inputs1, inputs2 in tqdm.tqdm(dataloader):
            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)
            origin_input2 = inputs2.to(device)
            
            #要把inputs2持續餵給model才能講出完整的話
            for num_word in range(MAX_LEN):
                print(inputs2)
                outputs = model(inputs1,inputs2)
                #print('outputs:',outputs.size())
                outputs_argmax = torch.argmax(outputs,dim=-1)
                #print('outputs_argmax:',outputs_argmax.size())
                #print('outputs_argmax[0][num_word]:',outputs_argmax[0][num_word])
                outputs_index = outputs_argmax[0][num_word].detach().cpu().numpy()
                #如果檢查到<end>、<unk>、0 代表結束，要回傳答案
                if check_end(outputs_index) or num_word== MAX_LEN-1:
                    generate_ans = origin_input2[0].detach().cpu().numpy()
                    generate_ans = generate_ans[1:]
                    generate_ans = np.append(generate_ans,int(outputs_index))
                    break
                else:
                    #繼續預測  要把這個time step預測出來的字 加到input上
                    origin_input2[0][num_word+1] = torch.tensor(outputs_index)                
                    inputs2 = origin_input2
            
            break
    print(generate_ans)
    generate_ans = convert_word(index_word, generate_ans)
    return generate_ans

def check_end(index):
    if index_word[int(index)]=='<end>' or index_word[int(index)]=='<unk>' or int(index) == 0:
        return True


def convert_word(index_word, string):
    string = list(string)
    return_string = ''
    for index in string:
        if index_word[index] == '<end>':
            return return_string
        elif index_word[index] == '<unk>':
            return return_string
        elif index == 0:
            return return_string
        else:
            return_string += index_word[index]
        
    return return_string
            



if __name__ == "__main__":


    input = input("請輸入文字:")

    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CURRENT_PATH = os.path.dirname(__file__)
    WEIGHTS_NAME = 'model_weights.pth'
    WEIGHTS = f'{CURRENT_PATH}/weights/model_gru_200/{WEIGHTS_NAME}'
    
    MAX_LEN = 41
    BATCH_SIZE = 100
    vocab_size = 5289   #訓練資料共有多少詞彙(字)
    hidden_size =  512
    embedding_dim = 200
    
    data_index_word = f'{CURRENT_PATH}/data/QA_index_word_PTT_Gossiping_more.pkl'
    with open(data_index_word,'rb') as fp:
        index_word = pickle.load(fp)

    data_word_index = f'{CURRENT_PATH}/data/QA_word_index_PTT_Gossiping_more.pkl'
    with open(data_word_index,'rb') as fp:
        word_index = pickle.load(fp)
    
    input = ChatDataset(input ,mode='test',max_len=MAX_LEN, vocab_size=vocab_size)
    
    dataloader = DataLoader(input,batch_size=1)

    model = Seq2Seq(Embedding_input_vocab_size=vocab_size, 
                   Embedding_output_vocab_size=vocab_size,
                   GRU_hidden_size = hidden_size, 
                   embedding_dim = embedding_dim,).to(device)
    model.load_state_dict(torch.load(WEIGHTS))
    model.eval()

    generate_ans = test(model,
                        dataloader,
                        index_word,
                        word_index)
    print(generate_ans)

