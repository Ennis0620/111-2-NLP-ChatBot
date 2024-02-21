import torch
import torch.nn as nn
import os, shutil
from transformers import BertTokenizerFast, AutoModel, BertModel
from dataset_bert_4_v2 import ChatDataset
from torch.utils.data import Dataset, DataLoader
from model.model_bert_4_LSTM_v2 import Seq2Seq
import numpy as np
import tqdm
from utils.early_stop import early_stop
import matplotlib.pyplot as plt
from transformers import BertConfig
from torchsummaryX import summary
import random
import pickle

def preprocess_input(input_text, bert_tokenizer, max_len):
    encoded_inputs = bert_tokenizer.encode_plus(
        input_text,
        add_special_tokens=False,
        max_length=max_len,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    input_ids = encoded_inputs['input_ids']
    return input_ids

def test(model, bert_tokenizer,input_text,index_word,word_index,max_len):

    def sentence_pad_input1(sentence):
        # 分割句子為單個字元
        chars = list(sentence)
        # 添加開始和結束標記
        chars = chars
        # 將句子補齊到指定的最大長度
        padded_chars = chars[:max_len] + ['<pad>'] * (max_len - len(chars))
        # 將字元轉換為索引
        indexed_chars = [word_index.get(char, word_index['<unk>']) for char in padded_chars]
        return indexed_chars
    
    input_text = preprocess_input(input_text, bert_tokenizer, max_len).to(device)
    target_text = [0] * max_len
    target_text[0] = word_index['<sos>']
    input_text = torch.tensor(input_text).to(torch.long)
    target_text = torch.tensor(target_text).to(torch.long).unsqueeze(0)
    print(input_text.size())
    print(target_text.size())

    with torch.no_grad():
        inputs1 = input_text.to(device)
        inputs2 = target_text.to(device)
        origin_input2 = inputs2.to(device)
        #要把inputs2持續餵給model才能講出完整的話
        for num_word in range(MAX_LEN):
            outputs = model(inputs1,inputs2)
            outputs_argmax = torch.argmax(outputs,dim=-1)
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
            
    generate_ans = convert_word(index_word, generate_ans)
    return generate_ans

def test2(model, bert_tokenizer,input_text,index_word,word_index,max_len):
    # print('model',model)
    # print('bert_tokenizer',bert_tokenizer)
    # print('input_text',input_text)
    # print('index_word',index_word)
    # print('word_index',word_index)
    # print('max_len',max_len)

    def sentence_pad_input1(sentence):
        # 分割句子為單個字元
        chars = list(sentence)
        # 添加開始和結束標記
        chars = chars
        # 將句子補齊到指定的最大長度
        padded_chars = chars[:max_len] + ['<pad>'] * (max_len - len(chars))
        # 將字元轉換為索引
        indexed_chars = [word_index.get(char, word_index['<unk>']) for char in padded_chars]
        return indexed_chars
    print('preprocess ... ')
    input_text = preprocess_input(input_text, bert_tokenizer, max_len).to(device)
    target_text = [0] * max_len
    target_text[0] = word_index['<sos>']
    input_text = torch.tensor(input_text).to(torch.long)
    target_text = torch.tensor(target_text).to(torch.long).unsqueeze(0)
    print('inference ... ')
    with torch.no_grad():
        inputs1 = input_text.to(device)
        inputs2 = target_text.to(device)
        origin_input2 = inputs2.to(device)
        #要把inputs2持續餵給model才能講出完整的話
        for num_word in range(MAX_LEN):
            outputs = model(inputs1,inputs2)

            if num_word == 0:
                top_5 = torch.topk(outputs[:, num_word-1, :].flatten(), 5).indices
                # print(top_5)
                random_select = np.random.randint(0,5)
                random_select_top5 = top_5[random_select]
                # print('top5:',top_5[random_select])
                outputs_index = random_select_top5.detach().cpu().numpy()
                # print('top5 outputs_index:',outputs_index)
            else:
                outputs_argmax = torch.argmax(outputs,dim=-1)
                outputs_index = outputs_argmax[0][num_word].detach().cpu().numpy()
                # print('outputs_index:',outputs_index)

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
            
    generate_ans = convert_word(index_word, generate_ans)

    return generate_ans

def predict(input_text, model, bert_tokenizer, max_len):
    model.eval()

    with torch.no_grad():
        input_ids = preprocess_input(input_text, bert_tokenizer, max_len)
        predicted_ids = torch.zeros_like(input_ids)  # 初始化predicted_ids全0
        predicted_ids[:, 0] = bert_tokenizer.cls_token_id  # 第一個位置是<SOS>標記的index

        # print('input_ids:',input_ids)
        # print('predicted_ids:',predicted_ids)

        for i in range(1, max_len):
            logits = model(input_ids, predicted_ids)
            _, predicted = torch.max(logits[:, i-1, :], dim=1)
            print('predicted:',predicted)
            predicted_ids[0, i] = predicted.item()
            
            #print(predicted_ids)

            if predicted.item() == bert_tokenizer.eos_token_id or predicted.item() == bert_tokenizer.pad_token_id:
                print(predicted_ids)
                break

    predicted_ids = predicted_ids.squeeze(0).cpu().numpy().tolist()
    predicted_text = bert_tokenizer.decode(predicted_ids, skip_special_tokens=True)
    return predicted_text

def predict_2(input_text, model, bert_tokenizer, max_len):
    model.eval()

    with torch.no_grad():
        input_ids = preprocess_input(input_text, bert_tokenizer, max_len)
        predicted_ids = torch.zeros_like(input_ids)  # 初始化predicted_ids全0
        predicted_ids[:, 0] = bert_tokenizer.cls_token_id  # 第一個位置是<SOS>標記的index

        # print('input_ids:',input_ids)
        # print('predicted_ids:',predicted_ids)

        for i in range(1, max_len):
            logits = model(input_ids, predicted_ids)

            logits_softmax = torch.softmax(logits[:, i-1, :], dim=1)#變成機率
            # top_1 = torch.max(logits_softmax, dim=1)#values#indices #取機率最高那個
            # top_1_index = top_1.indices
            # top_1_value = top_1.values
            # print("softmax index :", top_1_index)
            # print('max index:',predicted)

            _, predicted = torch.max(logits[:, i-1, :], dim=1)

            #開頭一開始會random選5個
            if i == 1:
                top_5 = torch.topk(logits[:, i-1, :].flatten(), 5).indices
                random_select = np.random.randint(0,5)
                random_select_top5 = top_5[random_select]
                print('top5:',top_5[random_select])
                predicted_ids[0, i] = random_select_top5.item()
            
            #如果top1的分數高於0.1 代表很有信心 就直接選, 否則就隨機top3
            else:
                predicted_ids[0, i] = predicted.item()
                # top_1 = torch.max(logits_softmax, dim=1)#values#indices #取機率最高那個
                # top_1_value = top_1.values
                # if top_1_value >= 0.1:
                #     print(top_1_value)
                #     predicted_ids[0, i] = predicted.item()
                # else:
                #     top_3 = torch.topk(logits_softmax.flatten(), 3)
                #     top_3_index = top_3.indices
                #     top_3_value = top_3.values
                #     print('top_3_value',top_3_value)
                #     random_select = np.random.randint(0,3)
                #     random_select_top3_index = top_3_index[random_select]
                #     random_select_top3_value = top_3_value[random_select]
                #     print('random_select_top3_value',random_select_top3_value)

                    # predicted_ids[0, i] = random_select_top3_value.item()
                    

            if predicted.item() == bert_tokenizer.eos_token_id or predicted.item() == bert_tokenizer.pad_token_id:
                print(predicted_ids)
                break

    predicted_ids = predicted_ids.squeeze(0).cpu().numpy().tolist()
    predicted_text = bert_tokenizer.decode(predicted_ids, skip_special_tokens=True)
    return predicted_text

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

if __name__ =="__main__":


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    CURRENT_PATH = os.path.dirname(__file__)
    WEIGHTS_NAME = 'model_weights.pth'
    WEIGHTS = f'{CURRENT_PATH}/weights/model_bert_4_LSTM_emd200/{WEIGHTS_NAME}'
    print('load weight:',WEIGHTS)

    MAX_LEN = 41
    
    # 載入繁體中文BERT模型作為Encoder
    bert_model_name = 'bert-base-chinese'
    bert_tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
    bert_model = BertModel.from_pretrained(bert_model_name) 
    
    vocab_size = 5289
    #index2word
    data_index_word = f'{CURRENT_PATH}/data/QA_index_word_PTT_Gossiping_more.pkl'
    with open(data_index_word,'rb') as fp:
        index_word = pickle.load(fp)
    data_word_index = f'{CURRENT_PATH}/data/QA_word_index_PTT_Gossiping_more.pkl'
    with open(data_word_index,'rb') as fp:
        word_index = pickle.load(fp)

    model = Seq2Seq(
        bert_model=bert_model,
        output_size=vocab_size,
        embedding_dim=200
    ).to(device)

    # x = torch.LongTensor([[0]*MAX_LEN]).to(device)
    # x = x.to(torch.int)
    # print(summary(model,x,x))

    model.load_state_dict(torch.load(WEIGHTS))
    model.eval()
    while 1 :
        input_text = str(input('請輸入對話:'))
        predicted_text = test2(model=model,
                   bert_tokenizer=bert_tokenizer,
                   input_text=input_text,
                   index_word=index_word,
                   word_index=word_index,
                   max_len=MAX_LEN)
        print("回答:", predicted_text)
        # 回應
        # predicted_text = predict(input_text, model, bert_tokenizer, MAX_LEN)
        # predicted_text2 = predict_2(input_text, model, bert_tokenizer, MAX_LEN)
        
        # print("回答1:", predicted_text)
        # print("回答2:", predicted_text2)