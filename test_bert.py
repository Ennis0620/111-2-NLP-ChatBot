import torch
import torch.nn as nn
import os, shutil
from transformers import BertTokenizerFast, AutoModel, BertModel
from dataset_bert import ChatDataset
from torch.utils.data import Dataset, DataLoader
from model.model_bert import Seq2Seq
import numpy as np
import tqdm
from utils.early_stop import early_stop
import matplotlib.pyplot as plt
from transformers import BertConfig
from torchsummaryX import summary
import random


def preprocess_input(input_text, bert_tokenizer, max_len):
    encoded_inputs = bert_tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=max_len,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    input_ids = encoded_inputs['input_ids']
    return input_ids.to(device)


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




if __name__ =="__main__":


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    CURRENT_PATH = os.path.dirname(__file__)
    WEIGHTS_NAME = 'model_weights.pth'
    WEIGHTS = f'{CURRENT_PATH}/weights/model_bert_LSTM_emd100/{WEIGHTS_NAME}'
    print('load weight:',WEIGHTS)

    MAX_LEN = 41
    
    # 載入繁體中文BERT模型作為Encoder
    bert_model_name = 'bert-base-chinese'
    bert_tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
    bert_model = BertModel.from_pretrained(bert_model_name) 
    
    # 初始化模型
    config = BertConfig.from_pretrained(bert_model_name)
    vocab_size = bert_tokenizer.vocab_size
    hidden_size = 768  # BERT模型的隐藏層大小

    model = Seq2Seq(
        bert_model=bert_model,
        output_size=vocab_size,
        embedding_dim=100
    ).to(device)

    # x = torch.LongTensor([[0]*MAX_LEN]).to(device)
    # x = x.to(torch.int)
    # print(summary(model,x,x))

    model.load_state_dict(torch.load(WEIGHTS))
    model.eval()

    input_text = str(input('請輸入對話:'))
    # 回應
    predicted_text = predict(input_text, model, bert_tokenizer, MAX_LEN)
    predicted_text2 = predict_2(input_text, model, bert_tokenizer, MAX_LEN)
    
    print("回答1:", predicted_text)
    print("回答2:", predicted_text2)