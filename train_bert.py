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

def train(num_epochs,
          train_dataloader,
          valid_dataloader,
          model,
          criterion,
          ):
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_acc = 0.0
        model.train()
        for batch  in tqdm.tqdm(train_dataloader):

            input1 = batch['q_input'].squeeze(1).to(device)
            input2 = batch['a_input'].squeeze(1).to(device)
            target = batch['target'].squeeze(1).to(device)

            logits = model(input1, input2)
            # print('logits:',torch.argmax(logits,dim=2))
            # print('target:',target)

            logits = logits.view(-1, logits.shape[-1])
            target = target.view(-1)
            loss = criterion(logits, target)
            
            # 計算損失
            total_loss += loss.item()
            
            acc = calculate_accuracy(logits, target)
            total_acc += acc
            
            
            # 反向傳播和參數更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation_loss, validation_accuracy= valid(valid_dataloader,model,criterion)
        validation_loss = 0
        validation_accuracy = 0
        average_loss = total_loss / len(train_dataloader)
        accuracy = total_acc / len(train_dataloader)

        avg_loss = round(average_loss,6)
        accuracy = round(accuracy*100,4)
        # validation_loss = round(validation_loss,6)
        # validation_accuracy = round(validation_accuracy*100,4)

        
        print('Epoch: {} | train_loss: {} | train_acc: {}% | val_loss: {} | val_acc: {}%'\
              .format(epoch, avg_loss,accuracy,validation_loss,validation_accuracy))
        
        performance_value = [epoch,
                            avg_loss,
                            accuracy,
                            validation_loss,
                            validation_accuracy]

        # train_loss.append(avg_loss)
        # train_acc.append(accuracy)
        # valid_loss.append(validation_loss)
        # valid_acc.append(validation_accuracy)

        EARLY_STOP(avg_loss,
                    model=model,
                    performance_value = performance_value
                    )
            
        if EARLY_STOP.early_stop:
            print('Early stopping')
            break   

def valid(valid_dataloader,
          model,
          criterion):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    with torch.no_grad():
        for batch in tqdm.tqdm(valid_dataloader):
            input1 = batch['q_input'].squeeze(1).to(device)
            input2 = batch['a_input'].squeeze(1).to(device)
            target = batch['target'].squeeze(1).to(device)


            logits = model(input1, input2)

            logits = logits.view(-1, logits.shape[-1])
            target = target.view(-1)

            
            loss = criterion(logits, target)
            # 計算損失
            total_loss += loss.item()
            
            
            acc = calculate_accuracy(logits, target)
            total_acc += acc
            

    average_loss = total_loss / len(valid_dataloader)
    avg_acc = total_acc / len(valid_dataloader)

    return average_loss,avg_acc

def plot_statistics(train_loss,
                    valid_loss,
                    train_performance,
                    valid_performance,
                    performance_name,
                    SAVE_MODELS_PATH):
    '''
    統計train、valid的loss、performance
    '''
    
    t_loss = plt.plot(train_loss)
    v_loss = plt.plot(valid_loss)
    
    plt.legend([t_loss,v_loss],
               labels=['train_loss',
                        'valid_loss'])

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(f'{SAVE_MODELS_PATH}/train_loss',bbox_inches='tight')
    plt.figure()

    t_per = plt.plot(train_performance)
    v_per = plt.plot(valid_performance)

    plt.legend([t_per,v_per],
               labels=[f'train_{performance_name}',
                       f'valid_{performance_name}'])

    plt.xlabel("epoch")
    plt.ylabel(f"{performance_name}")
    plt.savefig(f'{SAVE_MODELS_PATH}/train_performance',bbox_inches='tight')
    plt.figure()

def calculate_accuracy(logits, targets):
    _, predicted = torch.max(logits, dim=1)
    # print(predicted)
    # 将0的位置设为False
    mask = targets != 0
    
    # 计算非0位置上的预测准确率
    correct = torch.sum(predicted[mask] == targets[mask]).item()
    total = torch.sum(mask).item()
    
    accuracy = correct / total
    # print(accuracy)

    return accuracy


def predict(input_ids, model, bert_tokenizer, max_len):
    model.eval()

    with torch.no_grad():
        predicted_ids = torch.zeros_like(input_ids)  # 初始化predicted_ids全0
        predicted_ids[:, 0] = bert_tokenizer.cls_token_id  # 第一個位置是<SOS>標記的index

        print('input_ids:',input_ids)
        print('predicted_ids:',predicted_ids)

        for i in range(1, max_len):
            logits = model(input_ids, predicted_ids)
            _, predicted = torch.max(logits[:, i-1, :], dim=1)
            # print(predicted)
            predicted_ids[0, i] = predicted.item()
            
            # print(predicted_ids)

            if predicted.item() == bert_tokenizer.eos_token_id or predicted.item() == bert_tokenizer.pad_token_id:
                break

    predicted_ids = predicted_ids.squeeze(0).cpu().numpy().tolist()
    predicted_text = bert_tokenizer.decode(predicted_ids, skip_special_tokens=True)
    return predicted_text


if __name__ =='__main__':


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CURRENT_PATH = os.path.dirname(__file__)
    BATCH_SIZE = 512
    WEIGHTS_NAME = 'model_weights.pth'
    WEIGHTS = f'{CURRENT_PATH}/weights/model_bert_LSTM_emd100/{WEIGHTS_NAME}'


    SAVE_MODELS_PATH = f'{CURRENT_PATH}/weights/model_bert_LSTM_emd100/'
    # check = str(input(f'刪除:{SAVE_MODELS_PATH},(y or n):'))
    # if check == 'y':
    #     try:    
    #         shutil.rmtree(SAVE_MODELS_PATH)
    #     except:
    #         pass
    #     os.makedirs(SAVE_MODELS_PATH)

    EPOCHS = 5000
    MAX_LEN = 41
    # 載入繁體中文BERT模型作為Encoder
    bert_model_name = 'bert-base-chinese'
    bert_tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
    bert_model = BertModel.from_pretrained(bert_model_name).to(device)
    
    
    # 資料準備
    data_path = f'{CURRENT_PATH}/data/PTT_Gossiping.txt'
    data = []
    with open(data_path,'r') as fp:
        all = fp.readlines()
        for per in all:
            per = per.strip('\n')
            per_split = per.split()
            Q = per_split[0]
            A = per_split[1]
            data.append((Q,A))

    #打亂
    valid_ratio = 0.0
    data = np.array(data)
    np.random.seed(6670)
    indices = list(range(len(data)))
    t_split = int(np.floor(valid_ratio * len(data)))
    
    train_indices,valid_indices = indices[t_split:],indices[:t_split]
    train_d = data[train_indices]
    valid_d = data[valid_indices]

    train_data = ChatDataset(train_d,bert_tokenizer,MAX_LEN)
    valid_data = ChatDataset(valid_d,bert_tokenizer,MAX_LEN)

    train_dataloader = DataLoader(train_data,batch_size=BATCH_SIZE)
    valid_dataloader = DataLoader(valid_data,batch_size=BATCH_SIZE)

    # 初始化模型
    config = BertConfig.from_pretrained(bert_model_name)
    vocab_size = bert_tokenizer.vocab_size# 使用BERT tokenizer的詞彙量最為生成的類別數量
    hidden_size = 768  # BERT模型的隐藏層大小
    
    model = Seq2Seq(
        bert_model=bert_model,
        output_size=vocab_size,
        embedding_dim=100,
    ).to(device)
    model.load_state_dict(torch.load(WEIGHTS))
    # 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    EARLY_STOP = early_stop(save_path=SAVE_MODELS_PATH,
                        mode='min',
                        monitor='train_loss',
                        patience=500)
    

    train(num_epochs=EPOCHS,
          train_dataloader=train_dataloader,
          valid_dataloader=valid_dataloader,
          model=model,
          criterion=criterion,)

    
    # plot_statistics(train_loss=train_loss,valid_loss=valid_loss,train_performance=train_acc,valid_performance=valid_acc,performance_name='acc',SAVE_MODELS_PATH=SAVE_MODELS_PATH)



    
