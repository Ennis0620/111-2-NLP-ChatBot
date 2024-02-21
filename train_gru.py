import torch
import torch.nn as nn
import os, shutil, tqdm
from model.model_gru import Seq2Seq
from dataset_gru import ChatDataset
from utils.early_stop import early_stop
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader



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

        for (batch_inputs, batch_targets), output_text in tqdm.tqdm(train_dataloader):
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            output_text = output_text.to(device)
            output_text = torch.unsqueeze(output_text, 2)
            # print('batch_inputs:',batch_inputs)
            # print('batch_targets:',batch_targets)
            # print('output_text:',output_text)
            # 前向傳播
            outputs = model(batch_inputs, batch_targets)
            outputs = outputs.view(-1,vocab_size)
            output_text = output_text.view(-1)
            
            # print(outputs.size())
            # print(batch_targets.size())
            # 計算損失
            loss = criterion(outputs, output_text)
            total_loss += loss.item()
            
            # 計算準確率
            acc = calculate_accuracy(outputs, output_text)
            total_acc += acc
            
            # 反向傳播和參數更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #validation_loss, validation_accuracy= valid(valid_dataloader,model,criterion)
        validation_loss = 0.0
        validation_accuracy = 0.0

        total_loss = total_loss / len(train_dataloader)
        total_acc = total_acc / len(train_dataloader)

        avg_loss = round(total_loss,6)
        accuracy = round(total_acc*100,4)
        validation_loss = round(validation_loss,6)
        validation_accuracy = round(validation_accuracy*100,4)

        
        print('Epoch: {} | train_loss: {} | train_acc: {}% | val_loss: {} | val_acc: {}%'\
              .format(epoch, avg_loss,accuracy,validation_loss,validation_accuracy))

        performance_value = [epoch,
                            avg_loss,
                            accuracy,
                            validation_loss,
                            validation_accuracy]

        train_loss.append(avg_loss)
        train_acc.append(accuracy)
        valid_loss.append(validation_loss)
        valid_acc.append(validation_accuracy)

        EARLY_STOP(avg_loss,
                    model=model,
                    performance_value = performance_value
                    )
            
        if EARLY_STOP.early_stop:
            print('Early stopping')
            break   
    
    return train_loss,valid_loss,train_acc,valid_acc

def valid(valid_dataloader,
          model,
          criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    validation_loss = 0
    validation_acc = 0.0
    with torch.no_grad():
        for (val_inputs, val_targets), val_output_text in tqdm.tqdm(valid_dataloader):
            val_inputs = val_inputs.to(device)
            val_targets = val_targets.to(device)
            val_output_text = val_output_text.to(device)
            
            # 前向傳播
            val_outputs = model(val_inputs, val_targets)
            val_outputs = val_outputs.view(-1,vocab_size)
            val_output_text = val_output_text.view(-1)
            
            # 計算損失
            validation_loss += criterion(val_outputs, val_output_text).item()
            
            # 計算準確率
            acc = calculate_accuracy(val_outputs, val_output_text)
            validation_acc += acc

    validation_loss /= len(valid_dataloader)
    validation_acc = validation_acc / len(valid_dataloader)

    
    return validation_loss, validation_acc


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

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CURRENT_PATH = os.path.dirname(__file__)
    SAVE_MODELS_PATH = f'{CURRENT_PATH}/weights/model_gru_200'
    WEIGHTS_NAME = 'model_weights.pth'
    WEIGHTS = f'{CURRENT_PATH}/weights/model_gru_200/{WEIGHTS_NAME}'
    # check = str(input(f'刪除:{SAVE_MODELS_PATH},(y or n):'))
    
    # if check == 'y':
    #     try:    
    #         shutil.rmtree(SAVE_MODELS_PATH)
    #     except:
    #         pass
    #     os.makedirs(SAVE_MODELS_PATH)

    data_path = f'{CURRENT_PATH}/data/PTT_Gossiping_more.txt'
    data = []
    valid_ratio = 0.0
    MAX_LEN = 41
    BATCH_SIZE = 512
    vocab_size = 5289   #訓練資料共有多少詞彙(字)
    hidden_size =  512
    embedding_dim = 200
    EPOCHS = 500

    # 資料準備
    with open(data_path,'r') as fp:
        all = fp.readlines()
        for per in all:
            per = per.strip('\n')
            per_split = per.split()
            Q = per_split[0]
            A = per_split[1]
            data.append((Q,A))
    #打亂
    data = np.array(data)
    np.random.seed(6670)
    indices = list(range(len(data)))
    t_split = int(np.floor(valid_ratio * len(data)))
    
    train_indices,valid_indices = indices[t_split:],indices[:t_split]
    train_d = data[train_indices]
    valid_d = data[valid_indices]

    train_data = ChatDataset(train_d,mode='train',max_len=MAX_LEN, vocab_size=vocab_size)
    valid_data = ChatDataset(valid_d,mode='valid',max_len=MAX_LEN, vocab_size=vocab_size)

    train_dataloader = DataLoader(train_data,batch_size=BATCH_SIZE)
    valid_dataloader = DataLoader(valid_data,batch_size=BATCH_SIZE)

    model = Seq2Seq(Embedding_input_vocab_size=vocab_size, 
                   Embedding_output_vocab_size=vocab_size,
                   GRU_hidden_size = hidden_size, 
                   embedding_dim = embedding_dim,).to(device)
    model.load_state_dict(torch.load(WEIGHTS))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)

    EARLY_STOP = early_stop(save_path=SAVE_MODELS_PATH,
                        mode='min',
                        monitor='train_loss',
                        patience=500)
    
    train_loss,valid_loss,train_acc,valid_acc = train(num_epochs=EPOCHS,
          train_dataloader=train_dataloader,
          valid_dataloader=valid_dataloader,
          model=model,
          criterion=criterion,)

    
    plot_statistics(train_loss=train_loss,valid_loss=valid_loss,train_performance=train_acc,valid_performance=valid_acc,performance_name='acc',SAVE_MODELS_PATH=SAVE_MODELS_PATH)


