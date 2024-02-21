import torch
import torch.nn as nn
#from torchinfo import summary
from torchsummaryX import summary
from transformers import BertTokenizerFast, BertModel
from transformers import BertConfig
import os
# from dataset_bert_4 import ChatDataset
# from torch.utils.data import Dataset,DataLoader

class Seq2Seq(nn.Module):
    def __init__(self, bert_model, output_size, embedding_dim):
        super(Seq2Seq, self).__init__()
        self.encoder = bert_model
        self.encoder.requires_grad_(False)  # 設定BERT模型的requires_grad為False
        self.encoder.eval()
        self.encoder_LSTM = nn.LSTM(768 ,512 ,batch_first=True)

        self.decoder_embedding = nn.Embedding(output_size, embedding_dim)
        self.decoder = nn.LSTM(embedding_dim, 512, batch_first=True)
        self.linear = nn.Linear(512, output_size)

    def forward(self, input1, input2):
        input1_input_ids = input1[0]
        input1_token_type_ids = input1[1]
        input1_attention_mask = input1[2]
        
        #print('input1_input_ids',input1_input_ids.size())
        #print('input1_token_type_ids',input1_token_type_ids.size())
        #print('input1_attention_mask',input1_attention_mask.size())
        encoder_hidden_state = self.encoder(input_ids=input1_input_ids,
                                            token_type_ids=input1_token_type_ids,
                                            attention_mask = input1_attention_mask).last_hidden_state  # 用bert的最後的hidden state shape:[batch_size, seq_len, hidden_size]
        #print('encoder_hidden_state:',encoder_hidden_state.size())
        encoder_LSTM_out,(encoder_LSTM_hidden,encoder_LSTM_cell) = self.encoder_LSTM(encoder_hidden_state)
        #print('encoder_LSTM_hidden:',encoder_LSTM_hidden.size())
        #print('encoder_LSTM_cell:',encoder_LSTM_cell.size())

        decoder_embedded = self.decoder_embedding(input2)
        #print('decoder_embedded:',decoder_embedded.size())

        decoder_output, _ = self.decoder(decoder_embedded,
                                         (encoder_LSTM_hidden,
                                          encoder_LSTM_cell))#給最後的hidden state
        logits = self.linear(decoder_output)

        return logits
    
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
    
if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    # dataset = ChatDataset(data,bert_tokenizer,max_len)
    # dataloader = DataLoader(dataset,batch_size=1)
    # d = next(iter(dataloader))
    # print(d)
    # 建立模型實例
    
    seq_len = 41

    # 載入繁體中文BERT模型作為Encoder
    bert_model_name = 'bert-base-chinese'
    bert_tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
    bert_model = BertModel.from_pretrained(bert_model_name)

    
    
    vocab_size = 5289
    
    model = Seq2Seq(bert_model=bert_model,
                    output_size=vocab_size,
                    embedding_dim = 300
                    ).to(device)
    
    # batch_size = 1
    
    #製造一個
    x = torch.LongTensor([[0]*seq_len]).to(device)
    x = x.to(torch.int)
    
    out = model(x,x) # works  
    #print(out.size())
    #print(x.size())
    summary(model,x,x)   