import torch
import torch.nn as nn
#from torchinfo import summary
from torchsummaryX import summary
from transformers import BertTokenizerFast, BertModel
from transformers import BertConfig

class Seq2Seq(nn.Module):
    def __init__(self, bert_model, output_size):
        super(Seq2Seq, self).__init__()
        #直接用bert取代 encoder和decoder的embedding
        self.emd = bert_model
        self.emd.requires_grad_(False)  # 設定BERT模型的requires_grad為False
        self.emd.eval()

        self.encoder_LSTM = nn.LSTM(768 ,512 ,batch_first=True)
        self.decoder_LSTM = nn.LSTM(768, 512, batch_first=True)

        self.linear = nn.Linear(512, output_size)

    def forward(self, input1, input2):
        #直接用bert取代 encoder和decoder的embedding
        encoder_hidden_state = self.emd(input1).last_hidden_state  # 用bert的最後的hidden state shape:[batch_size, seq_len, hidden_size]
        encoder_LSTM_out,(encoder_LSTM_hidden,encoder_LSTM_cell) = self.encoder_LSTM(encoder_hidden_state)
        print('encoder_LSTM_hidden:',encoder_LSTM_hidden.size())
        decoder_hidden_state = self.emd(input2).last_hidden_state # 用bert的最後的hidden state shape:[batch_size, seq_len, hidden_size]
        print('decoder_hidden_state:',decoder_hidden_state.size())
        decoder_output, _ = self.decoder_LSTM(decoder_hidden_state,
                                         (encoder_LSTM_hidden,
                                          encoder_LSTM_cell))#給最後的hidden state
        
        print('decoder_output:',decoder_output.size()) 
        logits = self.linear(decoder_output)
        # print('logits:',logits.size())
        return logits
    
if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 建立模型實例
    
    seq_len = 41

    # 載入繁體中文BERT模型作為Encoder
    bert_model_name = 'bert-base-chinese'
    bert_tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
    bert_model = BertModel.from_pretrained(bert_model_name)

    config = BertConfig.from_pretrained(bert_model_name)
    #print(config)
    vocab_size = config.vocab_size
    
    # print(bert_model.config.hidden_size)
    print("Vocabulary Size:", vocab_size)
    
    model = Seq2Seq(bert_model=bert_model,
                    output_size=vocab_size,
                    ).to(device)

    # batch_size = 1
    
    #製造一個
    x = torch.LongTensor([[0]*seq_len]).to(device)
    x = x.to(torch.int)
    
    out = model(x,x) # works  
    #print(out.size())
    #print(x.size())
    summary(model,x,x)   