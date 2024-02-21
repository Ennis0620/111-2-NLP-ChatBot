import torch
import torch.nn as nn
#from torchinfo import summary
from torchsummaryX import summary
from transformers import BertTokenizerFast, BertModel
from transformers import BertConfig

class Seq2Seq(nn.Module):
    def __init__(self, bert_model, output_size, embedding_dim):
        super(Seq2Seq, self).__init__()
        self.encoder = bert_model
        self.encoder.requires_grad_(False)  # 設定BERT模型的requires_grad為False
        self.encoder.eval()

        self.decoder_embedding = nn.Embedding(output_size, embedding_dim)
        self.decoder = nn.LSTM(embedding_dim, 768, batch_first=True)
        self.linear = nn.Linear(768, output_size)

    def forward(self, input1, input2):

        encoder_hidden_state = self.encoder(input1).last_hidden_state  # 用bert的最後的hidden state shape:[batch_size, seq_len, hidden_size]
        encoder_hidden_last = encoder_hidden_state[:, -1, :].unsqueeze(0).contiguous() #保持tensor的連續性 

        decoder_embedded = self.decoder_embedding(input2)
        decoder_output, _ = self.decoder(decoder_embedded,
                                         (encoder_hidden_last,
                                          encoder_hidden_last))
        logits = self.linear(decoder_output)

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

    print(bert_model.config.hidden_size)
    print("Vocabulary Size:", vocab_size)
    
    model = Seq2Seq(bert_model=bert_model,
                    output_size=vocab_size,
                    embedding_dim = 300
                    ).to(device)

    # batch_size = 1
    
    #製造一個
    x = torch.LongTensor([[0]*seq_len]).to(device)
    x = x.to(torch.int)
    
    out = model(x,x) # works  
    print(out.size())
    # print(x.size())
    summary(model,x,x)   