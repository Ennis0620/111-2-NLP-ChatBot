import torch
import torch.nn as nn
#from torchinfo import summary
from torchsummaryX import summary

class Seq2Seq(nn.Module):
    def __init__(self, Embedding_input_vocab_size, 
                 Embedding_output_vocab_size, 
                 GRU_hidden_size, 
                 embedding_dim):
        super(Seq2Seq, self).__init__()
        self.GRU_hidden_size = GRU_hidden_size
        self.LSTM_hidden_size = GRU_hidden_size
        # 定義編碼器
        self.encoder_embedding = nn.Embedding(Embedding_input_vocab_size, embedding_dim)
        self.encoder_lstm = nn.LSTM(embedding_dim, self.LSTM_hidden_size, batch_first=True)

        # 定義解碼器
        self.decoder_embedding = nn.Embedding(Embedding_output_vocab_size, embedding_dim)
        self.decoder_lstm = nn.LSTM(embedding_dim, self.LSTM_hidden_size, batch_first=True)
        self.decoder_linear = nn.Linear(GRU_hidden_size, Embedding_output_vocab_size)

    def forward(self, input1, input2):

        encoder_embedded = self.encoder_embedding(input1)
        encoder_output, (encoder_hidden, encoder_cell) = self.encoder_lstm(encoder_embedded)
        # print(encoder_embedded)
        # print(encoder_hidden)
        # print(encoder_cell)
        # print('encoder_hidden:',encoder_hidden.size())
        # print('encoder_cell:',encoder_cell.size())
        # print(encoder_hidden[-1:].size())
        # print(encoder_cell[-1:].size())

        decoder_embedded = self.decoder_embedding(input2)
        # print(decoder_embedded)
        # print(decoder_embedded.size())
        decoder_output, _ = self.decoder_lstm(decoder_embedded, 
                                              (encoder_hidden[-1:], encoder_cell[-1:]))
        decoder_output = self.decoder_linear(decoder_output)

        return decoder_output


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 建立模型實例
    input_vocab_size = 5289
    output_vocab_size = 5289
    hidden_size = 512
    embedding_dim = 200
    seq_len = 41
    
    model = Seq2Seq(Embedding_input_vocab_size=input_vocab_size, 
                   Embedding_output_vocab_size= output_vocab_size,
                   GRU_hidden_size = hidden_size, 
                   embedding_dim = embedding_dim,).to(device)

    batch_size = 1
    
    #製造一個
    x = torch.LongTensor([[0]*seq_len]).to(device)
    x = x.to(torch.int)
    
    print(x.size())
    #out = model(x,x) # works  
    #print(out.size())

    summary(model,x,x)