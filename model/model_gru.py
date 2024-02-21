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

        # 定義編碼器
        self.encoder_embedding = nn.Embedding(Embedding_input_vocab_size, embedding_dim)
        self.encoder_rnn = nn.GRU(embedding_dim, GRU_hidden_size ,batch_first=True)

        # 定義解碼器
        self.decoder_embedding = nn.Embedding(Embedding_output_vocab_size, embedding_dim)
        self.decoder_rnn = nn.GRU(embedding_dim, GRU_hidden_size ,batch_first=True)
        self.decoder_linear = nn.Linear(GRU_hidden_size, Embedding_output_vocab_size)

    def forward(self, input_seq, target_seq):
        encoder_embedded = self.encoder_embedding(input_seq)
        encoder_output, encoder_hidden = self.encoder_rnn(encoder_embedded)
        # print('encoder_hidden:',encoder_hidden.size())
        # print('decoder_embedded:',decoder_embedded.size())

        decoder_embedded = self.decoder_embedding(target_seq)
        decoder_output, _ = self.decoder_rnn(decoder_embedded, encoder_hidden[-1:])  # 只使用最後的隱藏狀態
        # print('decoder_output:',decoder_output.size())
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
    
    #print(x.size())
    #out = model(x,x) # works  
    #print(out.size())

    summary(model,x,x)