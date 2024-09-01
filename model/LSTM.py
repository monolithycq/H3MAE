import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size):
        super(RNN,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.rnn = nn.LSTM(self.input_size,self.hidden_size,self.num_layers,batch_first=True)
        #batch_first如果为True，输出数据格式是(batch, window_len,hidden_size)
        self.linear = nn.Linear(self.hidden_size,self.output_size)

    def forward(self,x,task):
        r_out, state = self.rnn(x.transpose(1,2))
        # print(state.shape)
        #h.shape (D∗num_layers,B,hidden_size)
        output = self.linear(torch.mean(r_out,dim=1)) #output:[batch_size,window_len,output_size]
        # print(output.shape)

        # output = output[:,-1,:]#output[batch_size,1,output_size]
        return output


class BiLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size):
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2*self.hidden_size, self.output_size)
    def forward(self,x, task):
        r_out, state = self.bilstm(x.transpose(1, 2))
        if task == 'classification':
            output = self.linear(torch.mean(r_out, dim=1))
        else:
            output = self.linear(r_out[:,-1,:])

        return output
