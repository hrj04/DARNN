import torch
import torch.nn as nn
import torch.nn.functional as F 


class DARNN(nn.Module):
    def __init__(self, hidden_dim_encoder, hidden_dim_decoder, stock_num, seq_len):
        super().__init__()  
        self.encoder = EncoderWithInputAttention(hidden_dim_encoder, stock_num, seq_len)
        self.decoder = DecoderWithTemporalAttention(hidden_dim_decoder, hidden_dim_encoder, seq_len)
        self.W_y = nn.Linear(hidden_dim_decoder+hidden_dim_encoder, hidden_dim_decoder)
        self.v_y = nn.Linear(hidden_dim_decoder, 1)
        
    def forward(self, x, y):
        h_t_enc = self.encoder(x)
        d_t, c_t = self.decoder(y, h_t_enc)
        
        # Equation (22)
        y_hat_t = self.v_y(self.W_y(torch.cat([d_t, c_t], dim=1))) # (batch, 1)
        
        return y_hat_t
    
class EncoderWithInputAttention(nn.Module):
    def __init__(self, hidden_dim_encoder, stock_num, seq_len):
        super().__init__()
        self.stock_num = stock_num
        self.seq_len = seq_len
        self.hidden_dim_encoder = hidden_dim_encoder 
        
        # Equation (3)-(7)
        self.lstm = nn.LSTM(input_size=stock_num, hidden_size=hidden_dim_encoder, batch_first=True)
        
        self.U_e = nn.Linear(seq_len, seq_len)
        self.W_e = nn.Linear(2*hidden_dim_encoder, seq_len)
        self.v_e = nn.Linear(self.seq_len, 1)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        batch_size, *_ = x.size()
        device = x.device
        h_t_enc = []
        for t in range(self.seq_len):
            x_t = x[:, t, :].unsqueeze(1) # (batch, 1, stock_num)
            if t == 0:
                h_t = torch.zeros(1, batch_size, self.hidden_dim_encoder).to(device)
                s_t = torch.zeros(1, batch_size, self.hidden_dim_encoder).to(device)
            e_t_k = []
            for k in range(self.stock_num):
                x_k = x[:, :, k] # (batch, seq_len)
                hs_t = torch.concat([h_t.squeeze(), s_t.squeeze()], dim=1) # (batch, 2*hidden_dim_encoder)
                
                # Equation (8)
                re1 = self.tanh(self.W_e(hs_t) + self.U_e(x_k)) # (batch, seq_len)
                e_t_k.append(self.v_e(re1)) # (batch, 1)
            e_t_k = torch.cat(e_t_k, dim=1) # (batch, stock_num)
            
            # Equation (9)
            alpha_t_k = F.softmax(e_t_k, dim=1).unsqueeze(1) # (batch, 1, stock_num)
            
            # Equation (10)
            x_tilde_t = alpha_t_k * x_t # (batch, 1, stock_num)
            
            # Equation (11)
            _, (h_t, s_t) = self.lstm(x_tilde_t, (h_t, s_t)) # (1, batch, hidden_dim_encoder)
            h_t_enc.append(h_t)
        h_t_enc = torch.cat(h_t_enc, dim=0) # (seq_len, batch, hidden_dim_encoder)
        h_t_enc = h_t_enc.permute(1, 0, 2) # (batch, seq_len, hidden_dim_encoder)
        return h_t_enc # (batch, seq_len, hidden_dim_encoder)
    
class DecoderWithTemporalAttention(nn.Module):
    def __init__(self, hidden_dim_decoder, hidden_dim_encoder, seq_len):
        super().__init__()
        self.hidden_dim_decoder = hidden_dim_decoder 
        self.seq_len = seq_len
        self.W_d = nn.Linear(2*hidden_dim_decoder, hidden_dim_encoder)
        self.U_d = nn.Linear(hidden_dim_encoder, hidden_dim_encoder)
        self.tanh = nn.Tanh()
        self.v_d = nn.Linear(hidden_dim_encoder, 1)
        self.w_tilde = nn.Linear(hidden_dim_encoder+1, 1)
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim_decoder, batch_first=True)
    
    def forward(self, y, h_t_enc):
        batch_size, *_ = y.size()
        device = y.device
        for t in range(self.seq_len-1):
            y_t = y[:, t] # (batch, 1)
            if t == 0:
                d_t = torch.zeros(1, batch_size, self.hidden_dim_decoder).to(device)
                s_t = torch.zeros(1, batch_size, self.hidden_dim_decoder).to(device)
            
            l_i_t = []
            for i in range(self.seq_len):
                h_i = h_t_enc[:,i,:] # (batch, hidden_dim_encoder)
                ds_t = torch.cat([d_t.squeeze(), s_t.squeeze()], dim=1) # (batch, 2*hidden_dim_decoder)
                
                # Equation (12)
                re1 = self.tanh(self.W_d(ds_t) + self.U_d(h_i)) # (batch, hidden_dim_encoder)
                l_i_t.append(self.v_d(re1)) # (batch, 1)
            l_i_t = torch.cat(l_i_t, dim=1) # (batch, seq_len)
            
            # Equation (13)
            beta_i_t = F.softmax(l_i_t, dim=1) # (batch, seq_len)
            
            # Equation (14)
            c_t = (beta_i_t.unsqueeze(-1) * h_t_enc).sum(dim=1) # (batch, hidden_dim_encoder)
            
            # Equation (15)
            y_tilde_t = self.w_tilde(torch.cat([c_t, y_t], dim=1)).unsqueeze(-1) # (batch, 1)
            
            # Equation (16)
            _, (d_t, s_t) = self.lstm(y_tilde_t, (d_t, s_t)) # (1, batch, hidden_dim_decoder)
            return d_t.squeeze(), c_t # (batch, hidden_dim_decoder), (batch, hidden_dim_encoder)
        
        
        
class EncoderWithInputAttention_Exp2(nn.Module):
    def __init__(self, hidden_dim_encoder, stock_num, seq_len):
        super().__init__()
        self.stock_num = stock_num
        self.seq_len = seq_len
        self.hidden_dim_encoder = hidden_dim_encoder 
        self.lstm = nn.LSTM(input_size=stock_num, hidden_size=hidden_dim_encoder, batch_first=True)
        
        self.U_e = nn.Linear(seq_len, seq_len)
        self.W_e = nn.Linear(2*hidden_dim_encoder, seq_len)
        self.v_e = nn.Linear(self.seq_len, 1)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        batch_size, *_ = x.size()
        device = x.device
        h_t_enc = []
        for t in range(self.seq_len):
            x_t = x[:, t, :].unsqueeze(1) # (batch, 1, stock_num)
            if t == 0:
                h_t = torch.zeros(1, batch_size, self.hidden_dim_encoder).to(device)
                s_t = torch.zeros(1, batch_size, self.hidden_dim_encoder).to(device)
            e_t_k = []
            for k in range(self.stock_num):
                x_k = x[:, :, k] # (batch, seq_len)
                hs_t = torch.concat([h_t.squeeze(), s_t.squeeze()], dim=1) # (batch, 2*hidden_dim_encoder)
                re1 = self.tanh(self.W_e(hs_t) + self.U_e(x_k)) # (batch, seq_len)
                e_t_k.append(self.v_e(re1)) # (batch, 1)
            e_t_k = torch.cat(e_t_k, dim=1) # (batch, stock_num)
            alpha_t_k = F.softmax(e_t_k, dim=1).unsqueeze(1) # (batch, 1, stock_num)
            x_tilde_t = alpha_t_k * x_t # (batch, 1, stock_num)
            
            _, (h_t, s_t) = self.lstm(x_tilde_t, (h_t, s_t)) # (1, batch, hidden_dim_encoder)
            h_t_enc.append(h_t)
        h_t_enc = torch.cat(h_t_enc, dim=0) # (seq_len, batch, hidden_dim_encoder)
        h_t_enc = h_t_enc.permute(1, 0, 2) # (batch, seq_len, hidden_dim_encoder)
        return h_t_enc, alpha_t_k

class DARNN_Exp2(nn.Module):
    def __init__(self, hidden_dim_encoder, hidden_dim_decoder, stock_num, seq_len):
        super().__init__()  
        self.encoder = EncoderWithInputAttention_Exp2(hidden_dim_encoder, stock_num, seq_len)
        self.decoder = DecoderWithTemporalAttention(hidden_dim_decoder, hidden_dim_encoder, seq_len)
        self.W_y = nn.Linear(hidden_dim_decoder+hidden_dim_encoder, hidden_dim_decoder)
        self.v_y = nn.Linear(hidden_dim_decoder, 1)
        
    def forward(self, x, y):
        h_t_enc, alpha_t_k = self.encoder(x)
        d_t, c_t = self.decoder(y, h_t_enc)
        y_hat_t = self.v_y(self.W_y(torch.cat([d_t, c_t], dim=1))) # (batch, 1)
        
        return y_hat_t, alpha_t_k