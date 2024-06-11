import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class Learner(nn.Module):
    def __init__(self, args, hid_dim = 128, weights = None,):
        super(Learner, self).__init__()
        self.block_1 = nn.Sequential(nn.Linear(args.input_dim, hid_dim), nn.LeakyReLU(0.1))
        self.block_2 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.LeakyReLU(0.1))
        self.fclayer = nn.Sequential(nn.Linear(hid_dim, 1))
        self.dropout = args.use_dropout
        self.fc_dropout = nn.Dropout(0.2)

        if weights != None:
            self.load_state_dict(deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))

    def forward_mixup(self, x1, x2, lam=None):
        x1 = self.block_1(x1)
        if self.dropout:
            x1 = self.fc_dropout(x1)
        x2 = self.block_1(x2)
        if self.dropout:
            x2 = self.fc_dropout(x2)
        x = lam * x1 + (1 - lam) * x2
        
        x = self.block_2(x)
        output = self.fclayer(x)
        return output

    def forward(self, x, mc_dropout=False):
        x = self.block_1(x)
        if self.dropout:
            if mc_dropout:
                x = F.dropout(x, 0.2, training=True)
            else:
                x = self.fc_dropout(x)
        x = self.block_2(x)
        output = self.fclayer(x)
        return output

    def repr_forward(self, x):
        with torch.no_grad():
            x = self.block_1(x)
            repr = self.block_2(x)
            return repr
        
    def repr_forward1(self, x):
        with torch.no_grad():
            return self.block_1(x)
    

# ---> :https://github.com/laiguokun/LSTNet
class Learner_TimeSeries(nn.Module):
    def __init__(self, args, data, weights = None):
        super(Learner_TimeSeries, self).__init__()
        self.use_cuda = args.cuda
        self.P = int(args.window)
        # self.m = int(data.m)
        self.m = int(data[1]) # ts_data = [scale, m]
        self.hidR = int(args.hidRNN)
        self.hidC = int(args.hidCNN)
        self.hidS = int(args.hidSkip)
        self.Ck = args.CNN_kernel
        self.skip = args.skip
        self.pt = int((self.P - self.Ck)/self.skip)
        print(f'self.pt = {self.pt}')
        self.hw = args.highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p = args.dropout)
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            print(self.hidR + self.skip * self.hidS, self.m)
            self.linear1 = nn.Linear(int(self.hidR + self.skip * self.hidS), self.m)
        else:
            self.linear1 = nn.Linear(self.hidR, self.m)
        if (self.hw > 0): #highway -> autoregressiion
            self.highway = nn.Linear(self.hw, 1)
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = F.tanh

        if weights != None:
            self.load_state_dict(deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))

    def forward(self, x, mc_dropout=False):
        batch_size = x.size(0)
        #CNN
        c = x.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        if mc_dropout:
            c = F.dropout(c, 0.2, training=True)
        else:
            c = self.dropout(c)
        c = torch.squeeze(c, 3)
        
        # RNN time number <-> layer number
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        if mc_dropout:
            r = F.dropout(torch.squeeze(r,0), 0.2, training=True)
        else:
            r = self.dropout(torch.squeeze(r,0))

        
        #skip-rnn
        
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, int(self.pt), int(self.skip))
            s = s.permute(2,0,3,1).contiguous()
            s = s.view(int(self.pt), int(batch_size * self.skip), int(self.hidC))
            _, s = self.GRUskip(s)
            s = s.view(batch_size, int(self.skip * self.hidS))
            if mc_dropout:
                s = F.dropout(s, 0.2, training=True)
            else:
                s = self.dropout(s)
            r = torch.cat((r,s),1)
        
        # FC
        res = self.linear1(r)
        
        #highway auto-regression
        if (self.hw > 0):
            z = x[:, -self.hw:, :]
            z = z.permute(0,2,1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1,self.m)
            res = res + z
            
        if (self.output):
            res = self.output(res)

        return res

    def repr_forward(self, x):
        batch_size = x.size(0)
        #CNN
        c = x.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)
        
        # RNN time number <-> layer number
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r,0))

        
        #skip-rnn
        
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, int(self.pt), int(self.skip))
            s = s.permute(2,0,3,1).contiguous()
            s = s.view(int(self.pt), int(batch_size * self.skip), int(self.hidC))
            _, s = self.GRUskip(s)
            s = s.view(batch_size, int(self.skip * self.hidS))
            s = self.dropout(s)
            r = torch.cat((r,s),1)
        
        # FC
        return r
        res = self.linear1(r)
        
        #highway auto-regression
            

    def forward_mixup(self, x1, x2, lam):
        batch_size = x1.size(0)
        #CNN
        c1 = x1.view(-1, 1, self.P, self.m)
        c1 = F.relu(self.conv1(c1))
        c1 = self.dropout(c1)
        c1 = torch.squeeze(c1, 3)

        #CNN
        c2 = x2.view(-1, 1, self.P, self.m)
        c2 = F.relu(self.conv1(c2))
        c2 = self.dropout(c2)
        c2 = torch.squeeze(c2, 3)
        
        # just mixup after conv block
        c = lam * c1 + (1 - lam) * c2

        # RNN time number <-> layer number
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r,0))

        #skip-rnn
        
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, int(self.pt), int(self.skip))
            s = s.permute(2,0,3,1).contiguous()
            s = s.view(int(self.pt), int(batch_size * self.skip), int(self.hidC))
            _, s = self.GRUskip(s)
            s = s.view(batch_size, int(self.skip * self.hidS))
            s = self.dropout(s)
            r = torch.cat((r,s),1)
        
        # FC
        res = self.linear1(r)
        
        #highway auto-regression --> not mixup
        if (self.hw > 0):
            x = lam * x1 + (1 - lam) * x2
            z = x[:, -self.hw:, :]
            z = z.permute(0,2,1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1,self.m)
            res = res + z
            
        if (self.output):
            res = self.output(res)
        return res