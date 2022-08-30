import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
import numpy as np

class Encoder(nn.Module):
    def __init__(self,config):
        super(Encoder,self).__init__()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.embedding_t = nn.Embedding(config.DICT_SIZE_2,config.MODEL_SIZE)
        self.embedding_v = nn.Embedding(config.DICT_SIZE_1,config.MODEL_SIZE)
        for i in range(config.NUM_LAYER):
            self.__setattr__("layer_{}".format(i),
            nn.LSTM(config.MODEL_SIZE,config.MODEL_SIZE))
    
    def forward(self,inputs):
        device = self.device
        config = self.config
        in1, in2 = inputs
        lengths = [len(x) for x in in1]
        in1 = [torch.tensor(x).to(device) for x in in1]
        in2 = [torch.tensor(x).to(device) for x in in2]
        in1 = rnn_utils.pad_sequence(in1)
        tensor1 = self.embedding_t(in1)
        in2 = rnn_utils.pad_sequence(in2)
        tensor2 = self.embedding_v(in2)
        tensor = tensor1 + tensor2
        for i in range(config.NUM_LAYER):
            skip = tensor
            tensor = rnn_utils.pack_padded_sequence(tensor,lengths,enforce_sorted=False)
            tensor, (h, c) = getattr(self,"layer_{}".format(i))(tensor)
            tensor = rnn_utils.pad_packed_sequence(tensor)[0]
            tensor = tensor + skip

        cx = c
        hx = h
        ys = [y[:i] for y,i in zip(torch.unbind(tensor,axis=1),lengths)]

        return ys, (hx, cx)

class Attn(nn.Module):
    def __init__(self,config):
        super(Attn,self).__init__()
        self.config = config
        self.Q = nn.Linear(config.MODEL_SIZE,config.MODEL_SIZE)
        self.K = nn.Linear(config.MODEL_SIZE,config.MODEL_SIZE)
        self.V = nn.Linear(config.MODEL_SIZE,config.MODEL_SIZE)
        self.W = nn.Linear(config.MODEL_SIZE,1)

    def forward(self,q,k,v,mask):
        q = self.Q(q)
        k = self.K(k)
        v = self.V(v)
        q = q.unsqueeze(1)
        k = k.unsqueeze(0)
        attn_weight = self.W(torch.tanh(q+k))
        attn_weight = attn_weight.squeeze(-1)
        attn_weight = torch.where(mask,attn_weight,torch.tensor(-1e6).to(q.device))
        attn_weight = attn_weight.softmax(1)
        attn_weight = attn_weight.unsqueeze(-1)
        context = attn_weight*v
        context = context.sum(1)
        return context

class Decoder(nn.Module):    
    def __init__(self,config):
        super(Decoder,self).__init__()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.attn = Attn(config)
        self.dropout1 = nn.Dropout(config.DROPOUT)
        self.dropout2 = nn.Dropout(config.DROPOUT)
        self.dropout3 = nn.Dropout(config.DROPOUT)
        self.embedding = nn.Embedding(config.DICT_SIZE_1,config.MODEL_SIZE)
        for i in range(config.NUM_LAYER):
            self.__setattr__("layer_{}".format(i),
            nn.LSTM(config.MODEL_SIZE,config.MODEL_SIZE))
        self.lstm = nn.LSTM(2*config.MODEL_SIZE,config.MODEL_SIZE)
        self.fc = nn.Linear(config.MODEL_SIZE,config.DICT_SIZE_1)

        self.loss_function = nn.CrossEntropyLoss(ignore_index=0)
    
    def forward(self,inputs,l_states,enc,mask):
        config = self.config
        tensor = self.embedding(inputs)
        for i in range(config.NUM_LAYER):
            skip = tensor
            tensor, l_states[i] = getattr(self,"layer_{}".format(i))(tensor,l_states[i])
            tensor = tensor + skip
        
        context = self.attn(tensor,enc,enc,mask)
        tensor = torch.cat([tensor,context],-1)
        tensor, l_states[-1] = self.lstm(tensor,l_states[-1])
        tensor = self.fc(tensor)

        return tensor, l_states

    def get_loss(self,enc,states,targets):
        device = self.device
        config = self.config

        targets = [torch.tensor([config.START_TOKEN]+x+[config.END_TOKEN]).to(device) for x in targets]
        targets = rnn_utils.pad_sequence(targets)
        inputs = targets[:-1]
        targets = targets[1:]

        mask = [torch.ones(x.shape[0]).to(device) for x in enc]
        mask = rnn_utils.pad_sequence(mask)
        mask = mask.unsqueeze(0)
        mask = mask.eq(1)
        enc = rnn_utils.pad_sequence(enc)

        h,c = states
        enc = self.dropout1(enc)
        h = self.dropout2(h)
        c = self.dropout3(c)
        l_states = [(h,c) for _ in range(config.NUM_LAYER+1)]

        tensor, l_states = self.forward(inputs,l_states,enc,mask)

        tensor = tensor.reshape(-1,config.DICT_SIZE_1)
        targets = targets.reshape(-1)
        loss = self.loss_function(tensor,targets)

        return loss

    def translate(self,enc,states):
        device = self.device
        config = self.config
        h,c = states
        lengths = [x.shape[0] for x in enc]
        mask = [torch.ones(x).to(device) for x in lengths]
        mask = rnn_utils.pad_sequence(mask)
        mask = mask.unsqueeze(0)
        mask = mask.eq(1)
        enc = rnn_utils.pad_sequence(enc)

        l_states = [(h,c) for _ in range(config.NUM_LAYER+1)]
        preds = [[config.START_TOKEN] for _ in range(len(lengths))]
        dec_input = torch.tensor(preds).to(device).view(1,-1)
        for _ in range(config.MAX_OUTPUT_SIZE):
            tensor, l_states = self.forward(dec_input,l_states,enc,mask)
            dec_input = torch.argmax(tensor,-1)[-1:]
            for i in range(len(lengths)):
                if preds[i][-1]!=config.END_TOKEN:
                    preds[i].append(int(dec_input[0,i]))
        preds = [x[1:-1] if x[-1]==config.END_TOKEN else x[1:] for x in preds]
        return preds

class Model(nn.Module):
    def __init__(self,config):
        super(Model,self).__init__()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.optimizer = optim.Adam(self.parameters(),lr=config.LR)
        self.to(self.device)
    
    def forward(self,inputs,mode,targets=None):
        if mode:
            return self.train_on_batch(inputs,targets)
        else:
            return self.translate(inputs)

    def train_on_batch(self,inputs,targets):
        self.optimizer.zero_grad()
        self.train()
        enc, (h,c) = self.encoder(inputs)
        loss =self.decoder.get_loss(enc,(h,c),targets)
        loss.backward()
        self.optimizer.step()
        return float(loss)
    
    def translate(self, inputs):
        with torch.no_grad():
            self.eval()
            enc, (h,c) = self.encoder(inputs)
            return self.decoder.translate(enc,(h,c))

    def save(self,path,name):
        torch.save(self.state_dict(),path)

    def load(self,path,name):
        self.load_state_dict(torch.load(path))