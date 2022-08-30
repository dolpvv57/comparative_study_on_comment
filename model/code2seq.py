import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils

class Encoder(nn.Module):
    def __init__(self,config):
        super(Encoder,self).__init__()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.embedding = nn.Embedding(config.DICT_SIZE_1,config.MODEL_SIZE)
        for i in range(config.NUM_LAYER):
            self.__setattr__("layer_fw_{}".format(i),nn.LSTM(config.MODEL_SIZE,config.MODEL_SIZE))
            self.__setattr__("layer_bw_{}".format(i),nn.LSTM(config.MODEL_SIZE,config.MODEL_SIZE))
        #self.lstm = nn.LSTM(config.ENC_SIZE,config.ENC_SIZE,config.NUM_LAYER,bidirectional=True)
        self.fc = nn.Linear(4*config.MODEL_SIZE,config.MODEL_SIZE)
    
    def forward(self,inputs):
        device = self.device
        config = self.config
        paths = [path for x in inputs for path in x]    #inputs是64个样本组成的列表，每个样本也是一个列表，里面有200条路径，一共有64*200=12800条路径
        lengths = [min([len(x),config.MAX_SBT_LEN]) for x in paths]     #统计每一条路径的长度，大于20的取20
        fw_paths = [torch.tensor(path).to(device) for path in paths]    #每一条path取tensor
        bw_paths = [path.flip(0) for path in fw_paths]  #flip表示对某一个轴反向排序，这里相当于对每一条路径反向排列
        fw_paths = [path[:config.MAX_SBT_LEN] for path in fw_paths]     #每一条路径截断
        bw_paths = [path[:config.MAX_SBT_LEN] for path in bw_paths]     #每一条路径截断
        fw_paths = rnn_utils.pad_sequence(fw_paths)     #补0到batch中的最大长度，维度(20,12800)
        bw_paths = rnn_utils.pad_sequence(bw_paths)     #补0到batch中的最大长度，维度(20,12800)
        fw_paths = self.embedding(fw_paths)     #embedding，维度(20,12800,256)
        bw_paths = self.embedding(bw_paths)     #embedding，维度(20,12800,256)
        start = fw_paths[0:1]   #取所有路径的第一个sequence
        end = bw_paths[0:1]     #取所有路径的第一个sequence，也就是实际的最后一个sequence

        #tensor = fw_paths
        #tensor = rnn_utils.pack_padded_sequence(tensor,lengths,enforce_sorted=False)
        #tensor,(h,c) = self.lstm(tensor)
        #tensor = rnn_utils.pad_packed_sequence(tensor)[0]
        #h = h.view(config.NUM_LAYER,2,-1,config.ENC_SIZE)
        #h = torch.cat([h[-1,0:1],h[-1,1:2]],-1)
        
        tensor = fw_paths
        for i in range(config.NUM_LAYER):   #正向LSTM
            skip = tensor
            tensor = rnn_utils.pack_padded_sequence(tensor,lengths,enforce_sorted=False)    #删掉占位符0，此时维度为(*,12800,256)
            tensor, (fh, fc) = getattr(self,"layer_fw_{}".format(i))(tensor)    #获取LSTM模型，输入tensor(*,12800,256)，即(time_step,batch_size,词向量长度input_size),输出的tensor维度(*,12800,256)，h,c分别代表分线程和主线程的hidde state,Ht和Ct(最后一个时刻)，维度均为(1,12800,256)，初始h和c默认为0
            tensor = rnn_utils.pad_packed_sequence(tensor)[0]   #长度不够补0到最长的mask，为400，此时返回一个元组，第一项是padding后的结果，第二项是去掉0的原来序列长度，取第一项，tensor维度(20,12800,256)
            tensor = tensor + skip  #把之前的tensor和新生成的tensor合起来作为下一层的输入tensor
        
        tensor = bw_paths
        for i in range(config.NUM_LAYER):   ##反向LSTM
            skip = tensor
            tensor = rnn_utils.pack_padded_sequence(tensor,lengths,enforce_sorted=False)        #删掉占位符0，此时维度为(*,12800,256)
            tensor, (bh, bc) = getattr(self,"layer_bw_{}".format(i))(tensor)    #获取LSTM模型，输入tensor(*,12800,256)，即(time_step,batch_size,词向量长度input_size),输出的tensor维度(*,12800,256)，h,c分别代表分线程和主线程的hidde state,Ht和Ct(最后一个时刻)，维度均为(1,12800,256)，初始h和c默认为0
            tensor = rnn_utils.pad_packed_sequence(tensor)[0]   #长度不够补0到最长的mask，为400，此时返回一个元组，第一项是padding后的结果，第二项是去掉0的原来序列长度，取第一项，tensor维度(20,12800,256)
            tensor = tensor + skip      #把之前的tensor和新生成的tensor合起来作为下一层的输入tensor
        
        h = torch.cat([fh,bh],-1)   #把前向和后向LSTM的h合并，维度(1,12800,512)，h是最后一个时刻的
        tensor = torch.cat([start,h,end],-1)    #第一个sequence和最后一个sequence接上，维度(1,12800,1024)
        tensor = self.fc(tensor)    #全连接层，(1,12800,256)
        tensor = torch.tanh(tensor)     #tanh激活
        tensor = torch.unbind(tensor,1)     #以第一维度(12800)做切片，得到12800个元素，每个元素(1,256)
        tensor = [torch.cat(tensor[i:i+config.K],0) for i in range(0,len(tensor),config.K)]     #每200条路径合成一组，相当于一个样本，tensor有64个元素，每个元素里的size是(200,256)

        return tensor

class Attn(nn.Module):
    def __init__(self,config):
        super(Attn,self).__init__()
        self.config = config
        self.Q = nn.Linear(config.MODEL_SIZE,config.MODEL_SIZE)
        self.K = nn.Linear(config.MODEL_SIZE,config.MODEL_SIZE)
        self.V = nn.Linear(config.MODEL_SIZE,config.MODEL_SIZE)
        self.W = nn.Linear(config.MODEL_SIZE,1)

    def forward(self,q,k,v,mask):
        config = self.config
        q = self.Q(q)
        k = self.K(k)
        v = self.V(v)
        q = q.unsqueeze(1)
        k = k.unsqueeze(0)
        attn_weight = self.W(torch.tanh(q+k))
        attn_weight = attn_weight.squeeze(-1)
        attn_weight = torch.where(mask==1,attn_weight,torch.tensor(-1e6).to(q.device))
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
        for i in range(config.NUM_LAYER-1):
            self.__setattr__("layer_{}".format(i),
            nn.LSTM(config.MODEL_SIZE,config.MODEL_SIZE))
        self.lstm = nn.LSTM(2*config.MODEL_SIZE,config.MODEL_SIZE)
        self.fc = nn.Linear(config.MODEL_SIZE,config.DICT_SIZE_1)

        self.loss_function = nn.CrossEntropyLoss(reduction='sum')
    
    def forward(self,inputs,l_states,enc,mask):
        config = self.config
        tensor = self.embedding(inputs)
        for i in range(config.NUM_LAYER-1):
            skip = tensor
            tensor, l_states[i] = getattr(self,"layer_{}".format(i))(tensor,l_states[i])
            tensor = tensor + skip
        
        context = self.attn(tensor,enc,enc,mask)
        tensor, l_states[-1] = self.lstm(torch.cat([tensor,context],-1),l_states[-1])
        tensor = self.fc(tensor)

        return tensor, l_states

    def get_loss(self,enc,targets):
        device = self.device
        config = self.config
        lengths1 = [len(x)+1 for x in targets]
        targets = [torch.tensor([config.START_TOKEN]+x+[config.END_TOKEN]).to(device) for x in targets]
        targets = rnn_utils.pad_sequence(targets)
        lengths2 = [x.shape[0] for x in enc]
        mask = [torch.ones(x).to(device) for x in lengths2]
        mask = rnn_utils.pad_sequence(mask)
        mask = mask.unsqueeze(0)
        mask = mask.repeat(max(lengths1),1,1)
        enc = rnn_utils.pad_sequence(enc)
        dec_hidden = enc.mean(0,True)
        dec_cell = enc.mean(0,True)
        enc = self.dropout1(enc)
        dec_hidden = self.dropout2(dec_hidden)
        dec_cell = self.dropout3(dec_cell)
        l_states = [(dec_hidden,dec_cell) for _ in range(config.NUM_LAYER)]
        inputs = targets[:-1]
        targets = targets[1:]
        tensor,l_states = self.forward(inputs,l_states,enc,mask)
        loss = 0
        for i in range(len(lengths1)):
            loss = loss + self.loss_function(tensor[:lengths1[i],i],targets[:lengths1[i],i])

        return loss / sum(lengths1)

    def translate(self,enc):
        device = self.device
        config = self.config
        lengths = [x.shape[0] for x in enc]
        mask = [torch.ones(x).to(device) for x in lengths]
        mask = rnn_utils.pad_sequence(mask)
        mask = mask.unsqueeze(0)
        enc = rnn_utils.pad_sequence(enc)
        dec_hidden = enc.mean(0,True)
        dec_cell = enc.mean(0,True)
        l_states = [(dec_hidden,dec_cell) for _ in range(config.NUM_LAYER)]
        preds = [[config.START_TOKEN] for _ in range(len(lengths))]
        dec_input = torch.tensor(preds).to(device).view(1,-1)
        for t in range(config.MAX_OUTPUT_SIZE):
            tensor, l_states = self.forward(dec_input,l_states,enc,mask)
            dec_input = torch.argmax(tensor,-1)[-1:].detach()
            for i in range(len(lengths)):
                if preds[i][-1]!=config.END_TOKEN:
                    preds[i].append(int(dec_input[0,i]))
        preds = [x[1:-1] if x[-1]==config.END_TOKEN else x[1:] for x in preds]
        return preds


class Code2Seq(nn.Module):
    def __init__(self,config):
        super(Code2Seq,self).__init__()
        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed(config.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.set_trainer()
        self.to(self.device)
    
    def forward(self,inputs,mode,targets=None):
        #torch.cuda.empty_cache()
        if mode:
            return self.train_on_batch(inputs,targets)
        else:
            return self.translate(inputs)

    def train_on_batch(self,inputs,targets):
        self.optimizer.zero_grad()
        self.train()
        enc = self.encoder(inputs)
        loss =self.decoder.get_loss(enc,targets)
        loss.backward()
        self.optimizer.step()
        return float(loss)
    
    def translate(self, inputs):
        self.eval()
        enc = self.encoder(inputs)
        return self.decoder.translate(enc)

    def save(self,path):
        checkpoint = {
            'config':self.config,
            'encoder':self.encoder,
            'decoder':self.decoder,
            'optimizer':self.optimizer
        }
        torch.save(checkpoint,path)

    def load(self,path):
        checkpoint = torch.load(path)
        self.config = checkpoint['config']
        self.encoder = checkpoint['encoder']
        self.decoder = checkpoint['decoder']
        self.optimizer = checkpoint['optimizer']

    def set_trainer(self):
        config = self.config
        self.optimizer = optim.Adam(params=[
            {"params":self.parameters()}
        ],lr=config.LR)