import torch
from torch import nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    def __init__(self, 
                vocab_size,    # 词典大小
                embedding_dim, # 词向量维度
                pretrained_weight, # 预训练权重
                update_w2v, # 是否更新模型参数
                hidden_dim, # LSTM隐藏层维度
                num_layers, # LSTM隐藏层数量
                drop_keep_prob, # dropout的概率
                n_class,   # 标签的类别数量
                bidirectional, # 是否使用双向LSTM
                **kwargs):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_class = n_class
        self.bidirectional = bidirectional

        # 使用预训练的Embedding
        self.embedding = nn.Embedding.from_pretrained(pretrained_weight)
        self.embedding.weight.requires_grad = update_w2v

        self.lstm_layer = nn.LSTM(input_size=embedding_dim, 
                                hidden_size=self.hidden_dim,
                                num_layers=self.num_layers, 
                                bidirectional=self.bidirectional,
                                dropout=drop_keep_prob)

        if self.bidirectional:
            # 双向LSTM的维度为隐藏层维度*4
            self.dense1 = nn.Linear(hidden_dim * 4, hidden_dim)
            self.dense2 = nn.Linear(hidden_dim, n_class)
        else:
            # 双向LSTM的维度为隐藏层维度*2
            self.dense1 = nn.Linear(hidden_dim * 2, hidden_dim)
            self.dense2 = nn.Linear(hidden_dim, n_class)            

def forward(self, inputs):

    # 进行word embddding
    embeddings = self.embedding(inputs)

    # 将编码后的embeddings传入LSTM
    # embeddings.permute([1, 0, 2])
    # [seq_length, batch_size, embedding_dim]
    states, hidden = self.lstm_layer(embeddings.permute([1, 0, 2]))

    # 将双向LSTM的输出进行拼接
    # 正向LSTM的输出为states[-1]
    # 反向LSTM的输出为states[0]
    encoding = torch.cat([states[0], states[-1]], dim=1)

    # 接两个全连接层进行分类输出
    outputs = self.dense1(encoding)
    outputs = self.dense2(outputs)

    return outputs


class LSTM_attention(nn.Module):
    def __init__(self, 
                vocab_size,    # 词典大小
                embedding_dim, # 词向量维度
                pretrained_weight, # 预训练权重
                update_w2v, # 是否更新模型参数
                hidden_dim, # LSTM隐藏层维度
                num_layers, # LSTM隐藏层数量
                drop_keep_prob, # dropout的概率
                n_class,   # 标签的类别数量
                bidirectional, # 是否使用双向LSTM
                **kwargs):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_class = n_class
        self.bidirectional = bidirectional      

        # 使用预训练的Embedding
        self.embedding = nn.Embedding.from_pretrained(pretrained_weight)
        self.embedding.weight.requires_grad = update_w2v

        self.lstm_layer = nn.LSTM(input_size=embedding_dim, 
                                hidden_size=self.hidden_dim,
                                num_layers=num_layers, 
                                bidirectional=self.bidirectional,
                                dropout=drop_keep_prob)        
        
        
        # 用来做attention的可学习的矩阵
        self.weight_W = nn.Parameter(torch.Tensor(self.num_layers*hidden_dim, 2*hidden_dim))    
        self.weight_proj = nn.Parameter(torch.Tensor(self.num_layers*hidden_dim, 1)) 

        if self.bidirectional:
            self.dense1 = nn.Linear(hidden_dim * 2, hidden_dim)
            self.dense2 = nn.Linear(hidden_dim, n_class)
        else:
            self.dense1 = nn.Linear(hidden_dim * 2, hidden_dim)
            self.dense2 = nn.Linear(hidden_dim, n_class)         

        nn.init.uniform_(self.weight_W, -0.1, 0.1) 
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)     

    def forward(self, inputs):

        # 进行word embedding
        embeddings = self.embedding(inputs)

        # 将编码后的embeddings传入LSTM
        # embeddings.permute([0, 0, 2])
        # [batch_size, seq_length, embedding_dim]
        states, hidden = self.lstm_layer(embeddings.permute([0, 1, 2]))

        # attention操作
        # att = self.weight_proj * tanh(states * self.weight_W)

        # 先将LSTM输出的所有hidden_states和weight_W相乘
        # 再做激活函数tanh做非线性变换
        u = torch.tanh(torch.matmul(states, self.weight_W))

        # 再和weight_proj矩阵相乘得到attention score矩阵
        att = torch.matmul(u, self.weight_proj)

        #再经过softmax变换转换成0~1之间的概率值
        att_score = F.softmax(att, dim=1)

        # 每个输出的states乘以attention score得到不同的权重
        score_x = states * att_score
        
        # 每个socre_x求和得到最终的encoding
        encoding = torch.sum(score_x, dim=1)

        # 再经过两个全连接层进行分类输出
        outputs = self.dense1(encoding)
        outputs = self.dense2(outputs)

        return outputs