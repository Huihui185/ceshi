import math
import typing
import numpy as np
import torch
import torch.nn as nn
from torch.functional import Tensor
import torch.nn.functional as F
# from channel_attention import ChannelAttention
# from revin_norm import RevIN


# attention 模块
class AttentionBlock(nn.Module):
  """An attention mechanism similar to Vaswani et al (2017)
  The input of the AttentionBlock is `BxTxD` where `B` is the input
  minibatch size, `T` is the length of the sequence `D` is the dimensions of
  each feature.
  The output of the AttentionBlock is `BxTx(D+V)` where `V` is the size of the
  attention values.
  Arguments:
      dims (int): the number of dimensions (or channels) of each element in
          the input sequence
      k_size (int): the size of the attention keys
      v_size (int): the size of the attention values
      seq_len (int): the length of the input and output sequences
  """
  def __init__(self, dims, k_size, v_size, seq_len=None):
    super(AttentionBlock, self).__init__()
    # self.key_layer = nn.Linear(dims, k_size)
    # self.query_layer = nn.Linear(dims, k_size)
    # self.value_layer = nn.Linear(dims, v_size)
    self.key_layer = nn.Conv1d(dims, k_size,5,padding=2)
    self.query_layer = nn.Conv1d(dims, k_size,5,padding=2)
    self.value_layer = nn.Conv1d(dims, v_size,5,padding=2)

    self.sqrt_k = math.sqrt(k_size)

  def forward(self, minibatch):
    keys = self.key_layer(minibatch)
    queries = self.query_layer(minibatch)
    values = self.value_layer(minibatch)
    logits = torch.bmm(queries, keys.transpose(2,1))
    # Use numpy triu because you can't do 3D triu with PyTorch
    # TODO: using float32 here might break for non FloatTensor inputs.
    # Should update this later to use numpy/PyTorch types of the input.
    mask = np.triu(np.ones(logits.size()), k=1).astype('uint8')
    mask = torch.from_numpy(mask)
    # do masked_fill_ on data rather than Variable because PyTorch doesn't
    # support masked_fill_ w/-inf directly on Variables for some reason.
    #logits.data.masked_fill_(mask, float('-inf'))
    #logits.data.masked_fill(mask, float('-inf'))
    probs = F.softmax(logits, dim=1) / self.sqrt_k
    read = torch.bmm(probs, values)
    return minibatch + read




# 子单元
class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvGRUCell, self).__init__()
        """
        x dim = 3
        b * h = b & h should have same dim
        hidden channels for h = intermediate_channels
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # 卷积核为一个数组
        self.kernel_size = kernel_size
        # 填充为高和宽分别填充的尺寸
        # self.padding_size = kernel_size[0] // 2, kernel_size[1] // 2,
        self.padding_size = kernel_size // 2
        self.bias = bias

        self.conv_x = nn.Conv1d(
            self.input_dim + self.hidden_dim, self.hidden_dim * 2,
            kernel_size=kernel_size, padding= self.padding_size, bias=True
        )
        self.conv_y = nn.Conv1d(
            self.input_dim + self.hidden_dim, self.hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding_size,  bias=True
        )
        self.conv_m = nn.Conv1d(
            self.hidden_dim, self.hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding_size, bias=True
        )
        self.attention = AttentionBlock(200,128 , 200)
        self.intermediate_channels = self.hidden_dim
    def forward(self, x, h):
        y = x.clone()
        x = torch.cat([x, h], dim=1)
        x = self.conv_x(x)
        a, b = torch.split(x, self.intermediate_channels, dim=1)
        a = torch.sigmoid(a)
        b = torch.sigmoid(b)
        y = torch.cat([y, b * h], dim=1)
        y = torch.tanh(self.conv_y(y))
        h = a * h + (1 - a) * y

        # aattention
        # mt = torch.tanh(self.conv_m(h))
        # at = F.softmax(mt, 1)
        # r = h * (at+1)
        # r = self.attention(h)
        return h
    def init_hidden(self, batch_size, sequence):
        # 返回两个是因为cell的尺寸与h一样
        return torch.zeros(batch_size, self.hidden_dim, sequence, device=self.conv_x.weight.device)


class ConvGru(nn.Module):
    def __init__(self, seq,input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True,
                 return_all_layers=False):
        super(ConvGru, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        # 为了储存每一层的参数尺寸
        cell_list = []
        for i in range(0, num_layers):
            # 注意这里利用lstm单元得出到了输出h，h再作为下一层的输入，依次得到每一层的数据维度并储存
            cur_input_dim = input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvGRUCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size,
                                          bias=self.bias
                                          ))
        # 将上面循环得到的每一层的参数尺寸/维度，储存在self.cell_list中，后面会用到
        # 师兄加的
        cell_list.append(nn.Dropout(p=0))
        cell_list.append(nn.Linear(in_features=seq, out_features=1))

        # 注意这里用了ModuLelist函数，模块化列表
        self.cell_list = nn.ModuleList(cell_list)


    # 这里forward有两个输入参数，input_tensor 是一个4维数据
    # （t时间步,b输入batch_ize,c肌电通道特征维度12,w肌电序列时间窗长200）
    # hidden_state=None 默认刚输入hidden_state为空，等着后面给初始化
    def forward(self, input_tensor, hidden_state=None):
        # 先调整一下输出数据的排列
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3)
        # 取出序列的数据，供下面初始化使用，这里的seq是序列的长度
        b, _, _, seq = input_tensor.size()
        # 初始化hidd_state,利用后面和gru单元中的初始化函数
        # hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, seq=seq)

        # 储存输出数据的列表
        layer_output_list = []
        layer_state_list = []
        seq_len = input_tensor.size(1)

        # 初始化输入数据
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):   # 第几层

            h = hidden_state[layer_idx]
            output_inner = []

            for t in range(seq_len):
                # 每一个时间步都更新 h
                # 注意这里self.cell_list是一个模块(容器)
                h = self.cell_list[layer_idx](cur_layer_input[:, t, :, :],  h)     #有疑问   原本是 input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c]
                # 储存输出，注意这里 h 就是此时间步的输出
                output_inner.append(h)   # h.shape = [64,64,200,1]

            # 这一层的输出作为下一次层的输入,
            layer_output = torch.stack(output_inner, dim=1)   # 将维度扩充了 shape = [64,1,64,200,1]
            # 自己加的注意力机制
            # layer_output = self.attention(torch.squeeze(layer_output))
            # layer_output = torch.unsqueeze(layer_output, dim=1)
            # layer_output = torch.unsqueeze(layer_output, dim=4)


            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            # 储存每一层的状态h，c
            layer_state_list.append(h)

        # 选择要输出所有数据，还是输出最后一层的数据
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = layer_state_list[-1:]

        # 郭师兄在这里加了个线性层
        layer_output_list = self.cell_list[-1](self.cell_list[-2](layer_output_list[0]).
                                               reshape(b, seq_len, self.hidden_dim[-1], -1)).permute(0, 1, 3,2)

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, seq):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, seq))
        return init_states


class CONVGRUNet(nn.Module):
    def __init__(self,seq, input_dim, output_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=False,return_all_layers=False):
        super(CONVGRUNet, self).__init__()
        # self.c_attention = ChannelAttention(input_dim)   # 加的通道注意力
        self.convgru = ConvGru(seq,input_dim, hidden_dim, kernel_size, num_layers, batch_first=True,
                               bias=True,return_all_layers=False)

        self.fc = nn.Linear(12*200,256)   # self-attention 版本的
        self.fc1 = nn.Linear(256, output_dim)
        self.attention = AttentionBlock(12, 12, 12)

    def forward(self, inputs,hidden = None):
        # pdb.set_trace()
        inputs = inputs.unsqueeze(1).permute(0, 1, 3, 2)
        o1, h1 = self.convgru(inputs,hidden)
        # 原本
        # o1 = o1.reshape(o1.shape[0], o1.shape[1] * o1.shape[2] * o1.shape[3])
        # out = self.fc(o1)
        # 加self-attention的
        o1 = self.attention(h1[-1])
        # o1 = h1[-1]
        o1 = o1.reshape(o1.shape[0], o1.shape[1] * o1.shape[2] )
        out = self.fc(o1)
        out = self.fc1(out)
        return out


x = torch.rand([64,1,12,200])  # batch,t,featuredim,window_len(序列长度)
model = CONVGRUNet(200, 12, 12, [64, 32, 12], 11, 3, True)  #200窗长，12 输入维度，12 输出维度，隐藏层，11卷积核，3层数