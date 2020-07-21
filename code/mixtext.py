import torch
import torch.nn as nn
from pytransformers import *
from pytransformers.modeling_bert import BertEmbeddings, BertPooler, BertLayer

class BertModel4Mix(BertPreTrainedModel):
    def __init__(self, config):
        """
        Bert模型改造成Mix
        :param config:
        """
        super(BertModel4Mix, self).__init__(config)
        #做embedding
        self.embeddings = BertEmbeddings(config)
        #模型
        self.encoder = BertEncoder4Mix(config)
        #BertPooler 是一个全连接加一个tanh激活
        self.pooler = BertPooler(config)
        #初始化参数
        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(
            old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ 修剪模型head.
            heads_to_prune: 字典 {layer_num: 这一层要修剪的head的列表}
            具体查看 PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, input_ids2=None, lbeta=None, mix_layer=1000, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        """
        设置各种mask和input_ids 做embedding，然后调用BertEncoder4Mix 计算hidden_state
        :param input_ids: encode后的id
        :param input_ids2: 另一个要mix的 encode的id
        :param lbeta: beta分布
        :param mix_layer: 要做mix的层, 为了传递给BertEncoder4Mix
        :param attention_mask:  注意力的attention mask， 就是过滤掉padding的那些token
        :param token_type_ids: token 的类型id，就是属于哪个句子
        :param position_ids: transformer的position_id, 表示token的相对或者绝对位置等
        :param head_mask: 对head做的mask
        :return:
        """
        #创建attention_mask， [batch_size, sequence_length]
        if attention_mask is None:
            #如果input_ids2存在，使用input_ids2建立attention_mask， 1表示所有token都计算attention
            if input_ids2 is not None:
                attention_mask2 = torch.ones_like(input_ids2)
            attention_mask = torch.ones_like(input_ids)
        #设置token_type_ids
        if token_type_ids is None:
            # 0表示所有token都是属于同一个句子
            token_type_ids = torch.zeros_like(input_ids)
            if input_ids2 is not None:
                token_type_ids2 = torch.zeros_like(input_ids2)
        # attention_mask形状变成 [1,1,batch_size, sequence_length]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # 兼容fp16
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        # 不明白这里做什么, extended_attention_mask变成了00
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if input_ids2 is not None:
            #与input_ids 同样的操作， extended_attention_mask2变成了很小的值, extended_attention_mask变成了0
            extended_attention_mask2 = attention_mask2.unsqueeze(1).unsqueeze(2)
            extended_attention_mask2 = extended_attention_mask2.to(dtype=next(self.parameters()).dtype)
            extended_attention_mask2 = (1.0 - extended_attention_mask2) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                #我们可以为每层指定head mask
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            # 转换成浮点数如果需要兼容f16
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            #head_mask,  [None, None, None, None, None, None, None, None, None, None, None, None]
            head_mask = [None] * self.config.num_hidden_layers
        #id转换成词向量embedding_output [batch_size, seq_len, embedding_dimension]
        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        #如果input_ids2存在，也做embedding
        if input_ids2 is not None:
            embedding_output2 = self.embeddings(
                input_ids2, position_ids=position_ids, token_type_ids=token_type_ids2)
        #把输入1和输入2的调用BertEncoder4Mix encdoer到一起
        if input_ids2 is not None:
            encoder_outputs = self.encoder(embedding_output, embedding_output2, lbeta, mix_layer,
                                           extended_attention_mask, extended_attention_mask2, head_mask=head_mask)
        else:
            #调用BertEncoder4Mix,
            encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask)

        sequence_output = encoder_outputs[0]
        #序列的第一个标记（分类标记）的最后一层隐藏状态由线性层和Tanh激活函数进一步处理。 在预训练期间，从下一个句子预测（分类）目标训练线性层权重。
        pooled_output = self.pooler(sequence_output)

        # 添加 hidden_states 和 attentions， 返回的格式 sequence_output, pooled_output, (hidden_states), (attentions)
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs


class BertEncoder4Mix(nn.Module):
    def __init__(self, config):
        """

        :param config:  Bert 模型的config， 默认是加载transformers的bert的config
        """
        super(BertEncoder4Mix, self).__init__()
        #是否输出attentions，默认为False
        self.output_attentions = config.output_attentions
        #是否输出隐藏状态，默认为False
        self.output_hidden_states = config.output_hidden_states
        #使用BertLayer配置模型, 组装模型
        self.layer = nn.ModuleList([BertLayer(config)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, hidden_states2=None, lbeta=None, mix_layer=1000, attention_mask=None, attention_mask2=None, head_mask=None):
        """
        BertEncoder4Mix, 真正的mix hidden states操作在这里
        :param hidden_states: 第一个输入的隐藏状态
        :param hidden_states2:第二个输入的隐藏状态
        :param lbeta: beta分布取的值
        :param mix_layer: 要mix的layer，例如这里是bert的第11层, 当不做mix时，设置默认mix_layer为1000，目的是为了下面循环不做mix
        :param attention_mask: 输入1的 attention_mask
        :param attention_mask2: 输入2的 attention_mask
        :param head_mask:
        :return:
        """
        #保存每一层的hidden states和attentions，都放入列表
        all_hidden_states = ()
        all_attentions = ()
        # 执行mix，论文上的混合公式
        if mix_layer == -1:
            if hidden_states2 is not None:
                #论文上的混合公式，得到新的混合的hidden_states
                hidden_states = lbeta * hidden_states + (1 - lbeta) * hidden_states2

        for i, layer_module in enumerate(self.layer):
            #当当前层小于或等于mix_layer时，inputs1 和inputs2 分别计算hidden_states
            if i <= mix_layer:
                #是否输出隐藏状态，如果现在的layer小于要mix_layer,那么
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
                #调用transformers的BertLayer, 计算attention和hidden_states， 输出outputs
                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i])
                #获取这一次计算后的得到新的隐藏层状态
                hidden_states = layer_outputs[0]
                # 如果self.output_attentions 为True，输出attention，
                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)
                #如果输入2存在，也做同样计算
                if hidden_states2 is not None:
                    layer_outputs2 = layer_module(hidden_states2, attention_mask2, head_mask[i])
                    hidden_states2 = layer_outputs2[0]
            # 只有当循环到等于mix_layer时，使用mixup公式混合
            if i == mix_layer:
                if hidden_states2 is not None:
                    hidden_states = lbeta * hidden_states + (1 - lbeta) * hidden_states2
            # 循环到大于mix_layer的层时, 普通方式计算
            if i > mix_layer:
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

        #最后一层结束后的hidden_states
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        #outputs是最后一层的的hidden_states[batch_size,seq_len,Embedding_demision]
        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        # 最后一层的hidden state, (all hidden states), (all attentions)
        return outputs

# MixText 调用BertModel4Mix， BertModel4Mix调用BertEncoder4Mix
class MixText(nn.Module):
    def __init__(self, num_labels=2, mix_option=False, model='bert-base-chinese'):
        """
        配置Mix模型或Bert模型
        :param num_labels:标签个数，几分类
        :param mix_option:  是否使用MixText模型，还是普通Bert模型
        """
        super(MixText, self).__init__()
        if mix_option:
            self.bert = BertModel4Mix.from_pretrained(model)
        else:
            self.bert = BertModel.from_pretrained(model)

        #最后2层全连接，做分类, 输出标签个数的分类
        self.linear = nn.Sequential(nn.Linear(768, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, num_labels))

    def forward(self, x, x2=None, lbeta=None, mix_layer=1000):
        """
        前向网络，这个模型没做什么，只是调用BertModel4Mix
        :param x: 一个encode后的x
        :param x2: 另一个要做mix的x
        :param lbeta: beta分布的l
        :param mix_layer: 要混合的layer是哪个一个，例如bert的第11层
        :return:
        """
        if x2 is not None:
            all_hidden, pooler = self.bert(x, x2, lbeta, mix_layer)

            pooled_output = torch.mean(all_hidden, 1)

        else:
            #调用真正的模型获取结果
            all_hidden, pooler = self.bert(x)
            # [batch_size,seq_len, embedding_dim], 去掉seq_len, all_hidden 的形状 torch.Size([1, 256, 768]), pooled_out形状 torch.Size([1, 768])
            pooled_output = torch.mean(all_hidden, 1)
        #通过线性模型预测, pooled_output[batch_size, embedding_dim], 维度变成[batch_size, num_labels]
        predict = self.linear(pooled_output)
        #返回预测值
        return predict

