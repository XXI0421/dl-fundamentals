"""
Full definition of a GPT Language Model, all of it in this single file.
GPT语言模型的完整定义，全部在这个单文件中。

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math  # 导入数学库，用于sqrt等计算

import torch  # 导入PyTorch核心库
import torch.nn as nn  # 导入神经网络模块
from torch.nn import functional as F  # 导入函数式API（softmax等）

from mingpt.utils import CfgNode as CN  # 从utils导入配置节点类（类似YACS的配置系统）

# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    实现Google BERT仓库中当前使用的GELU激活函数（与OpenAI GPT相同）。
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        # 应用GELU激活函数：0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        # 相比ReLU更平滑，在0附近有非零梯度，适合Transformer
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    一个标准的多头掩码自注意力层，最后带有一个投影。
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    这里本可以使用torch.nn.MultiheadAttention，但我包含了一个显式实现以展示这并不复杂。
    """
    
    # ====================【关键解剖点1: 与Week 2的差异】====================
    # Week 2实现：手动定义W_q, W_k, W_v三个独立矩阵
    # minGPT优化：使用单个Linear(3*n_embd)一次性计算Q,K,V，效率更高，利用GPU并行
    
    def __init__(self, config):
        super().__init__()  # 调用父类构造函数
        assert config.n_embd % config.n_head == 0  # 断言：嵌入维度必须能被头数整除，确保可分
        
        # key, query, value projections for all heads, but in a batch
        # 为所有头计算键、查询、值的投影，但使用批量方式合并为一次矩阵乘法
        # (B, T, C) -> (B, T, 3*C) 然后拆分为Q,K,V
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)  # 合并的QKV投影层，参数量3*C^2
        
        # output projection
        # 输出投影：将注意力结果映射回嵌入空间
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)  # 输出投影层，形状(C, C)
        
        # regularization
        # 正则化：注意力dropout（应用于注意力权重）和残差dropout（应用于输出）
        self.attn_dropout = nn.Dropout(config.attn_pdrop)  # 注意力权重dropout，防止过拟合
        self.resid_dropout = nn.Dropout(config.resid_pdrop)  # 残差连接dropout
        
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # 因果掩码：确保注意力只应用于输入序列的左侧（未来信息不可见）
        # ====================【关键解剖点2: 固定因果掩码】====================
        # 注意：这是固定大小的掩码(block_size, block_size)！
        # 如果要支持KV-Cache，需要动态扩展此掩码，因为生成长度可能超过block_size
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))  # (1, 1, block_size, block_size)
        # register_buffer将掩码注册为持久化但不可训练的缓冲区，随模型保存但不参与梯度计算
        
        self.n_head = config.n_head  # 保存头数（通常为12）
        self.n_embd = config.n_embd  # 保存嵌入维度（通常为768）

    def forward(self, x):
        B, T, C = x.size()  # 解包输入尺寸：批次大小B，序列长度T，嵌入维度C（n_embd）

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # 为所有头批量计算查询、键、值，并将头维度移到批次维度
        # 步骤1：通过线性层一次性计算合并的QKV，然后分割
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)  # (B, T, 3*C) -> 3个(B, T, C)
        
        # 步骤2：重塑为多头形式：(B, T, C) -> (B, T, n_head, C//n_head) -> (B, n_head, T, head_size)
        # Week 2差异：Week 2可能使用view+permute，这里是view+transpose，效果相同但效率略有差异
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, head_size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, head_size)
        # 现在每个头独立处理：批次中每个样本的每个头都有(T, head_size)的矩阵

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # 因果自注意力计算：Q @ K^T得到注意力分数矩阵
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, n_head, T, T)，缩放因子1/sqrt(head_size)
        
        # 应用因果掩码：将未来位置（上三角）的注意力分数设为负无穷，softmax后变为0
        # 注意：这里使用self.bias[:,:,:T,:T]动态切片，支持长度T小于block_size的情况
        # ====================【KV-Cache思考点】====================
        # 如果使用KV-Cache，T会逐渐增长（1, 2, 3...），每次都需要重新计算整个注意力矩阵
        # 优化方案：Cache中存储历史K,V，每次只计算当前位置的attn分数，避免重复计算O(T^2)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))  # 将掩码为0的位置设为-inf
        
        att = F.softmax(att, dim=-1)  # 在最后一个维度（T）上应用softmax，得到概率分布，每行和为1
        att = self.attn_dropout(att)  # 对注意力权重应用dropout（随机置零部分权重，防止过拟合）
        
        y = att @ v  # 注意力加权求和：(B, n_head, T, T) x (B, n_head, T, head_size) -> (B, n_head, T, head_size)
        # 每个位置t的注意力输出是所有历史位置的V的加权平均
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # 重组所有头的输出：(B, T, n_head, head_size) -> (B, T, C)
        # contiguous()确保内存连续，view才能正确执行，避免错误
        
        # output projection
        # 输出投影并应用dropout
        y = self.resid_dropout(self.c_proj(y))  # (B, T, C) -> (B, T, C)，残差dropout在投影后应用
        return y  # 返回注意力输出，形状与输入x相同(B, T, C)，可直接用于残差连接

class Block(nn.Module):
    """ an unassuming Transformer block """
    """ 一个不起眼的Transformer块（但实际蕴含关键设计决策） """

    def __init__(self, config):
        super().__init__()  # 调用父类构造函数
        
        # ====================【关键解剖点3: Pre-Norm架构】====================
        # Week 2/2017 Transformer原始论文使用Post-Norm：x = LN(x + Sublayer(x))
        # minGPT/Llama使用Pre-Norm：x = x + Sublayer(LN(x))
        # 优势：梯度流更稳定，训练更深网络时避免梯度消失/爆炸，Llama沿用此设计
        
        self.ln_1 = nn.LayerNorm(config.n_embd)  # 第一层归一化：在注意力前应用（Pre-Norm）
        self.attn = CausalSelfAttention(config)  # 因果自注意力子层
        
        self.ln_2 = nn.LayerNorm(config.n_embd)  # 第二层归一化：在MLP前应用（Pre-Norm）
        
        # MLP使用ModuleDict组织，更清晰但功能等效于Sequential
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),  # 升维投影：C -> 4*C（GPT-2标准扩展4倍）
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),  # 降维投影：4*C -> C
            act     = NewGELU(),  # GELU激活函数（Week 2可能使用ReLU或GELU近似）
            dropout = nn.Dropout(config.resid_pdrop),  # 残差dropout
        ))
        m = self.mlp  # 别名，方便lambda使用
        # 使用lambda定义MLP前向传播：fc -> act -> proj -> dropout
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))  # x形状保持(B, T, C)

    def forward(self, x):
        # ====================【Pre-Norm vs Post-Norm对比】====================
        # Post-Norm（2017 Transformer）：x = LN(x + Attention(x))  # 梯度流经过LN可能衰减
        # Pre-Norm（minGPT/Llama）：    x = x + Attention(LN(x))   # 残差连接直接传递梯度，更清晰
        
        # 步骤1：先对x进行LayerNorm，然后通过注意力，最后残差连接
        x = x + self.attn(self.ln_1(x))  # (B, T, C) -> LN -> Attention -> 残差相加
        
        # 步骤2：先对x进行LayerNorm，然后通过MLP，最后残差连接
        x = x + self.mlpf(self.ln_2(x))  # (B, T, C) -> LN -> MLP -> 残差相加
        return x  # 返回处理后的张量，形状(B, T, C)

class GPT(nn.Module):
    """ GPT Language Model """
    """ GPT语言模型完整实现 """

    @staticmethod
    def get_default_config():
        # 静态方法：返回默认配置
        C = CN()  # 创建配置节点
        
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        # 配置中必须提供model_type或(n_layer, n_head, n_embd)之一
        C.model_type = 'gpt'  # 模型类型标识
        C.n_layer = None  # Transformer层数（None表示需要从model_type推断）
        C.n_head = None  # 注意力头数
        C.n_embd = None  # 嵌入维度（隐藏层大小）
        
        # these options must be filled in externally
        # 这些选项必须从外部填充
        C.vocab_size = None  # 词表大小（如50257 for GPT-2）
        C.block_size = None  # 最大序列长度（如1024 for GPT-2）
        
        # dropout hyperparameters
        # Dropout超参数（正则化强度）
        C.embd_pdrop = 0.1  # 嵌入dropout率
        C.resid_pdrop = 0.1  # 残差dropout率
        C.attn_pdrop = 0.1  # 注意力dropout率
        return C  # 返回默认配置对象

    def __init__(self, config):
        super().__init__()  # 调用父类构造函数
        
        assert config.vocab_size is not None  # 断言：词表大小必须指定
        assert config.block_size is not None  # 断言：块大小（序列长度）必须指定
        self.block_size = config.block_size  # 保存块大小到实例

        # 检查配置：要么提供model_type，要么提供详细的层数/头数/维度参数（异或关系）
        type_given = config.model_type is not None  # 是否提供了模型类型
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])  # 是否提供了详细参数
        assert type_given ^ params_given  # 异或检查：必须且只能提供一种配置方式（XOR：一个为真一个为假）

        if type_given:
            # translate from model_type to detailed configuration
            # 从模型类型转换为详细配置参数
            config.merge_from_dict({
                # names follow the huggingface naming conventions
                # 命名遵循huggingface的命名约定
                # GPT-1
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M参数
                # GPT-2 configs
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M参数
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M参数
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M参数
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M参数（1.5B）
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # I made these tiny models up
                # 以下是karpathy自定义的微型模型，用于实验
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])  # 根据model_type选择对应配置并合并

        # 定义Transformer主模块字典（有序结构）
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # 词嵌入矩阵（Word Token Embedding）：(V, C)
            wpe = nn.Embedding(config.block_size, config.n_embd),  # 位置嵌入矩阵（Word Position Embedding）：(block_size, C)
            # ====================【关键解剖点4: 绝对位置编码】====================
            # minGPT使用可学习的绝对位置嵌入（与Week 2一致，与Transformer原始论文一致）
            # Day 4任务：替换为RoPE相对位置编码，消除长度限制，改善外推性
            drop = nn.Dropout(config.embd_pdrop),  # 嵌入dropout层
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # Transformer块堆栈（n_layer个Block）
            ln_f = nn.LayerNorm(config.n_embd),  # 最终层归一化（Final LayerNorm），在最后一层Block后应用
        ))
        
        # 语言模型头（Language Model Head）：将隐藏状态映射到词表分布
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # (C, V)，无偏置
        
        # ====================【Week 2差异：权重绑定说明】====================
        # Week 2可能使用独立的输出嵌入
        # minGPT可选择是否与输入嵌入wte共享权重（这里未显式绑定，但GPT-2通常绑定）
        # 注意：lm_head.weight与wte.weight形状相同(C, V)，可以共享以减少参数量

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        # 初始化所有权重，并对残差投影应用特殊的缩放初始化（根据GPT-2论文）
        self.apply(self._init_weights)  # 递归应用初始化到所有子模块
        
        # 对残差投影层（c_proj）应用特殊初始化：标准差按sqrt(2*n_layer)缩放
        # 原理：深层网络中残差路径的累积方差需要控制，避免梯度消失/爆炸
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):  # 所有Block中的注意力输出投影和MLP输出投影
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))  # 缩放初始化

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        # 报告参数数量（注意：不计入lm_head的参数，因为通常与wte共享权重）
        n_params = sum(p.numel() for p in self.transformer.parameters())  # 只计算transformer部分的参数量
        print("number of parameters: %.2fM" % (n_params/1e6,))  # 打印百万级别的参数量

    def _init_weights(self, module):
        # 自定义权重初始化函数
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # Linear层：正态分布初始化，std=0.02
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  # 偏置初始化为0
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # 嵌入层：同样正态分布初始化
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)  # LayerNorm偏置初始化为0
            torch.nn.init.ones_(module.weight)  # LayerNorm权重初始化为1（单位变换起点）

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        通过从huggingface/transformers检查点复制权重来初始化预训练GPT模型。
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}  # 仅支持这些模型类型
        from transformers import GPT2LMHeadModel  # 动态导入transformers库

        # create a from-scratch initialized minGPT model
        # 创建一个从头初始化的minGPT模型
        config = cls.get_default_config()  # 获取默认配置
        config.model_type = model_type  # 设置模型类型
        config.vocab_size = 50257  # openai's model vocabulary：GPT-2 BPE词表大小
        config.block_size = 1024   # openai's model block_size：最大序列长度
        model = GPT(config)  # 实例化模型
        sd = model.state_dict()  # 获取minGPT的状态字典（参数字典）

        # init a huggingface/transformers model
        # 初始化huggingface/transformers模型
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)  # 从HF下载并加载预训练权重
        sd_hf = model_hf.state_dict()  # 获取HF模型的状态字典

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        # 复制权重，同时确保所有参数在名称和形状上对齐匹配
        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')]  # 忽略HF中特定的masked_bias（minGPT不需要）
        
        # 需要转置的权重列表（因为OpenAI使用Conv1D而minGPT使用Linear，形状转置后匹配）
        # ====================【关键解剖点5: Conv1D vs Linear】====================
        # OpenAI原始实现使用Conv1D(1x1卷积)实现全连接，形状为(n_out, n_in)
        # PyTorch Linear形状为(n_in, n_out)，因此需要转置(.t())
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla nn.Linear.
        # this means that we have to transpose these weights when we import them
        # OpenAI检查点使用"Conv1D"模块，但我们只想使用标准nn.Linear，因此导入时需要转置这些权重
        assert len(keys) == len(sd)  # 断言：键的数量应匹配
        
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                # 对Conv1D权重特殊处理：需要转置
                assert sd_hf[k].shape[::-1] == sd[k].shape  # 断言：转置后形状应匹配
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())  # 转置并复制权重
            else:
                # vanilla copy over the other parameters
                # 普通参数直接复制
                assert sd_hf[k].shape == sd[k].shape  # 断言：形状必须完全匹配
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])  # 直接复制

        return model  # 返回加载了预训练权重的模型

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        这个长函数实际上在做一件简单的事，只是非常谨慎：
        我们将模型所有参数分为两类：一类应用权重衰减（正则化），另一类不应用（偏置、LayerNorm/嵌入权重）。
        然后返回PyTorch优化器对象。
        
        ====================【Week 2差异：权重衰减策略】====================
        优化器配置：区分weight decay和no weight decay参数组
        原理：偏置和归一化层/嵌入层权重通常不应用L2正则化（weight decay），避免破坏 scale invariance
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        # 将所有参数分为两类：应用权重衰减的和不应用的
        decay = set()  # 应用权重衰减的参数名集合（通常是Linear的weight）
        no_decay = set()  # 不应用权重衰减的参数名集合（bias, LayerNorm, Embedding）
        
        whitelist_weight_modules = (torch.nn.Linear, )  # 白名单：这些模块的weight应用weight decay
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)  # 黑名单：这些模块的weight不应用weight decay

        # 遍历所有模块和参数进行分类
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # 完整参数名（如果模块名为空则直接用参数名）
                
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                # 注意：named_modules和named_parameters是递归的，我们会多次看到同一个张量p
                # 但这样做的好处是知道每个张量p属于哪个父模块...

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    # 所有偏置都不进行权重衰减（优化稳定）
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    # 白名单模块的权重进行权重衰减（Linear层的weight）
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    # 黑名单模块的权重不进行权重衰减（LayerNorm和Embedding的weight）
                    no_decay.add(fpn)

        # validate that we considered every parameter
        # 验证：确保每个参数都被分类（没有遗漏，也没有重复）
        param_dict = {pn: p for pn, p in self.named_parameters()}  # 创建参名字典
        inter_params = decay & no_decay  # 交集：同时在两个集合中的参数（应无）
        union_params = decay | no_decay  # 并集：被分类的所有参数
        
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        # 断言：交集为空，没有参数同时在两个集合中
        
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        # 断言：所有参数都在并集中（没有未分类的参数）

        # create the pytorch optimizer object
        # 创建PyTorch优化器对象（AdamW）
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},  # 应用衰减组
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},  # 不衰减组
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        # 使用AdamW优化器（Adam with decoupled weight decay），支持不同的weight decay设置
        return optimizer

    def forward(self, idx, targets=None):
        # 前向传播函数
        # idx: 输入token索引，形状(b, t)，targets: 目标token索引（用于训练），形状(b, t)
        
        device = idx.device  # 获取输入所在设备（CPU/GPU）
        b, t = idx.size()  # 解包批次大小b和序列长度t
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        # 断言：输入序列长度不能超过模型最大块大小
        
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # 生成位置索引：(1, t)

        # forward the GPT model itself
        # GPT模型主前向传播流程
        
        tok_emb = self.transformer.wte(idx)  # 词嵌入查找：(b, t) -> (b, t, n_embd)，从词表获取词向量
        pos_emb = self.transformer.wpe(pos)  # 位置嵌入查找：(1, t) -> (1, t, n_embd)，获取位置编码
        # ====================【Week 2差异：绝对位置编码】====================
        # 这里使用可学习的绝对位置编码，需要预先知道最大长度(block_size)
        # Day 4将替换为RoPE：通过旋转位置编码动态计算，支持更长序列，改善长度外推
        
        x = self.transformer.drop(tok_emb + pos_emb)  # 词嵌入+位置嵌入后应用dropout，(b, t, n_embd)
        # 注意：广播机制使(1, t, C)的pos_emb加到(b, t, C)的tok_emb上

        for block in self.transformer.h:
            x = block(x)  # 逐个通过Transformer Block：(b, t, n_embd) -> ... -> (b, t, n_embd)
        
        x = self.transformer.ln_f(x)  # 最终层归一化：(b, t, n_embd)，稳定输出分布
        
        logits = self.lm_head(x)  # 语言模型头：(b, t, n_embd) -> (b, t, vocab_size)，每个位置预测下一个token的分数

        # if we are given some desired targets also calculate the loss
        # 如果提供了目标序列（训练模式），计算交叉熵损失
        loss = None
        if targets is not None:
            # 将logits展平为(b*t, vocab_size)，targets展平为(b*t,)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            # ignore_index=-1：忽略填充token（如果有的话）

        return logits, loss  # 返回logits（推理用）和loss（训练用）

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None, top_p=None, temperature_decay=1.0):
        """

        增强版生成函数：支持Nucleus Sampling(Top-p)与动态温度衰减
        
        参数：
            idx: 输入序列 (B, T)
            max_new_tokens: 生成token数
            temperature: 初始温度 (>1更随机，<1更保守)
            do_sample: 是否采样（False=贪婪）
            top_k: 只从概率前k个中采样（None=全部）
            top_p: Nucleus采样阈值（0.9=从累积概率90%的核内采样，None=不启用）
            temperature_decay: 温度衰减因子（每步乘以该值，<1.0逐渐降温）
        
        示例：
            # 创意开头，确定结尾的故事生成
            model.generate(idx, 100, temperature=1.2, top_p=0.9, temperature_decay=0.98)

        
        ====================【关键解剖点6: 生成逻辑与KV-Cache改造点】====================
        当前实现：每次生成新token时，重新计算整个序列的attention（包括历史token）
        复杂度：O(T^2)每步，总O(T^3)，生成慢
        
        Day 5改造方案：
        1. 实现KV-Cache类，缓存历史K,V矩阵
        2. 每步只计算当前位置的Q，与缓存的K,V做attention（O(T)每步）
        3. 需要修改CausalSelfAttention支持past_key_values参数
        4. 修改此处循环逻辑，传入cache并更新
        """
        for step in range(max_new_tokens):  # 自回归循环，逐个生成token
            
            # if the sequence context is growing too long we must crop it at block_size
            # 如果序列长度超过block_size，必须裁剪（只保留最后block_size个token）
            # ====================【KV-Cache关联：裁剪破坏Cache连续性】====================
            # 问题：如果使用KV-Cache，裁剪idx会导致缓存与输入错位
            # 解决：使用KV-Cache后，idx可以保持只传最新的token，历史通过Cache传递
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # forward the model to get the logits for the index in the sequence
            # 前向传播获取序列当前步的logits（只关心最后一个位置的输出）
            logits, _ = self(idx_cond)  # (b, t, vocab_size)，t <= block_size
            
            # pluck the logits at the final step and scale by desired temperature
            # 取最后一个时间步的logits，并应用温度缩放（控制随机性）
            logits = logits[:, -1, :]   # (b, vocab_size)，temperature>1更随机，<1更确定

            # ====================【Day 3: 温度动态衰减】====================
            # 计算当前步的有效温度（逐渐降低，使结尾更确定）
            current_temp = temperature * (temperature_decay ** step)
            # 例如：step=0, temp=1.2; step=50, temp=1.2*(0.98^50)≈0.44（趋于确定）
            min_temp = 0.1  # 温度下限（不要低于0.1，否则数值不稳定）
            current_temp = max(current_temp, min_temp)
            logits = logits / current_temp

            # ====================【Day 3: Top-p Nucleus采样】====================
            # 与Top-k互斥（通常只用其一，或先用Top-k裁剪再用Top-p）
            # optionally crop the logits to only the top k options
            # 可选：Top-K采样，只保留概率最高的k个选项，其余设为负无穷
            if top_p is not None and 0 < top_p < 1 :
                # 1. 计算概率分布
                probs = F.softmax(logits, dim=-1)

                # 2. 按概率降序排序
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)  # (B, vocab_size)

                # 3. 计算累积概率
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)  # (B, vocab_size)

                # 4. 找到累积概率超过top_p的位置（核外）
                # 例如top_p=0.9，保留累积90%概率的tokens，其余mask
                sorted_indices_to_remove = cumsum_probs > top_p  # (B, vocab_size) bool矩阵   

                # 5. 关键技巧：至少保留第一个token（概率最高的）
                # 将第一个位置的移除标记设为False（保留）
                sorted_indices_to_remove[..., 0] = False
                
                # 6. 映射回原始索引位置
                # 创建与sorted_indices_to_remove相同shape的原始索引张量
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                
                # 7. 将这些位置的logits设为-inf（softmax后概率为0）
                logits[0, indices_to_remove] = float('-inf')
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
        
            # 采样或贪婪
            probs = F.softmax(logits, dim=-1)
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            
            idx = torch.cat((idx, idx_next), dim=1)
    
        return idx
