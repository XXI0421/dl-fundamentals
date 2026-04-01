"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
简单训练循环；可适用于任何神经网络的样板代码，因此这个文件中的内容与GPT没有特定关联。

====================【Day 2解剖重点】====================
1. 基础训练循环：随机采样 + 梯度裁剪
2. 缺失功能：学习率调度（warmup+cosine）、验证集监控、早停机制
3. 改造目标：添加Validation Loop + Early Stopping + Checkpoint Saving
4. Day 4衔接：梯度裁剪对RMSNorm至关重要（需调低阈值至0.5-1.0）
=====================================================
"""

import time  # 导入时间模块，用于计算迭代耗时
from collections import defaultdict  # 导入默认字典，用于管理回调函数字典
import math

import torch  # PyTorch核心库
from torch.utils.data.dataloader import DataLoader  # 数据加载器
from mingpt.utils import CfgNode as CN  # 配置节点类（类似YACS的配置系统）

class Trainer:
    """
    训练器类：封装训练循环、优化器配置和回调机制
    
    关键局限（Day 2改造点）：
    - 当前只有训练集，没有验证集监控（无法检测过拟合）
    - 没有学习率调度器（LR Scheduling），固定3e-4
    - 没有模型检查点保存（Checkpointing）
    - 没有早停机制（Early Stopping）
    """

    @staticmethod
    def get_default_config():
        """获取默认训练配置"""
        C = CN()
        # device to train on
        # 训练设备：'auto'自动检测，或指定'cuda'/'cpu'
        C.device = 'auto'
        # dataloder parameters
        # 数据加载器参数
        C.num_workers = 4  # 数据加载的并行工作进程数（CPU密集型，多进程加速）
        # optimizer parameters
        # 优化器参数（注意：缺少学习率调度相关配置，这是主要缺失项）
        C.max_iters = None  # 最大迭代次数（None表示无限循环，需外部强制停止）
        C.batch_size = 64  # 批次大小（每步处理的样本数）
        C.learning_rate = 3e-4  # 初始学习率（固定值，没有warmup和cosine decay）
        C.betas = (0.9, 0.95)  # Adam的beta参数（一阶/二阶动量系数，0.95减少早期梯度权重）
        C.weight_decay = 0.1  # 权重衰减系数（只应用于矩阵乘法权重，偏置和LN不应用）
        C.grad_norm_clip = 1.0  # 梯度裁剪阈值（防止梯度爆炸，Llama架构需特别关注）
        
        # ====================【Day 2添加配置】====================
        # 以下配置项在EnhancedTrainer中添加：
        # C.lr_scheduler = 'cosine'  # 学习率调度策略：'cosine'或'none'
        # C.warmup_iters = 2000  # 预热步数（线性增长lr到峰值）
        # C.min_lr = 1e-5  # 最小学习率（cosine decay下限）
        # C.val_interval = 1000  # 验证间隔（每1000步验证一次）
        # C.patience = 5  # 早停耐心值（验证损失不下降5次则停止）
        # C.checkpoint_dir = './checkpoints'  # 检查点保存目录
        return C

    def __init__(self, config, model, train_dataset):
        """
        初始化训练器
        
        参数：
            config: 训练配置对象（CfgNode）
            model: GPT模型实例（nn.Module）
            train_dataset: 训练数据集（torch.utils.data.Dataset）
        
        注意：缺少val_dataset参数，需在EnhancedTrainer中添加
        """
        self.config = config  # 保存配置对象引用
        self.model = model  # 保存模型引用
        self.optimizer = None  # 优化器（延迟初始化，由model.configure_optimizers创建）
        self.train_dataset = train_dataset  # 训练数据集（随机采样）
        # ====================【缺失：验证集】====================
        # Day 2改造：添加 self.val_dataset = None
        self.callbacks = defaultdict(list)  # 回调函数字典：事件名 -> 回调函数列表

        # determine the device we'll train on
        # 确定训练设备（GPU优先）
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 自动检测CUDA可用性
        else:
            self.device = config.device  # 使用用户指定设备
        self.model = self.model.to(self.device)  # 将模型参数和缓冲区移至目标设备
        print("running on device", self.device)  # 打印设备信息（确认训练硬件）

        # variables that will be assigned to trainer class later for logging and etc
        # 这些变量将在训练过程中动态赋值，用于日志记录和状态跟踪
        self.iter_num = 0  # 当前迭代次数（全局步数，从0开始）
        self.iter_time = 0.0  # 上次迭代的时间戳（秒级时间）
        self.iter_dt = 0.0  # 单次迭代耗时（秒），用于计算吞吐量（tokens/sec）
        
        # ====================【Day 2添加状态变量】====================
        # 在EnhancedTrainer中初始化：
        # self.best_val_loss = float('inf')  # 最佳验证损失（用于保存最佳模型）
        # self.steps_since_improvement = 0  # 自上次改善以来的验证次数（早停计数器）
        # self.lr_scheduler = None  # 学习率调度器状态

    def add_callback(self, onevent: str, callback):
        """
        添加回调函数到指定事件
        支持事件：'on_batch_end', 'on_epoch_end'等（自定义扩展）
        """
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        """
        设置（覆盖）指定事件的回调函数列表
        与add_callback不同，这会清除该事件的已有回调
        """
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        """
        触发指定事件的所有回调函数
        将self（trainer实例）作为参数传递给每个回调，允许回调访问训练状态
        """
        for callback in self.callbacks.get(onevent, []):
            callback(self)  # 执行回调函数，传入trainer实例供其查询/修改状态

    def run(self):
        """
        主训练循环（原版实现）
        
        核心流程：
        1. 初始化优化器（AdamW + 权重衰减分组）
        2. 创建无限数据加载器（RandomSampler replacement=True）
        3. 无限循环直到max_iters：
           - 获取批次 -> 设备转移
           - 前向传播（计算logits和loss）
           - 反向传播（loss.backward()）
           - 梯度裁剪（防止爆炸，对RMSNorm尤其重要）
           - 优化器步骤（更新权重）
           - 触发回调（日志记录）
        4. 终止条件检查
        
        ====================【关键缺失分析】====================
        当前实现缺陷：
        1. 没有验证循环：无法监控泛化性能，无法检测过拟合
        2. 没有学习率调整：固定LR 3e-4，缺少warmup和cosine decay
        3. 没有检查点保存：训练中断丢失进度，无法恢复最佳模型
        4. 随机采样器：replacement=True可能导致采样不均（虽然概率低）
        =====================================================
        """
        model, config = self.model, self.config  # 解包以便快速访问

        # setup the optimizer
        # 设置优化器（调用模型类的configure_optimizers方法，实现权重衰减分组）
        self.optimizer = model.configure_optimizers(config)
        # optimizer包含两个参数组：
        # - 组1：应用weight decay的参数（Linear的weight）
        # - 组2：不应用weight decay的参数（bias, LayerNorm/Embedding的weight）

        # setup the dataloader
        # 设置训练数据加载器（关键配置分析）
        train_loader = DataLoader(
            self.train_dataset,
            # RandomSampler with replacement: 有放回的随机采样
            # num_samples=int(1e10) 表示理论上的"无限"采样（实际受限于max_iters）
            # 潜在问题：某些样本可能被重复采样多次，某些可能长期不被采样（虽然概率低）
            # 优势：不需要知道数据集总大小，天然支持无限训练
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,  # 当使用自定义sampler时，shuffle必须为False（互斥）
            pin_memory=True,  # 将数据固定在CUDA pinned memory中，加速主机到设备的传输（仅GPU有效）
            batch_size=config.batch_size,  # 每步处理的序列数（受GPU显存限制）
            num_workers=config.num_workers,  # 数据加载的并行进程数（CPU密集型预处理）
        )

        model.train()  # 设置模型为训练模式（启用Dropout、LayerNorm使用batch统计量）
        self.iter_num = 0  # 初始化全局步数计数器
        self.iter_time = time.time()  # 记录开始时间戳
        data_iter = iter(train_loader)  # 创建数据迭代器（Python迭代器协议）
        
        # ====================【Day 2改造：应添加验证集加载器】====================
        # if self.val_dataset is not None:
        #     val_loader = DataLoader(self.val_dataset, batch_size=config.batch_size, 
        #                            shuffle=False, num_workers=0)  # 验证不打乱，单进程即可

        while True:  # 无限训练循环（直到触发break条件）
            
            # fetch the next batch (x, y) and re-init iterator if needed
            # 获取下一个批次(x, y)，如果迭代器耗尽则重新初始化
            try:
                batch = next(data_iter)  # 尝试从迭代器获取下一批次（x, y元组）
            except StopIteration:
                # 由于RandomSampler设置了num_samples=1e10，理论上不应触发StopIteration
                # 但如果修改了sampler或数据集被修改，则安全地重新创建迭代器
                data_iter = iter(train_loader)
                batch = next(data_iter)
            
            # move batch to device (GPU/CPU)
            # 将批次数据从CPU内存（Dataset）转移到计算设备（GPU显存或CPU内存）
            batch = [t.to(self.device) for t in batch]  # 列表推导式转移每个张量
            x, y = batch  # 解包为输入x（当前token）和目标y（下一个token）
            # x shape: (batch_size, block_size) - 输入序列的token索引
            # y shape: (batch_size, block_size) - 目标序列的token索引（x的左移一位）

            # forward the model
            # 前向传播计算损失
            logits, self.loss = model(x, y)  # 返回logits（预测分数）和cross_entropy损失
            # logits shape: (batch_size, block_size, vocab_size)
            # loss shape: scalar tensor（标量，平均交叉熵）

            # backprop and update the parameters
            # 反向传播并更新参数（标准训练流程）
            model.zero_grad(set_to_none=True)  # 梯度清零（set_to_none比赋0更省内存，释放梯度张量）
            self.loss.backward()  # 反向传播：计算所有可学习参数的梯度（链式法则）
            
            # ====================【关键稳定措施：梯度裁剪】====================
            # 对全局梯度范数进行裁剪，防止梯度爆炸（RMSNorm架构尤其需要）
            # Day 4注意：当使用RMSNorm替代LayerNorm时，建议将grad_norm_clip从1.0调至0.5
            # 原理：RMSNorm缺少去均值操作，梯度可能比LayerNorm大1.5-2倍
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            
            self.optimizer.step()  # 优化器步骤：基于梯度更新所有参数（AdamW：动量+权重衰减）

            self.trigger_callbacks('on_batch_end')  # 触发批次结束回调（如日志记录、指标追踪）
            
            # timing and logging
            # 计时和状态更新
            self.iter_num += 1  # 全局步数递增（从0开始，第一次递增后为1）
            tnow = time.time()  # 获取当前时间戳
            self.iter_dt = tnow - self.iter_time  # 计算本次迭代耗时（秒）
            self.iter_time = tnow  # 更新时间戳为当前时间

            # ====================【Day 2改造点：验证与早停逻辑插入位置】====================
            # 应在每config.val_interval次迭代后执行：
            # 1. model.eval() 切换到评估模式
            # 2. 遍历val_loader计算平均val_loss
            # 3. 如果val_loss < best_val_loss：保存checkpoint，reset patience
            # 4. 否则：patience--，如果<=0则break（早停）
            # 5. model.train() 切回训练模式
            
            # termination conditions
            # 终止条件检查（唯一退出点）
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break  # 达到最大迭代次数，安全退出循环
                # 注意：如果没有设置max_iters，此循环将无限运行（需手动Ctrl+C中断）
        
        # ====================【Day 2改造：训练结束处理】====================
        # 应添加：保存最终checkpoint，记录训练摘要（总时间、最终损失等）


# ====================【Day 2新增：增强版Trainer类】====================
class EnhancedTrainer(Trainer):
    """
    增强版训练器：添加验证监控、学习率调度（warmup+cosine）、早停机制
    
    可直接替换原Trainer使用，向后兼容（val_dataset为可选参数）
    
    Day 4衔接注意事项：
    - 当使用RMSNorm时，建议调低grad_norm_clip至0.5（RMSNorm梯度更大）
    - 增加warmup_iters比例至5%（RMSNorm初始阶段更不稳定）
    """
    
    @staticmethod
    def get_default_config():
        """获取增强版默认配置（扩展原配置）"""
        C = Trainer.get_default_config()
        # 学习率调度配置（对应Day 1中提到的warmup+cosine decay）
        C.lr_scheduler = 'cosine'  # 学习率调度类型：'cosine'或'none'
        C.warmup_iters = 2000  # 预热迭代次数（线性增长lr，防止初始梯度爆炸）
        C.max_iters = 10000  # 总迭代次数（用于cosine decay终点计算）
        C.min_lr = 1e-5  # 最小学习率（cosine decay下限，非零避免停滞）
        
        # 验证与早停配置
        C.val_interval = 1000  # 每1000次迭代验证一次（监控泛化性能）
        C.patience = 5  # 早停耐心值（验证损失不下降5次则停止，节省计算）
        C.checkpoint_dir = './checkpoints'  # 检查点保存目录（自动创建）
        C.save_best_only = True  # 是否只保存最佳模型（节省磁盘空间）
        
        # Day 4 RMSNorm适配（可选，提前配置）
        C.grad_norm_clip = 0.5  # 梯度范数裁剪，建议值：RMSNorm需更低阈值防梯度爆炸
        return C
    
    def __init__(self, config, model, train_dataset, val_dataset=None):
        """
        初始化增强版训练器
        
        参数：
            config: 训练配置对象（使用get_default_config()获取默认值）
            model: GPT模型实例
            train_dataset: 训练数据集（RandomSampler采样）
            val_dataset: 验证数据集（可选，用于监控泛化性能和早停）
        """
        super().__init__(config, model, train_dataset)
        self.val_dataset = val_dataset  # 保存验证集（可能为None）
        self.best_val_loss = float('inf')  # 初始化最佳验证损失为无穷大（跟踪最佳模型）
        self.steps_since_improvement = 0  # 早停计数器（连续未改善的验证次数）
        self.lr_scheduler = None  # 学习率调度器状态（本实现使用手动lr调整）
        
        # 创建检查点目录（如果不存在）
        import os
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
    def get_lr(self, it):
        """
        学习率调度：warmup + cosine decay
        对应Day 1中提到的学习率调度策略（GPT-2/Llama标准实践）
        
        参数：
            it: 当前迭代次数（0-indexed）
        返回：
            计算后的学习率值（标量float）
            
        调度策略：
        1. 0 -> warmup_iters: 线性增长（0.0 -> learning_rate）
        2. warmup_iters -> max_iters: 余弦退火（learning_rate -> min_lr）
        """
        config = self.config
        
        # 1) 线性warmup阶段（防止初始梯度爆炸，尤其对RMSNorm重要）
        if it < config.warmup_iters:
            # 线性插值：当前步/总warmup步 * 峰值学习率
            return config.learning_rate * it / config.warmup_iters
        
        # 2) cosine decay阶段（余弦退火，精细化收敛）
        # 计算归一化进度（0.0到1.0之间）
        progress = (it - config.warmup_iters) / max(1, config.max_iters - config.warmup_iters)
        
        # 余弦退火公式：min_lr + 0.5*(max_lr-min_lr)*(1+cos(π*progress))
        # 起点（progress=0）：min_lr + 0.5*(range)*2 = learning_rate
        # 终点（progress=1）：min_lr + 0.5*(range)*0 = min_lr
        return config.min_lr + 0.5 * (config.learning_rate - config.min_lr) * (1 + math.cos(math.pi * progress))
    
    def validate(self):
        """
        在验证集上评估模型性能
        
        流程：
        1. 切换到eval模式（禁用Dropout，LayerNorm使用全局统计量）
        2. 遍历验证数据集（无梯度计算）
        3. 计算平均交叉熵损失
        
        返回：
            平均验证损失（标量float），如果无验证集则返回None
        """
        if self.val_dataset is None:
            return None  # 无验证集，跳过验证
            
        self.model.eval()  # 切换到评估模式（关键：Dropout关闭，BN/LN使用running stats）
        
        # 创建验证数据加载器（不打乱，单进程，固定批次大小）
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,  # 验证集不打乱，保证可复现性
            num_workers=0,  # 验证通常不需要多进程，避免进程创建开销
            pin_memory=True if self.device == 'cuda' else False
        )
        
        total_loss = 0.0  # 累计损失
        total_batches = 0  # 批次计数（用于平均）
        total_tokens = 0  # Token计数（用于计算BPC或 perplexity）
        
        with torch.no_grad():  # 禁用梯度计算（节省显存和计算，加速验证）
            for batch in val_loader:
                # 设备转移
                batch = [t.to(self.device) for t in batch]
                x, y = batch
                
                # 前向传播（只获取loss，忽略logits）
                logits, loss, _ = self.model(x, y, use_cache=False)
                
                # 累计（使用.item()获取Python标量，避免GPU内存占用）
                total_loss += loss.item()
                total_batches += 1
                total_tokens += x.numel()  # 统计总token数（可选，用于计算困惑度）
        
        avg_loss = total_loss / max(total_batches, 1)  # 计算平均损失（防除0）
        
        # 计算困惑度（Perplexity，语言模型标准指标）：exp(cross_entropy)
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')  # 防溢出
        
        self.model.train()  # 切回训练模式（重新启用Dropout等）
        return avg_loss  # 返回平均损失（用于早停判断）
    
    def save_checkpoint(self, filename, is_best=False):
        """
        保存模型检查点（含优化器状态、训练状态）
        
        参数：
            filename: 保存文件名（如'checkpoint_step1000.pt'）
            is_best: 是否为最佳模型（如果是，同时保存为best_model.pt）
        """
        import os
        filepath = os.path.join(self.config.checkpoint_dir, filename)
        
        # 构建检查点字典（包含恢复训练所需的所有状态）
        checkpoint = {
            'iter_num': self.iter_num,  # 当前迭代次数（恢复训练时继续）
            'model_state_dict': self.model.state_dict(),  # 模型权重
            'optimizer_state_dict': self.optimizer.state_dict(),  # 优化器状态（包括动量缓存）
            'best_val_loss': self.best_val_loss,  # 最佳验证损失（用于早停状态恢复）
            'config': self.config.to_dict(),  # 配置对象（恢复时重建）
        }
        
        # 保存到磁盘（使用PyTorch的pickle协议）
        torch.save(checkpoint, filepath)
        
        if is_best:
            # 同时保存为best_model.pt（软链接或复制，便于直接加载）
            best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pt')
            import shutil
            shutil.copy(filepath, best_path)  # 复制文件（比软链接更兼容）
            print(f"[CHECKPOINT] New best model saved to {best_path} "
                  f"(val_loss: {self.best_val_loss:.4f}, step: {self.iter_num})")
    
    def load_checkpoint(self, filepath):
        """
        从检查点恢复训练（加载模型状态、优化器状态、训练进度）
        
        参数：
            filepath: 检查点文件路径
        """
        import os
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
            
        print(f"[CHECKPOINT] Loading from {filepath}...")
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 恢复模型状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 恢复优化器状态（必须在模型已移到设备后调用）
        if self.optimizer is None:
            self.optimizer = self.model.configure_optimizers(self.config)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 恢复训练状态
        self.iter_num = checkpoint['iter_num']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"[CHECKPOINT] Resumed from step {self.iter_num}, "
              f"best_val_loss: {self.best_val_loss:.4f}")

    def run(self):
        """
        增强版主训练循环（含验证、早停、学习率调度）
        
        对比原版改进：
        1. 每步动态调整学习率（warmup + cosine）
        2. 定期验证（val_interval）
        3. 保存最佳模型（基于验证损失）
        4. 早停机制（patience）
        5. 定期保存检查点（用于恢复训练）
        """
        
        model, config = self.model, self.config
        
        # 初始化优化器（如果尚未创建）
        if self.optimizer is None:
            self.optimizer = model.configure_optimizers(config)
        
        # 训练数据加载器（与原实现相同：有放回随机采样）
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(
                self.train_dataset, 
                replacement=True, 
                num_samples=int(1e10)  # 理论无限采样
            ),
            shuffle=False,
            pin_memory=True if self.device == 'cuda' else False,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        
        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        
        print(f"[INIT] Starting training...")
        print(f"[CONFIG] max_iters={config.max_iters}, lr={config.learning_rate}, "
              f"warmup={config.warmup_iters}, val_interval={config.val_interval}, "
              f"patience={config.patience}, device={self.device}")
        
        # 主训练循环（无限循环直到条件触发）
        while True:
            # ====================【学习率调度（每步更新）】====================
            if config.lr_scheduler == 'cosine':
                lr = self.get_lr(self.iter_num)
                # 手动更新优化器的学习率（遍历所有参数组）
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            
            # 数据获取（与原实现相同）
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            
            batch = [t.to(self.device) for t in batch]
            x, y = batch
            
            # 前向传播
            logits, self.loss, _ = model(x, y)
            
            # 反向传播与优化
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            
            # 梯度裁剪（Day 4注意：RMSNorm需更低阈值，如0.5）
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            
            self.optimizer.step()
            
            # 触发回调（日志记录、指标追踪等）
            self.trigger_callbacks('on_batch_end')
            
            # ====================【增强功能：验证与早停】====================
            # 每val_interval次迭代执行验证（且iter_num>0避免初始随机验证）
            if self.val_dataset is not None and self.iter_num % config.val_interval == 0 and self.iter_num > 0:
                val_loss = self.validate()
                
                if val_loss is not None:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"[VAL] Step {self.iter_num}: "
                          f"train_loss={self.loss.item():.4f}, "
                          f"val_loss={val_loss:.4f}, "
                          f"lr={current_lr:.2e}, "
                          f"best={self.best_val_loss:.4f}")
                    
                    # 检查是否有改善（容忍微小波动，但这里使用严格小于）
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss  # 更新最佳记录
                        self.steps_since_improvement = 0  # 重置早停计数器
                        # 保存最佳模型检查点
                        self.save_checkpoint(f"checkpoint_step{self.iter_num}.pt", is_best=True)
                    else:
                        self.steps_since_improvement += 1
                        print(f"[EARLY STOP] No improvement {self.steps_since_improvement}/{config.patience}")
                        
                        # 早停判断：连续patience次未改善则停止
                        if self.steps_since_improvement >= config.patience:
                            print(f"[EARLY STOP] Triggered at step {self.iter_num}. "
                                  f"Best val_loss: {self.best_val_loss:.4f}")
                            break  # 退出训练循环
            
            # 定期保存常规检查点（用于恢复训练，非最佳模型）
            # 每2*val_interval保存一次，避免频繁IO
            if self.iter_num % (config.val_interval * 2) == 0 and self.iter_num > 0:
                if not config.save_best_only:
                    self.save_checkpoint(f"checkpoint_step{self.iter_num}.pt", is_best=False)
            
            # 计时与终止条件
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow
            
            # 终止条件检查
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                print(f"[DONE] Reached max_iters {config.max_iters}")
                break
        
        # 训练结束：保存最终检查点（即使不是最佳）
        self.save_checkpoint("final_checkpoint.pt", is_best=False)
        print(f"[DONE] Training completed at step {self.iter_num}. "
              f"Best val_loss: {self.best_val_loss:.4f}")
