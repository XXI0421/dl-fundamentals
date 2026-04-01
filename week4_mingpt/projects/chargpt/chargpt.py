"""
Trains a character-level language model.
"""

import os
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
from mingpt.im_trainer import EnhancedTrainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/chargpt'

    # data
    C.data = CharDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'

    # trainer
    C.trainer = EnhancedTrainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster

    return C

# -----------------------------------------------------------------------------

class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128
        return C

    def __init__(self, config, data):
        self.config = config

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.config.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

# -----------------------------------------------------------------------------
from torch.utils.data import random_split # 随机分割数据集

if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct the training dataset
    text = open('input.txt', 'r').read() # don't worry we won't run out of file handles
    full_dataset = CharDataset(config.data, text)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(3407)  # 固定种子保证可复现
    )
    print(f"📊 数据划分: 训练集{train_size}条, 验证集{val_size}条")
    # construct the model
    config.model.vocab_size = full_dataset.get_vocab_size()
    config.model.block_size = full_dataset.get_block_size()
    model = GPT(config.model)

    # construct the trainer object
    trainer = EnhancedTrainer(config.trainer, model, train_dataset, val_dataset)

    # iteration callback
    def batch_end_callback(trainer):
    # 关键修复：兼容random_split后的Subset对象
        dataset = train_dataset
        if hasattr(train_dataset, 'dataset'):  # 如果是Subset，取原数据集
            dataset = train_dataset.dataset
        
        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 500 == 0:
            model.eval()
            with torch.no_grad():
                context = "O God, O God!"
                # 使用修复后的dataset访问stoi
                x = torch.tensor([dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
                y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
                completion = ''.join([dataset.itos[int(i)] for i in y])
                print(completion)
            
            # 保存时也用dataset
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'stoi': dataset.stoi,  # 保存词表！
                'itos': dataset.itos,
                'vocab_size': dataset.vocab_size,
                'config': config.model.to_dict(),
                'iter_num': trainer.iter_num
            }, ckpt_path)
            print("saving model")
            
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)


    # run the optimization
    trainer.run()
