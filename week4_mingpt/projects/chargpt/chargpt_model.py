"""
Trains a character-level language model (原始model版本).
支持不同模型保存到不同文件: ./out/chargpt/<model_name>.pt
"""

import os
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/chargpt'
    C.system.model_name = 'model'  # 模型名称

    # data
    C.data = CharDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4

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
        chunk = self.data[idx:idx + self.config.block_size + 1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    model_name = config.system.model_name
    ckpt_path = os.path.join(config.system.work_dir, f"{model_name}.pt")
    print(f"模型将保存到: {ckpt_path}")

    text = open('input.txt', 'r').read()
    full_dataset = CharDataset(config.data, text)

    config.model.vocab_size = full_dataset.get_vocab_size()
    config.model.block_size = full_dataset.get_block_size()
    model = GPT(config.model)

    trainer = Trainer(config.trainer, model, full_dataset)

    def batch_end_callback(trainer):
        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 500 == 0:
            model.eval()
            with torch.no_grad():
                context = "O God, O God!"
                x = torch.tensor([full_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
                y = model.generate(x, 200, temperature=0.7, do_sample=True, top_k=10)[0]
                completion = ''.join([full_dataset.itos[int(i)] for i in y])
                print(completion)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'stoi': full_dataset.stoi,
                'itos': full_dataset.itos,
                'vocab_size': full_dataset.vocab_size,
                'config': config.model.to_dict(),
                'iter_num': trainer.iter_num,
                'model_name': model_name
            }, ckpt_path)
            print(f"saving model to {ckpt_path}")
            
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    trainer.run()
