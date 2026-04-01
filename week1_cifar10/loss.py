import torch

# 定义单神经元：z = wx + b, a = sigmoid(z), C = (a - y)^2
x = torch.tensor([2.0])          # 输入
y = torch.tensor([0.8])          # 目标
w = torch.tensor([0.5], requires_grad=True)
b = torch.tensor([0.1], requires_grad=True)

# 前向
z = w * x + b
a = torch.sigmoid(z)
C = (a - y) ** 2

# 反向
C.backward()

print(f"手算验证：")
print(f"∂C/∂w = {w.grad.item():.4f}")  # 对比你的手算结果
print(f"∂C/∂b = {b.grad.item():.4f}")
