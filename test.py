import torch
import numpy as np

# 定义目标函数，这里是一条直线模型 y = a*x + b
def model(params, x):
    a, b = params
    return a * x + b

# 定义误差函数，即观测数据与模型之间的差异
def error(params, x, y_observed):
    return model(params, x) - y_observed

# 生成一些观测数据
torch.manual_seed(0)
x_data = torch.linspace(0, 1, 100)
y_true = 2 * x_data + 1
y_noise = torch.normal(0, 0.1, size=x_data.shape) # type: ignore
y_data = y_true + y_noise

# 初始参数值
initial_params = torch.tensor([1.0, 1.0], requires_grad=True)

# 学习率
learning_rate = 0.0015

# 迭代优化
for i in range(100):
    # 计算误差
    loss = torch.sum(error(initial_params, x_data, y_data) ** 2)

    # 使用自动微分计算梯度
    loss.backward()

    # 更新参数
    with torch.no_grad():
        initial_params -= learning_rate * initial_params.grad # type: ignore

    # 梯度清零
    initial_params.grad.zero_()

# 输出拟合结果
print("Fitted parameters:", initial_params.detach().numpy())
