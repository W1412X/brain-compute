## 实验4 LSTM

### 构造数据

```python
import numpy as np
import torch  
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt  
from torch.utils.data import DataLoader, Dataset  

# 随机生成足够多的八位二进制数  
def generate_number(num=100000):
    a = np.random.randint(0, 128, num)
    b = np.random.randint(0, 128, num)
    sum_ab = a + b
    return [[list(map(int, np.binary_repr(i, 8))), list(map(int, np.binary_repr(u, 8))), list(map(int, np.binary_repr(k, 8)))] for i, u, k in zip(a, b, sum_ab)]
```

### LSTM 的基本结构

LSTM 单元主要包括三个门：遗忘门、输入门、输出门，以及细胞状态和隐藏状态。

#### 1. 遗忘门 (Forget Gate)

遗忘门决定了当前细胞状态应该遗忘多少信息。其公式为：

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

- $ f_t $：遗忘门的输出
- $ \sigma $：sigmoid 激活函数
- $ W_f $：遗忘门的权重矩阵
- $ h_{t-1} $：上一个时刻的隐藏状态
- $ x_t $：当前输入
- $ b_f $：遗忘门的偏置

#### 2. 输入门 (Input Gate)

输入门决定了当前输入将对细胞状态的更新程度。其公式为：

$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$

- $ i_t $：输入门的输出
- $ W_i $：输入门的权重矩阵
- $ b_i $：输入门的偏置

#### 3. 细胞状态候选 (Cell State Candidate)

细胞状态候选值用于更新细胞状态，其公式为：

$$
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
$$

- $ \tilde{C}_t $：细胞状态候选
- $ W_C $：细胞状态候选的权重矩阵
- $ b_C $：细胞状态候选的偏置

#### 4. 更新细胞状态 (Cell State Update)

更新细胞状态的公式为：

$$
C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t
$$

- $ C_t $：当前细胞状态
- $ C_{t-1} $：上一个时刻的细胞状态

#### 5. 输出门 (Output Gate)

输出门决定了当前细胞状态的哪些部分将被输出。其公式为：

$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$

- $ o_t $：输出门的输出
- $ W_o $：输出门的权重矩阵
- $ b_o $：输出门的偏置

#### 6. 隐藏状态 (Hidden State)

最终的隐藏状态通过细胞状态和输出门计算得出：

$$
h_t = o_t \cdot \tanh(C_t)
$$

- $ h_t $：当前隐藏状态

### 总结

- **输入门** $i_t$ 决定哪些新信息将被存储在细胞状态中。
- **遗忘门** $f_t$ 决定哪些信息将被丢弃。
- **输出门** $o_t$ 决定哪些信息将从细胞状态中输出，形成当前的隐藏状态 $h_t$。

> 难崩，写完前向传播发现反向传播自己写的话就有点难搞了，而且后来的实验貌似有写反响传播的，远没有这个复杂，所以还是调库把。

### 定义模型

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 8)  # 输出层

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 只取最后一个时间步的输出
        return out
```

### 训练

```python
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data  # 输入训练样本数目*2*8  
        self.labels = labels  # 加法结果 训练样本数目*8  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
```

```python
# 划分训练集，测试集    
data = generate_number()
train_data = data[:80000]
test_data = data[80000:]

train_x = torch.tensor(train_data, dtype=torch.float32)[:, :2, :]
train_y = torch.tensor(train_data, dtype=torch.float32)[:, -1:, :]
train_set = MyDataset(train_x, train_y.reshape(train_y.shape[0], 8))

test_x = torch.tensor(test_data, dtype=torch.float32)[:, :2, :]
test_y = torch.tensor(test_data, dtype=torch.float32)[:, -1:, :]
test_set = MyDataset(test_x, test_y.reshape(test_y.shape[0], 8))

# 定义模型  
model = LSTMModel(8, 32, 2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_loader = DataLoader(train_set, batch_size=64)

# 训练
epoch_num = 80
for epoch in tqdm(range(epoch_num)):
    for batch, label in train_loader:
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
```

```python
# 测试
test_re = np.where(model(test_x).detach().numpy() > 0.5, 1, 0).tolist()
real_y = np.where(test_y.numpy() == 1, 1, 0).reshape(test_y.shape[0], 8).tolist()

right_num = 0
for i in range(len(real_y)):
    if test_re[i] == real_y[i]:
        right_num += 1

print('Accuray:{}%'.format(100 * right_num / 20000))
```

> 可以尝试修改时间步为16，特征数为1，我试的效果不是很好，80轮正确率在24%左右