---
jupyter:
  kernelspec:
    display_name: venv
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.8.10
  nbformat: 4
  nbformat_minor: 2
---

::: {.cell .markdown}
## 卷积神经网络

### 组成

-   数据输入层，卷积计算层，激励层(ReLU激活函数)，池化层(平均池化和最大池化)，全连接层，输出层

    ### 按照实验书上写的

-   数据输入层\
    对原始数据做归一话操作，将图像数据处理为均值为0，方差为1的数据\

-   卷积计算层\
    \... \...

    > 这里注意要保证卷积核可以覆盖到图像的边缘区域，所以需要填充
:::

::: {.cell .code execution_count="84"}
``` python
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  
import torch.nn as nn
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 
def input_layer(data):#这里做输入层的归一化的操作
    data = data / 255.0 
    mean_list = torch.mean(data.float(), dim=(1, 2)).view(-1, 1, 1)
    std_list = torch.std(data.float(), dim=(1, 2)).view(-1, 1, 1)
    data = (data - mean_list) / std_list 
    return data
class NormalizeTransform:
    def __call__(self, img):
        img = transforms.ToTensor()(img) 
        return input_layer(img) 
transform = transforms.Compose([
    NormalizeTransform(),
])
train_data = torchvision.datasets.MNIST('./data', train=True, transform=transform,download=True)
test_data = torchvision.datasets.MNIST('./data', train=False,transform=transform)
```
:::

::: {.cell .code execution_count="87"}
``` python
import torch
import torch.nn as nn
import torch.nn.functional as F
class ConvNet(nn.Module):
    def __init__(self, out_size=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, out_size)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```
:::

::: {.cell .markdown}
### 训练
:::

::: {.cell .code execution_count="107"}
``` python
model=ConvNet(10)
train_loader=DataLoader(train_data,batch_size=32)
test_loader=DataLoader(test_data,batch_size=32)
epoch_num=10
loss_list=[]
accuracy_list=[]
criterion=nn.CrossEntropyLoss()  
def get_accuracy():#获取在测试集上的精确度  
    right_num=0
    total_num=0
    for data,label in test_loader:
        outputs=model(data)
        pred=torch.argmax(outputs,dim=1)
        for i,p in enumerate(pred):
            total_num+=1
            if(p==label[i]):
                right_num+=1  
    return right_num/total_num
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)  
for epoch in tqdm(range(epoch_num)):
    tmp_loss=[]
    for data,labels in train_loader:
        optimizer.zero_grad()
        outputs=model(data)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        tmp_loss.append(loss.float())
    loss_list.append(torch.mean(torch.tensor(tmp_loss)).float())
    accuracy_list.append(get_accuracy())
```

::: {.output .stream .stderr}
    100%|██████████| 10/10 [04:51<00:00, 29.16s/it]
:::
:::

::: {.cell .code}
``` python
import matplotlib.pyplot as plt  
plt.plot(np.linspace(0,epoch_num,epoch_num),loss_list)
plt.title('loss')
plt.show()
plt.plot(np.linspace(0,epoch_num,epoch_num),accuracy_list)
plt.title('accuracy')
plt.show()
```
:::