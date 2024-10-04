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
## 实验5 HMAX模型的实现

> 网上找资料找了很长时间，这个后面的几个层着实没咋看懂\
> 最后从[**一篇学长写的博客**](https://blog.csdn.net/ZEBRONE/article/details/125112108)中发现github上有HMAX的模型定义代码，就拿来用了

<https://github.com/wmvanvliet/pytorch_hmax>

-   这里的`hmax.py`文件里边定义了hmax的S1,C1,S2,C2层\

-   universal_patch_set有这个模型预训练的参数\

-   我们只需要加载这些参数然后处理数据就行了

    > 这里我感觉其是就是这几个层做一个特征工程，提取特征什么的，然后我们要实现的训练调参的就是最后的VTU层
:::

::: {.cell .markdown}
### 加载模型
:::

::: {.cell .code execution_count="22"}
``` python

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt  
import hmax
# Initialize the model with the universal patch set
# Initialize the model with the universal patch set
print('Constructing model')
model = hmax.HMAX('./universal_patch_set.mat')
#定义数据预处理函数  
transform = torchvision.transforms.Compose([torchvision.transforms.Grayscale(),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Lambda(lambda x: x * 255)])
#加载训练数据，测试数据  
train_data = torchvision.datasets.MNIST('./data', train=True, transform=transform, download=True)
test_data = torchvision.datasets.MNIST('./data', train=False, transform=transform)
#迭代器
BATCH_SIZE = 32
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)
# Determine whether there is a compatible GPU available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#这里直接方法奥CPU上训练，这样就可以无脑运行  
#想用GPU的话改一下这里device还有下面的输出记得先转到CPU上在做其他操作  
device='cpu'
# Run the model on the example images
print('Running model on', device)
model = model.to(device)
```

::: {.output .stream .stdout}
    Constructing model
    Running model on cpu
:::
:::

::: {.cell .markdown}
### 先展示一下处理过程
:::

::: {.cell .code}
``` python
sample_data=train_loader.dataset.__getitem__(0)  
plt.imshow(sample_data[0][0],cmap='gray')
plt.show()
all_layers=model.get_all_layers(sample_data[0].reshape(1,1,28,28))
s1=all_layers[0][0]
c1=all_layers[1][0]
s2=all_layers[2][0]
c2=all_layers[3][0]
#S1层的某个波段的四个方向
fig, axes = plt.subplots(1, len(s1[0]), figsize=(15, 5))   
for ax, img in zip(axes, s1[0]):
    ax.imshow(img)
    ax.axis('off')
plt.tight_layout()
plt.title('S1')
plt.show()
#C1层  
fig, axes = plt.subplots(1, len(c1[0]), figsize=(15, 5))   
for ax, img in zip(axes, c1[0]):
    ax.imshow(img)
    ax.axis('off')
plt.tight_layout()
plt.title('C1')
plt.show()
#S2
fig, axes = plt.subplots(1, len(s2[0][0][:4]), figsize=(15, 5))   
for ax, img in zip(axes, s2[0][0][:4]):
    ax.imshow(img)
    ax.axis('off')
plt.tight_layout()
plt.title('S2')
plt.show()
#C2输出的是一个向量
```
:::

::: {.cell .markdown}
### 利用Hmax处理数据
:::

::: {.cell .code}
``` python
#经过S1，C1，S2，C2处理的输入到VTU层的数据    
train_data_vtu=[]
train_data_labels=[]
test_data_vtu=[]  
test_data_labels=[]
print('处理数据')  
for X, y in tqdm(train_loader):
    c2=model.forward(X.to(device))
    max_out=torch.max(c2,dim=1)[0].numpy().tolist()
    train_data_vtu+=max_out  
    train_data_labels+=[int(i) for i in y]
for X, y in tqdm(test_loader):
    c2=model.forward(X.to(device))
    max_out=torch.max(c2,dim=1)[0].numpy().tolist()
    test_data_vtu+=max_out
    test_data_labels+=[int(i) for i in y]
```
:::

::: {.cell .markdown}
### VTU层使用SVM，线性核(比较快)

> 这里反正就耐心等吧，本来数据集也大(2分钟左右)\
> 可以自己随便写一个网络训练或者其他的方法试试效果
:::

::: {.cell .code}
``` python
from sklearn.svm import SVC
model = SVC(kernel="linear")
model.fit(train_data_vtu,train_data_labels)
print("Accuray:",model.score(test_data_vtu,test_data_labels))
```

::: {.output .stream .stdout}
    Accuray: 0.954
:::
:::