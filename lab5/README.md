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

::: {.cell .code execution_count="35"}
``` python

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pickle
import numpy as np
from tqdm import tqdm
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
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE)
# Determine whether there is a compatible GPU available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#这里直接方法奥CPU上训练，这样就可以无脑运行  
#想用GPU的话改一下这里device还有下面的输出记得先转到CPU上在做其他操作  
device='cpu'
# Run the model on the example images
print('Running model on', device)
model = model.to(device)  
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

::: {.output .stream .stdout}
    Constructing model
    Running model on cpu
    处理数据
:::

::: {.output .stream .stderr}
    100%|██████████| 1875/1875 [14:01<00:00,  2.23it/s]
    100%|██████████| 313/313 [02:19<00:00,  2.24it/s]
:::
:::

::: {.cell .markdown}
### VTU层使用SVM，线性核(比较快)

> 这里反正就耐心等吧，本来数据集也大(2分钟左右)\
> 可以自己随便写一个网络训练或者其他的方法试试效果
:::

::: {.cell .code execution_count="37"}
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