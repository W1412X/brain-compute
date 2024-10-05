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
## BP算法分析与实现
:::

::: {.cell .markdown}
### 反向传播算法

[**一个介绍反向传播算法的文章，本代码依据此文章的公式实现**](https://blog.csdn.net/fsfjdtpzus/article/details/106256925)

-   代码写的有点烂，不是很规范\
-   可以自定义Layer，保证输入是2，输出是1以及前一层的输出和后一层的输入相同即可
:::

::: {.cell .markdown}
### 导入库
:::

::: {.cell .code execution_count="14"}
``` python
import numpy as np  
```
:::

::: {.cell .code execution_count="15"}
``` python
np.random.seed(5)
np.random.random((2,5))  
```

::: {.output .execute_result execution_count="15"}
    array([[0.22199317, 0.87073231, 0.20671916, 0.91861091, 0.48841119],
           [0.61174386, 0.76590786, 0.51841799, 0.2968005 , 0.18772123]])
:::
:::

::: {.cell .markdown}
### 定义任意一层的类
:::

::: {.cell .code execution_count="32"}
``` python
learn_rate=0.1  
class Layer():
    def __init__(self,input_size,output_size,learn_rate=learn_rate):
        self.input_size=input_size
        self.output_size=output_size  
        self.learn_rate=learn_rate  
        self.input=None    
        self.output=None  
        np.random.seed(1412)
        self.weights=np.random.random((output_size,input_size))  
    def forward(self,x):#x是 1*input_size 
        self.input=x  
        y=np.zeros(self.output_size)
        for i in range(len(y)):
            y[i]=self.weights[i].reshape(1,-1) @ x    
        y=np.array([i if i>0 else 0.00001*i for i in y])#使用relu函数  
        return y.reshape(-1,1)  
    def backward(self,loss):#得到一个loss树组，长度=output_size
        for ind1 in range(self.weights.shape[0]):
            for ind2 in range(self.weights.shape[1]):
                tmp=1 if self.input[ind2][0]>0 else 0.00001  
                #更新权重  
                self.weights[ind1][ind2]+=self.learn_rate*tmp*self.input[ind2][0]*loss[ind1][0]  
        return self.weights    
```
:::

::: {.cell .markdown}
### 定义网络类
:::

::: {.cell .code execution_count="61"}
``` python
class Net():
    def __init__(self,layers):
        self.layers=layers  
    def predict(self,x):
        output=x
        for ind in range(len(self.layers)):
            output=self.layers[ind].forward(output)
        return output  
    def train(self,x,y):
        #计算输出
        output=x
        for ind in range(len(self.layers)):
            output=self.layers[ind].forward(output)
        loss_out=y-output
        loss_all_layers=[np.array(loss_out).reshape(-1,1)]
        for ind in [i for i in range(len(self.layers))][::-1][:-1]:
            tmp_loss=self.layers[ind].weights.T@loss_all_layers[-1]
            loss_all_layers.append(tmp_loss)  
        loss_all_layers=loss_all_layers[::-1]
        for ind in [i for i in range(len(self.layers))][::-1]:
            self.layers[ind].backward(loss_all_layers[ind])
```
:::

::: {.cell .code execution_count="78"}
``` python
layers_list=[[Layer(2,3),Layer(3,2),Layer(2,1)],[Layer(2,1)]]
x=[[0.35],[0.9]]
y=[[0.5]]
loss_list=[]
for layers in layers_list:
    losses=[]
    net=Net(layers=layers)  
    for i in range(40):
        net.train(x,y)
        losses.append((1/2)*np.sqrt((net.predict(x)[0][0]-y[0][0])**2))
    loss_list.append(losses)
```
:::

::: {.cell .markdown}
### 展示不同的结构的训练过程中的损失值以及预测值的变化
:::

::: {.cell .code}
``` python
import matplotlib.pyplot as plt   
fig,ax=plt.subplots()
fig.set_size_inches(10,6)
ax.plot(np.linspace(0,len(loss_list[0]),len(loss_list[0])),loss_list[0],linewidth=1,c='b',label='3-layer')
ax.plot(np.linspace(0,len(loss_list[1]),len(loss_list[1])),loss_list[1],linewidth=1,c='r',label='1-layer')
ax.legend()
fig.show()
```
:::