### 构建训练数据(真值表)
```python
train_data = [
    ([0, 0], 0),  # AND
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1),
]

train_data_or = [
    ([0, 0], 0),  # OR
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1),
]

train_data_not = [
    ([0], 1),  # NOT
    ([1], 0),
]
```

### 定义感知机类

```python
import numpy as np

class Perceptron:
    def __init__(self, input_size):
        self.weights = np.random.rand(input_size + 1)  # self.weights[0]是偏执项  

    def activation(self, x):  # 激活函数
        return 1 if x >= 0 else 0

    def predict(self, inputs):  # 预测
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.activation(summation)

    def train(self, training_data, epochs=10, learning_rate=0.1):
        for _ in range(epochs):
            for inputs, label in training_data:
                prediction = self.predict(inputs)
                error = label - prediction  # 计算误差
                self.weights[1:] += learning_rate * error * np.array(inputs)  # 更新权重
                self.weights[0] += learning_rate * error  # 更新偏执项
```

### 构建和训练感知机

```python
perceptron_and = Perceptron(input_size=2)
perceptron_and.train(train_data)

perceptron_or = Perceptron(input_size=2)
perceptron_or.train(train_data_or)

perceptron_not = Perceptron(input_size=1)
perceptron_not.train(train_data_not)
```

### 测试

```python
test_data_and = [([1, 1], 'AND'), ([0, 0], 'AND')]
test_data_or = [([1, 0], 'OR'), ([0, 1], 'OR')]
test_data_not = [([1], 'NOT'), ([0], 'NOT')]

for data, operation in test_data_and:
    print(f"{operation} output: {perceptron_and.predict(data)}")

for data, operation in test_data_or:
    print(f"{operation} output: {perceptron_or.predict(data)}")

for data, operation in test_data_not:
    print(f"{operation} output: {perceptron_not.predict(data)}")
```

### 运行结果

```
AND output: 1
AND output: 0
OR output: 1
OR output: 1
NOT output: 0
NOT output: 1
```