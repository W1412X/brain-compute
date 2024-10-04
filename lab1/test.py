import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))
    def train(self, training_data):
        for pattern in training_data:
            for i in range(self.size):
                for j in range(self.size):
                    if i != j:
                        self.weights[i, j] += (2 * pattern[i] - 1) * (2 * pattern[j] - 1)
            self.weights/=30
    def predict(self, input_pattern, iterations=100):
        pattern = np.copy(input_pattern)
        for _ in range(iterations):
            for i in range(self.size):
                sum_input = np.dot(self.weights[:, i], pattern) - self.weights[i, i] * pattern[i]
                pattern[i] = 1 if sum_input >= 0 else 0
        return pattern
    def visualize_pattern(self, pattern):
        image = pattern.reshape(6,5)
        plt.imshow(image)
        plt.show()
# 定义每个数字的 6x5 像素矩阵
digit_matrices = {
    0: [[1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]],
    1: [[0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0]],
    2: [[1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1]],
    3: [[1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]],
    4: [[1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0]],
    5: [[1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]],
    6: [[1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]],
    7: [[1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1]],
    8: [[1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]],
    9: [[1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]]
}

# 展开为1x30的矩阵
number_matrix_list = [np.array(digit_matrices[i]).flatten() for i in range(10)]
# 使用示例
size = 30

network = HopfieldNetwork(size)
network.train(number_matrix_list)

# 测试图案
c = np.array([[0, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 1, 1, 1, 1]])
c=c.flatten()
result_pattern = network.predict(c)
plt.imshow(network.weights)
plt.show()
network.visualize_pattern(c)
network.visualize_pattern(result_pattern)