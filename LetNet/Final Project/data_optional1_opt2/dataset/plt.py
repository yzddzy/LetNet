import numpy as np
import matplotlib.pyplot as plt

# 加载数据集
data = np.load('X_train.npy')

# 查看数据集形状
print("数据集形状:", data.shape)

# 查看数据集中的样本
sample = data[999]
print("第n个样本:", sample)

# 可视化样本
first_sample = sample.transpose(1, 2, 0)
plt.imshow(first_sample)
plt.axis('off')
plt.show()