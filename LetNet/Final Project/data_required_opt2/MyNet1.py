import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from dataset import get_data
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#自己构建MyNet神经网络模型
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3)  #输入通道数为1，输出通道数为6，卷积核大小为3x3
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  #最大池化层，核大小为2x2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3)  #输入通道数为6，输出通道数为16，卷积核大小为3x3
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  #最大池化层，核大小为2x2
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  #全连接层，输入大小为16x6x6，输出大小为120
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)  #全连接层，输入大小为120，输出大小为84
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)  #全连接层，输入大小为84，输出大小为10

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        return x
    
if __name__ == '__main__':

    #读取并处理数据
    X_train, X_test, Y_train, Y_test = get_data('dataset')

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    #创建MyNet模型和优化器
    model = MyNet()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    #训练和测试的损失和精度
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    #训练模型
    num_epochs = 500
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images
            labels = labels

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        #输出每次训练的Loss和Accuracy
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images
                labels = labels

                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss /= len(test_loader)
        test_accuracy = correct / total
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}]  Test Loss: {test_loss:.4f}  Accuracy: {test_accuracy * 100:.2f}%")
    print("Train complete!")

    #可视化train和test的Loss曲线
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
    plt.title('Training and Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    #可视化train和test的Accuracy曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs+1), test_accuracies, label='Test Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    #获取中间层特征
    model.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images
            labels = labels

            output1 = model.conv1(images)
            output2 = model.relu1(output1)
            output3 = model.pool1(output2)
            output4 = model.conv2(output3)
            output5 = model.relu2(output4)
            output6 = model.pool2(output5)
            output7 = output6.view(output6.size(0), -1)
            output8 = model.fc1(output7)
            output9 = model.relu3(output8)
            output10 = model.fc2(output9)
            output11 = model.relu4(output10)
            output12 = model.fc3(output11)

            intermediate_output = output4

            intermediate_output = intermediate_output.view(intermediate_output.size(0), -1)  
            all_features.append(intermediate_output.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)


    #手动实现PCA降维
    def pca_transform(features, n_components):
        #计算均值
        mean = np.mean(features, axis=0)

        #减去均值
        features_centered = features - mean

        #计算协方差矩阵
        cov_matrix = np.cov(features_centered.T)

        #计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        #选择前n_components个最大特征值对应的特征向量
        sorted_indices = np.argsort(eigenvalues)[::-1][:n_components]
        selected_eigenvectors = eigenvectors[:, sorted_indices]

        #将数据投影到选定的特征向量上
        pca_features = np.dot(features_centered, selected_eigenvectors)

        return pca_features

    #调用手动实现的PCA降维函数
    n_components = 2
    pca_features = pca_transform(all_features, n_components)

    #pairwise距离矩阵
    def cal_pairwise_dist(x):
        sum_x = np.sum(np.square(x), 1)
        dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
        return dist

    #困惑度和条件概率分布
    def cal_perplexity(dist, idx=0, beta=1.0):
        prob = np.exp(-dist * beta)
        prob[idx] = 0
        sum_prob = np.sum(prob)
        
        if sum_prob == 0 or np.isnan(sum_prob):
            prob = np.maximum(prob, 1e-12)
            perp = -12
        else:
            prob /= sum_prob
            prob = np.maximum(prob, 1e-12)
            perp = np.sum(-prob * np.log(prob))
        
        return perp, prob

    #寻找beta并计算prob
    def search_prob(x, tol=1e-5, perplexity=30.0):
        (n, d) = x.shape
        dist = cal_pairwise_dist(x)
        pair_prob = np.zeros((n, n))
        beta = np.ones((n, 1))
        base_perp = np.log(perplexity)

        for i in range(n):
            betamin = -np.inf
            betamax = np.inf
            perp, this_prob = cal_perplexity(dist[i], i, beta[i])

            perp_diff = perp - base_perp
            tries = 0
            while np.abs(perp_diff) > tol and tries < 50:
                if perp_diff > 0:
                    betamin = beta[i].copy()
                    if betamax == np.inf or betamax == -np.inf:
                        beta[i] = beta[i] * 2
                    else:
                        beta[i] = (beta[i] + betamax) / 2
                else:
                    betamax = beta[i].copy()
                    if betamin == np.inf or betamin == -np.inf:
                        beta[i] = beta[i] / 2
                    else:
                        beta[i] = (beta[i] + betamin) / 2

                perp, this_prob = cal_perplexity(dist[i], i, beta[i])
                perp_diff = perp - base_perp
                tries = tries + 1

            pair_prob[i,] = this_prob

        return pair_prob

    #手动实现t-SNE降维
    def tsne(x, no_dims=2, initial_dims=50, perplexity=30.0, max_iter=800):
        if isinstance(no_dims, float):
            print("Error: array x should have type float.")
            return -1
        if round(no_dims) != no_dims:
            print("Error: number of dimensions should be an integer.")
            return -1

        (n, d) = x.shape

        eta = 500
        y = np.random.randn(n, no_dims)
        dy = np.zeros((n, no_dims))
        P = search_prob(x, 1e-5, perplexity)
        P = P + np.transpose(P)
        P = P / np.sum(P)
        P = P * 4
        P = np.maximum(P, 1e-12)

        for iter in range(max_iter):
            sum_y = np.sum(np.square(y), 1)
            num = 1 / (1 + np.add(np.add(-2 * np.dot(y, y.T), sum_y).T, sum_y))
            num[range(n), range(n)] = 0
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-12)

            PQ = P - Q
            for i in range(n):
                dy[i,:] = np.sum(np.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (y[i,:] - y), 0)

            y = y - eta*dy
            y = y - np.tile(np.mean(y, 0), (n, 1))

            if (iter + 1) % 50 == 0:
                if iter > 100:
                    C = np.sum(P * np.log(P / Q))
                else:
                    C = np.sum(P/4 * np.log(P/4 / Q))
                print("Iteration ", (iter + 1), ": error is ", C)

            if iter == 100:
                P = P / 4

        print("Finished training!")
        return y
    
    #t-SNE降维
    tsne_features = tsne(all_features, no_dims=2, initial_dims=50, perplexity=30.0, max_iter=800)

    #可视化PCA降维的散点图
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(pca_features[:, 0], pca_features[:, 1], c=all_labels, cmap='tab10')
    plt.title('PCA Visualization')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.colorbar()

    #可视化t-SNE降维的散点图
    plt.subplot(1, 2, 2)
    plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=all_labels, cmap='tab10')
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

