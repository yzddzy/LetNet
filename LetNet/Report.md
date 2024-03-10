# Machine Learning Final Project
姓名：杨子董 学号：521120910138 

由于没有充足的计算资源，以下任务均选择option2来完成。
## 1.Mandatory Task: Fashion-MNIST clothing classification
### 1.1手写与调库实现实现PCA和t-SNE对比
这部分代码储存在data_required_opt2文件中,包含LetNet.py文件（调库实现PCA和t-SNE）, LeNet1.py文件（手写实现PCA和t-SNE）, MyNet.py文件（调库实现PCA和t-SNE）, MyNet1.py文件（手写实现PCA和t-SNE） 四个文件，同目录下还有生成的图片。经过测试，手写的PCA和t-SNE效果与调库的相似，但是可能由于算法和优化等原因，运算时长大于调库。下面是实现结果对比。

运用LeNet.py和LeNet1.py在相同条件下（学习率lr 0.0001，迭代次数num_epochs 500，可视化LeNet卷积层2的输出）进行测试对比:

调库实现PCA和t-SNE

![](https://notes.sjtu.edu.cn/uploads/upload_7faf568cbabc400926449ee57fed5836.png)

手写实现PCA和t-SNE

![](https://notes.sjtu.edu.cn/uploads/upload_a4e2478c1f76184648314ea791c0bfb6.png)

运用MyNet.py和MyNet1.py在相同条件下（学习率lr 0.0001，迭代次数 num_epochs 500，可视化MyNet卷积层2的输出）进行测试对比:

调库实现PCA和t-SNE

![](https://notes.sjtu.edu.cn/uploads/upload_a2f93432f21244f5042741ab8dba8e0a.png)

手写实现PCA和t-SNE

![](https://notes.sjtu.edu.cn/uploads/upload_96c49964f2d15fbf4486146d4081903a.png)


由图可见，无论是LeNet还是MyNet神经网络模型，手写实现的PCA和t-SNE与调库实现的效果基本一致，完成度较高，可以实现对调库的替换。但由于手写的PCA和t-SNE花费时间长于调库实现，而现在因为计算资源不足，训练时间已经较长，为节省时间，下面分析采用的是调库实现的可视化结果。
### 1.1LeNet
LeNet是一个经典的卷积神经网络模型,其结构为：卷积层1，池化层1，卷积层2，池化层2，卷积层3（经卷积核作用输出特征图大小为120X1X1的，可用全连接层代替），全连接层1，全连接层2，我们通过PyTorch中的nn.Module基类实现：
```
#实现LeNet神经网络模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

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
```
以下输出（除特殊说明外）均采用学习率lr 0.0001，迭代次数 num_epochs 500：

训练过程中的训练损失和测试损失的曲线（左）、训练精度和测试精度的曲线（右）

![](https://notes.sjtu.edu.cn/uploads/upload_9560e16518aebaa0e2f6e9d8445b9856.png)

这里我们可以看到随着训练次数的增加Train的Loss一直减小，Accuracy一直增加，但是Test的Loss和Accuracy似乎趋于稳定，Accuracy=80.00%。

扩大 num_epochs 1000：

![](https://notes.sjtu.edu.cn/uploads/upload_a88ae9165df156090808335ba416812f.png)

我们可以明显看到Test的Loss在增加，Accuracy收敛，Accuracy=80.10%，这也是LeNet模型的极限准确率。

卷积层2输出的PCA和t-SNE

![](https://notes.sjtu.edu.cn/uploads/upload_f298bb7822851c8c1bce4087b6dbb844.png)


全连接层1输出的PCA和t-SNE

![](https://notes.sjtu.edu.cn/uploads/upload_356cc7f567498a849ede3abc3419d022.png)

全连接层2（最后一层）输出的PCA和t-SNE

![](https://notes.sjtu.edu.cn/uploads/upload_08216a8edcd463e2d085804c874b56e2.png)

可视化这些中间层输出，我们可以清晰看到神经网络模型一步一步输出的效果，由PCA看到效果越来越好，因为PCA是通过线性变换将原始高维数据映射到低维空间，同时最大化投影数据的方差，而图像中显示类中距离较近，类间距离较远。由t-SNE也可以看到聚类效果整体向好，不过全连接层1输出的结果看起来并不比最终结果差，这是一个值得继续探究的点。

### 1.2MyNet
MyNet神经网络模型实在LeNet卷积神经网络模型基础上设计得到的，看到LeNet神经网络最后结果在t-SNE可视化中类和类之间的界限不是特别分明，我猜测可能是卷积核过大导致局部特征感知能力较弱，我就将conv1、conv2、conv3卷积核心从5减小到3，但这样运行后类与类之间混合在一起，分类效果较差，所以为了提升全局相关的特征感知能力，我将conv3的卷积核从3增加到6（实现可使用全连接层fc1代替），这时分类效果好了很多。考虑到在LeNet中发现全连接层1输出的结果看起来并不比最终结果差，我直接删去最后一个全连接层，让全连接层1成为最后一层，期待达到简化神经网络并提升能力的效果，但是删去之后类内距离有增加，类间距离有减小，效果变差，故重新添加上最后一个全连接层。这样MyNet神经网络模型设计完毕，下面是示意图和每一层的具体介绍：
![](https://notes.sjtu.edu.cn/uploads/upload_3dd8c693903bcb43bd2bf6297d93285e.jpg)


conv1：输入通道数为1，特征图大小为32x32，输出通道数为6，特征图大小为30x30，卷积核大小为3x3。目的是提取图像的低级特征，如边缘和纹理等。
pool1：采用Max pooling，输入通道数为6，特征图大小为30x30，输出通道数为6，特征图大小为15x15，核大小为2x2，步长为2。作用是减小特征图的空间尺寸，保留主要的特征，并且对平移和旋转变化具有一定的不变性。
conv2：输入通道数为6，特征图大小为15x15，输出通道数为16，特征图大小为13x13，卷积核大小为3x3。目的是进一步提取图像的高级特征，并且可以捕捉更复杂的模式。
pool2：采用Max pooling，输入通道数为16，特征图大小为13x13，输出通道数为16，特征图大小为6x6，核大小为2x2，步长为2。作用是继续减小特征图的空间尺寸。
fc1：将上一层的特征图展平为一维向量，输入通道数为16，特征图大小为5x5，输出大小为120。作用是将图像特征映射到更高维度的特征空间，为后续的分类做准备。
fc2：输入大小为120，输出大小为84。作用是进一步提取抽象的特征，并且减少特征维度。
fc3：输入大小为84，输出大小为10。作用是将抽象的特征映射到类别概率上，输出层的大小对应于分类问题的类别数量。
我们通过PyTorch中的nn.Module基类实现：
```
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
```
以下输出（除特殊说明外）均采用学习率lr 0.0001，迭代次数 num_epochs 500：

训练过程中的训练损失和测试损失的曲线（左）、训练精度和测试精度的曲线（右）

![](https://notes.sjtu.edu.cn/uploads/upload_58ff21823ff4b96f8a0bf03a6b94c130.png)
这里我们可以看到随着训练次数的增加Train的Loss一直减小，Accuracy一直增加，但是Test的Loss和Accuracy似乎趋于稳定，Accuracy=79.00%。

扩大 num_epochs 1000：

![](https://notes.sjtu.edu.cn/uploads/upload_001af8e57b1bbcc5eb7a550e2f6324bc.png)

我们可以明显看到Test的Loss在增加，Accuracy收敛，Accuracy=79.40%，这也是MyNet模型的极限准确率。

第二个卷积层输出的PCA和t-SNE

![](https://notes.sjtu.edu.cn/uploads/upload_a2f93432f21244f5042741ab8dba8e0a.png)



第二个全连接层输出的PCA和t-SNE

![](https://notes.sjtu.edu.cn/uploads/upload_68c585f2b9a3398b50df7092ffb98c13.png)


最后一层输出的PCA和t-SNE


![](https://notes.sjtu.edu.cn/uploads/upload_32db6a1035a881651dd0b5023292c6ba.png)

可视化MyNet中间层输出，我们可以清晰看到输出的效果一直在变好，图像中显示类中距离越来越近，类间距离越来越远，分类效果越来越好。我们可以清晰看到PCA中第最后一个全连接层比第二个全连接层的灰、蓝、棕部分明显更分开，t-SNE中每个颜色都更集中，距离其他颜色更远，这个应证了我们设计时重新添加最后一个全连接层效果更好的判断。该模型测试正确率79.40%和LetNet差不多，但是我们看t-SNE可视化效果类内距离比LeNet小，类间距离更大，分类更清晰，说明我们减小和增大卷积核的操作起了我们想要的在保持对全局特征感知能力的同时增强对局部特征的感知能力，MyNet模型在数据的特征提取和表示方面表现得更好。

## 2.Optional task 1: Image 2.reconstruction
这部分代码储存在data_option1_opt2文件中，有VAE.py一个文件，同目录下还有生成的图片。

VAE由两部分组成：编码器encoder和解码器decoder。编码器将输入数据映射到潜在空间中，而解码器则将潜在变量重新映射为重构数据。其中编码器使用了两个个全连接层：首先，将输入的图像数据（32x32x3）展平为一维向量（32323），然后通过一个全连接层得到一个中间表示，再经过一个ReLU激活函数。最后,通过一个全连接层输出潜在空间的均值mean和方差logvar（两倍的潜在变量维度），以便进行后续的采样。解码器同样使用了两个全连接层：首先，将潜在变量输入到一个全连接层，并通过ReLU激活函数。然后通过另一个全连接层得到最终的重构数据，再经过Sigmoid激活函数将数值范围限制在0到1之间。在模型的前向传播过程中[forward(self, x)]，首先通过编码器将输入数据映射为潜在空间的均值mean和方差logvar[encode(self, x)]。然后使用均值mean和方差logvar进行重参数化，通过加入随机噪声来采样潜在变量[reparameterize(self, mean, logvar)]。最后，将采样得到的潜在变量输入到解码器中进行重构[decode(self, z)]，并得到最终的输出结果。我们通过PyTorch中的nn.Module基类实现：
```
# VAE实现
class VAE(nn.Module):
    def __init__(self, lat_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(32*32*3, 512),
            nn.ReLU(),
            nn.Linear(512, lat_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(lat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 32*32*3),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = x.view(x.size(0), -1)
        latent = self.encoder(x)
        mean, logvar = torch.split(latent, lat_dim, dim=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mean + epsilon * std

    def decode(self, z):
        x_hat = self.decoder(z)
        x_hat = x_hat.view(x_hat.size(0), 3, 32, 32)
        return x_hat

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar
```

以下结果均在潜在向量维度lat_dim = 128，批处理大小bat_size = 32，插值列表a_ = [0, 0.2, 0.4, 0.6, 0.8, 1]，训练时采取MSE损失与KL散度&重构误差的和作为loss的条件下输出：

首先我们确定选取数据集中的第1张和第2张图像作为原始图像，他们主要特征区别时男性和女性：
![](https://notes.sjtu.edu.cn/uploads/upload_7245e15a688d1d149d1ba612a81ee6a7.png)

在loss中取MSE损失和KL散度&重构误差的和的比例为1:0.5，迭代次数epochs = 50：
![](https://notes.sjtu.edu.cn/uploads/upload_31d6a2f2fbb558fb1b1e6d61d7210fc5.png)

我们发现重建的图像趋于一致，没有区分度和原来图像相差较大，这是因为KL散度&重构误差的和在loss中权重过大，限制了潜在变量分布与标准正态分布之间的差异，导致生成图像多样性减少。

在loss中取MSE损失和KL散度&重构误差的和的比例为1:0.05，迭代次数epochs = 50：
![](https://notes.sjtu.edu.cn/uploads/upload_76abdef560abf6ef9fcd9e9ce498d91d.png)

在降低了KL散度&重构误差的和的权重后，我们明显看到重建的图像出现明显差异，区分度较大，和原来图像相似了不少，但是感觉蒙了层雾，融合的图片蕴含的左右图细节有缺失。

在loss中取MSE损失和KL散度&重构误差的和的比例为1:0.05，迭代次数epochs = 200：
![](https://notes.sjtu.edu.cn/uploads/upload_6c1eded632db236a1f3f865255f63546.png)

增加了迭代次数以后重建图像精度明显提高，背景和面部细节更明显，和原图更像，融合的图片蕴含的左右图细节更明显。

在loss中取MSE损失和KL散度&重构误差的和的比例为1:0.05，迭代次数epochs = 500：
![](https://notes.sjtu.edu.cn/uploads/upload_22fc50baedc4b0f62e033fc268fe5c8a.png)
增加了迭代次数到500以后，重建图像的细节进一步提升，与原图更像，融合的图片效果更自然，蕴含的左右图细节也更明显。


我们再多取几组相片查看插值生成效果如何，loss中取MSE损失和KL散度&重构误差的和的比例为1:0.05，迭代次数epochs = 200：

我们选取数据集中的第100张和第500张图像作为原始图像，他们主要特征区别为皮肤颜色以及秃头与有头发：
![](https://notes.sjtu.edu.cn/uploads/upload_cf047c45046891bd63f73a4ca61bff4a.png)
![](https://notes.sjtu.edu.cn/uploads/upload_7ba0339c9230bf2193b321349f3fefba.png)

我们可以看到中间a取不同值的图像都展现了两张图片融合的特征，比如a=0.2时，图像中人物五官更像左边的，而且头发更稀少，a=0.6时，两张图像的五官特征均有体现，皮肤颜色介于偏黄和偏白之间，头发也多了很多。插值生成效果较好。

我们选取数据集中的第650张和第900张图像作为原始图像，他们主要特征区别为背景以及衣服：
![](https://notes.sjtu.edu.cn/uploads/upload_54244c840e838aa7767ac99e3d8438a3.png)

![](https://notes.sjtu.edu.cn/uploads/upload_1abbd29894b4b6e8faa8c29d7032809f.png)
我们可以看到中间a取不同值的图像都展现了两张图片融合的特征，比如a=0.8时，图像背景就更接近于右边，偏棕色，衣服也更像右边，白色衬衫较为明显，同时也有少许红色领带，带有融合特征，a=0.2时，两张图像的五官特征均有体现，左边的五官更明显，特别是眼睛，衣服也更像左边，蓝色衬衫和红色领带较为明显。插值生成效果较好。

综上所述，实现的VAE效果较好，图像重建与原图相似且细节完整，插值生成的对不同的插值a较为敏感而且特征融合效果明显。



