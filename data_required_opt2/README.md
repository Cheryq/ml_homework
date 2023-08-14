###<center> Final Project: Classification
</center>

####Project framework

> dataset :包含训练与测试数据与读取文件
>>+ \_\_init__.py: 初始化文件
>>+ get\_data.py : 读取数据函数与处理文件
>>+ X_test.npy: 测试集样本数据
>>+ X_train.npy: 训练集样本数据
>>+ Y_test.npy: 测试集样本标签数据
>>+ Y_train.npy: 训练集样本标签数据

> LeNet_img: LeNet训练过程与结果中可视化图片
>>+ LeNet_Accuracy.png: LeNet训练过程中accuracy曲线
>>+ LeNet_Loss.png: LeNet训练过程中loss曲线
>>+ LeNet_PCA_conv.png: LeNet模型中层卷积层特征PCA可视化
>>+ LeNet_PCA_fcl.png: LeNet模型中层全连接层特征PCA可视化
>>+ LeNet_PCA_output.png: LeNet模型最后一层特征输出PCA可视化
>>+ LeNet_TSNE_conv.png: LeNet模型中层卷积层特征TSNE可视化
>>+ LeNet_TSNE_fcl.png: LeNet模型中层全连接层特征TSNE可视化
>>+ LeNet_TSNE_output.png: LeNet模型最后一层特征输出TSNE可视化

>Mynet_img: 自己设计的网络训练过程与结果中可视化图片
>>+ Mynet_Accuracy.png: Mynet训练过程中accuracy曲线
>>+ Mynet_Loss.png: Mynet训练过程中loss曲线
>>+ Mynet_PCA_conv.png: Mynet模型中层卷积层特征PCA可视化
>>+ Mynet_PCA_fcl.png: Mynet模型中层全连接层特征PCA可视化
>>+ Mynet_PCA_output.png: Mynet模型最后一层特征输出PCA可视化
>>+ Mynet_TSNE_conv.png:Mynet模型中层卷积层特征TSNE可视化
>>+ Mynet_TSNE_fcl.png: Mynet模型中层全连接层特征TSNE可视化
>>+ Mynet_TSNE_output.png: Mynet模型最后一层特征输出TSNE可视化

>LeNet.py: LeNet网络模型定义以及训练函数定义

>Mynet.py: 自己设计的网络模型定义以及训练函数

>PCA.py: PCA可视化方法的实现

>t_SNE.py: t_SNE可视化方法的实现

>train.py: 对模型的训练函数，包括输出训练过程中accuracy和loss图像

>main.py: 主函数，包括数据的读取步骤以及模型训练步骤以及调用可视化函数实现可视化

>LeNet_model.pt:训练完成的LeNet网络模型，可以直接用于读取

>Mynet_model.pt:训练完成的Mynet网络模型，可以直接用于读取

####Usage
两个网络模型已经训练完成，可以直接在原有main函数基础上运行以完成数据集的读取以及调用两个模型进行分类并可视化中层特征。

若要重新进行训练需要将157、183行对训练完成的模型读取代码进行注释，将162、187行代码注释取消对模型进行训练。
