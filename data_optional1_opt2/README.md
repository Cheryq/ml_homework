###<center>Optional task 1: Image reconstruction

####Project Framework:
> dataset :包含训练与测试数据与读取文件
>>* \_\_init__.py:初始化文件
>>* get_data_.py:数据读取以及处理文件
>>* X_train.npy: 训练集样本数据

>img: 模型训练loss曲线以及图像重构结果图片

>VAE.py: VAE模型的实现包括训练函数

>VAE.pth: 训练完成的VAE模型文件保存

>main.py: 主函数包括模型训练以及对选好的图片进行重构可视化

####Usage：

main.py主函数中默认是对训练好的模型进行读取，若要重新训练可以将26行读取模型的代码注释，将25行训练模型的代码取消注释。

我将图片分类为male和female两类，两个列表分别包含两类图片的编号，可以在此添加或删除图片编号来修改需要实现的图片。（需要注意male和female的个数需要一致，因为模型每次按顺序从两个列表读取图片）

图片结果保存在img中并且按顺序进行命名.
