import os
import numpy as np
from dataset import get_data,normalize
import LeNet
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from torch.utils.data import TensorDataset, DataLoader
from train import model_training
from PCA import PCA
from t_SNE import TSNE
import Mynet

def PCA_tSNE_show(name,label,conv,fcl,output):
    colors = ['r', 'k', 'yellow', 'g', 'darkorange', 
          'b', 'm', 'c', 'slategray', 'peru']
    cmap = ListedColormap(colors)
    
    conv_show=PCA(conv, 2)
    fcl_show=PCA(fcl, 2)
    output_show=PCA(output, 2)

    conv_show=np.column_stack((conv_show, np.array(label)))
    plt.scatter(conv_show[:, 0], conv_show[:, 1], c=conv_show[:, 2], cmap=cmap)
    plt.xlabel('dimension1')
    plt.ylabel('dimension2')
    plt.colorbar()
    plt.title('PCA_conv_Visualization')
    if name==0:
        plt.savefig('LeNet_img/LeNet_PCA_conv.png')
    else:
        plt.savefig('Mynet_img/Mynet_PCA_conv.png')
    #plt.show()
    plt.clf()

    
    fcl_show=np.column_stack((fcl_show, np.array(label)))
    plt.scatter(fcl_show[:, 0], fcl_show[:, 1], c=fcl_show[:, 2], cmap=cmap)
    plt.xlabel('dimension1')
    plt.ylabel('dimension2')
    plt.colorbar()
    plt.title('PCA_fcl_Visualization')
    if name==0:
        
        plt.savefig('LeNet_img/LeNet_PCA_fcl.png')
    else:
        
        plt.savefig('Mynet_img/Mynet_PCA_fcl.png')
    #plt.show()
    plt.clf()

    
    output_show=np.column_stack((output_show, np.array(label)))
    plt.scatter(output_show[:, 0], output_show[:, 1], c=output_show[:, 2], cmap=cmap)
    plt.xlabel('dimension1')
    plt.ylabel('dimension2')
    plt.colorbar()
    plt.title('PCA_output_Visualization')
    if name==0:
        
        plt.savefig('LeNet_img/LeNet_PCA_output.png')
    else:
        
        plt.savefig('Mynet_img/Mynet_PCA_output.png')
    #plt.show()
    plt.clf()

    output_show=TSNE(output,2,10)
    conv_show=TSNE(conv,2,1600)
    fcl_show=TSNE(fcl,2,84)

    conv_show=np.column_stack((conv_show, np.array(label)))
    plt.scatter(conv_show[:, 0], conv_show[:, 1], c=conv_show[:, 2], cmap=cmap)
    plt.xlabel('dimension1')
    plt.ylabel('dimension2')
    plt.colorbar()
    plt.title('TSNE_conv_Visualization')
    if name==0:
        plt.savefig('LeNet_img/LeNet_TSNE_conv.png')
    else:
        plt.savefig('Mynet_img/Mynet_TSNE_conv.png')
    #plt.show()
    plt.clf()

    
    fcl_show=np.column_stack((fcl_show, np.array(label)))
    plt.scatter(fcl_show[:, 0], fcl_show[:, 1], c=fcl_show[:, 2], cmap=cmap)
    plt.xlabel('dimension1')
    plt.ylabel('dimension2')
    plt.colorbar()
    plt.title('TSNE_fcl_Visualization')
    if name==0:
        
        plt.savefig('LeNet_img/LeNet_TSNE_fcl.png')
    else:
        
        plt.savefig('Mynet_img/Mynet_TSNE_fcl.png')
    #plt.show()
    plt.clf()

    
    output_show=np.column_stack((output_show, np.array(label)))
    plt.scatter(output_show[:, 0], output_show[:, 1], c=output_show[:, 2], cmap=cmap)
    plt.xlabel('dimension1')
    plt.ylabel('dimension2')
    plt.colorbar()
    plt.title('TSNE_output_Visualization')
    if name==0:
        
        plt.savefig('LeNet_img/LeNet_TSNE_output.png')
    else:
        
        plt.savefig('Mynet_img/Mynet_TSNE_output.png')
    #plt.show()
    plt.clf()

if __name__ == '__main__':
    ######################## Get train/test dataset ########################
    X_train, X_test, Y_train, Y_test = get_data('dataset')
    ########################################################################
    # 以上加载的数据为 numpy array格式
    # 如果希望使用pytorch或tensorflow等库，需要使用相应的函数将numpy arrray转化为tensor格式
    # 以pytorch为例：
    #   使用torch.from_numpy()函数可以将numpy array转化为pytorch tensor
    #
    # Hint:可以考虑使用torch.utils.data中的class来简化划分mini-batch的操作，比如以下class：
    #   https://pytorch.org/docs/stable/data.html#torch.utils.data.TensorDataset
    #   https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    ########################################################################

    ########################################################################
    ######################## Implement you code here #######################
    ########################################################################
    indices = [np.where(Y_train == i)[0] for i in range(10)]
    samples=[]#训练样本数据
    label=[]#训练样本标签
    for i in indices:
        for j in i[:30]:
            samples.append(X_train[j])
            label.append(Y_train[j])
    samples=torch.tensor(samples)

    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)
    Y_train = torch.from_numpy(Y_train)
    Y_test = torch.from_numpy(Y_test)

    # 使用 TensorDataset 将数据集封装成一个数据集对象
    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)

    train_set = DataLoader(train_dataset, batch_size=64)
    test_set = DataLoader(test_dataset, batch_size=64)

    model = LeNet.LeNet()
    
    model.load_state_dict(torch.load('LeNet_model.pt'))      #可以从这里加载已经训练好的模型！！！！！
    
    Loss_function = torch.nn.CrossEntropyLoss()#定义损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#定义优化器
    
    #model_training(0,model, Loss_function, optimizer,50,train_set,test_set)#如果不加载模型在这里进行训练！！！！！
    conv=torch.empty((0, 1600))
    fcl=torch.empty((0,84))
    output=torch.empty((0,10))
    for img in samples:#提取网络中层特征
        conv2_output,fcl2_output,x=model.visualize(img)
        a = torch.flatten(conv2_output, start_dim=0)
        b= torch.flatten(fcl2_output, start_dim=0)
        c= torch.flatten(x,start_dim=0)

        conv = torch.cat((conv, a.unsqueeze(0)), dim=0)
        fcl= torch.cat((fcl,b.unsqueeze(0)),dim=0)
        output = torch.cat((output,c.unsqueeze(0)), dim=0)
    conv=conv.detach().numpy()
    fcl=fcl.detach().numpy()
    output=output.detach().numpy()

    PCA_tSNE_show(0, label, conv,fcl, output)#进行PCA  t_SNE可视化


    model= Mynet.Mynet()
    model.load_state_dict(torch.load('Mynet_model.pt'))   #可以从这里加载已经训练好的模型！！！！！
    Loss_function = torch.nn.CrossEntropyLoss()#定义损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#定义优化器
    
    #model_training(1,model, Loss_function, optimizer,50,train_set,test_set)#如果不加载模型在这里进行训练！！！！！
    conv=torch.empty((0, 1600))
    fcl=torch.empty((0,84))
    output=torch.empty((0,10))
    for img in samples:#提取网络中层特征
        conv2_output,fcl2_output,x=model.visualize(img)
        a = torch.flatten(conv2_output, start_dim=0)
        b= torch.flatten(fcl2_output, start_dim=0)
        c= torch.flatten(x,start_dim=0)

        conv = torch.cat((conv, a.unsqueeze(0)), dim=0)
        fcl= torch.cat((fcl,b.unsqueeze(0)),dim=0)
        output = torch.cat((output,c.unsqueeze(0)), dim=0)
    conv=conv.detach().numpy()
    fcl=fcl.detach().numpy()
    output=output.detach().numpy()

    #进行PCA 或 t_SNE可视化
    PCA_tSNE_show(1, label, conv, fcl, output)