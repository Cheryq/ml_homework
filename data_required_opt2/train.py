import os
import numpy as np
import LeNet
import torch
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

def model_training(name,model,Loss_function,optimizer,epoch_time,train_set,test_set):
    train_loss_list=[]
    train_accuracy_list=[]
    test_loss_list=[]
    test_accuracy_list=[]
    X_epoch_list=[]
    for epoch in range(1, epoch_time):
        loss,acc=LeNet.train(model, train_set, optimizer, Loss_function, epoch)
        train_loss_list.append(loss)
        train_accuracy_list.append(acc)

        loss,acc=LeNet.test(model, test_set, Loss_function)
        test_loss_list.append(loss)
        test_accuracy_list.append(acc)

        X_epoch_list.append(epoch)
    
    plt.plot(X_epoch_list, train_loss_list, 'b', label='Training Loss')
    plt.plot(X_epoch_list, test_loss_list, 'r', label='Test Loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss over Epochs')

    if name==0:
        plt.savefig('LeNet_img/LeNet_Loss.png')
    else:
        plt.savefig('Mynet_img/Mynet_Loss.png')
    plt.clf()

    # 绘制训练准确率和测试准确率曲线
    plt.plot(X_epoch_list, train_accuracy_list, 'b', label='Training Accuracy')
    plt.plot(X_epoch_list, test_accuracy_list, 'r', label='Test Accuracy')
    plt.legend(loc='lower right')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy over Epochs')

    if name==0:
        plt.savefig('LeNet_img/LeNet_Accuracy.png')
    else:
        plt.savefig('Mynet_img/Mynet_Accuracy.png')
   
    plt.clf()
    # 保存模型
    if name==0:
        torch.save(model.state_dict(), 'LeNet_model.pt')
    else:
        torch.save(model.state_dict(), 'Mynet_model.pt')