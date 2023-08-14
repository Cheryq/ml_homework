import os
import numpy as np
from dataset import get_data,normalize
from VAE import VAE,train,Encoder,Decoder
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
def show_img(img):
    image = img.squeeze().permute(1, 2, 0).detach().numpy()
    plt.imshow(image)
    plt.axis('off')
    plt.show()
if __name__ == '__main__':
    ######################## Get train dataset ########################
    X_train = get_data('dataset')
    ########################################################################
    ######################## Implement you code here #######################
    ########################################################################
    train_loader = DataLoader(X_train, batch_size=32)
    # 初始化模型和优化器
    latent_dim = 64  # 潜在空间的维度
    model = VAE(latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #train(model, optimizer, train_loader, 100)
    model.load_state_dict(torch.load("VAE.pth"))
    alpha_val=[0.2,0.4,0.6,0.8]
    male=[23,14,27,32,35,116,54]
    female=[51,12,18,44,45,77,80]
    for i in range(len(male)):

        X1=torch.from_numpy(X_train[male[i]])
        X2=torch.from_numpy(X_train[female[i]])
    
        mu,var=model.encoder(X1)
        z1=model.reparameterize(mu, var)
        mu,var=model.encoder(X2)
        z2=model.reparameterize(mu, var)
        images_=[]
        image=model.decoder(z2)
        
        img_male=X_train[male[i]].copy()
        img_female=X_train[female[i]].copy()
        img_male=img_male.transpose(1,2,0)
        img_female=img_female.transpose(1,2,0)
        images_.append(img_female)
        image = image.squeeze().permute(1, 2, 0).detach().numpy()
        images_.append(image)
        for alpha in alpha_val:
            z=alpha*z1+(1-alpha)*z2
            Img_from_z=model.decoder(z)
            image = Img_from_z.squeeze().permute(1, 2, 0).detach().numpy()
            images_.append(image)
        image=model.decoder(z1)
        image = image.squeeze().permute(1, 2, 0).detach().numpy()
        images_.append(image)
        images_.append(img_male)
        num_images = len(images_)
        

        fig, axes = plt.subplots(1, num_images, figsize=(12, 4))

        # 在每个子图中显示图像
        for j, image in enumerate(images_):
            axes[j].imshow(image)
            axes[j].axis('off')

        # 调整子图之间的间距
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig('img/{}.png'.format(i))
        # 显示图像
        #plt.show()
        

            
