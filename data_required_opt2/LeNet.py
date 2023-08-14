import torch.nn as nn
import torch.nn.functional as F
import torch
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  
        self.pool1 = nn.MaxPool2d(2) 
        self.conv2 = nn.Conv2d(6, 16, 5)  
        self.pool2 = nn.MaxPool2d(2) 
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  
        self.fc2 = nn.Linear(120, 84)   
        self.fc3 = nn.Linear(84, 10)   

    def forward(self, img):
        conv1_output =F.relu(self.conv1(img))
        pool1_output= self.pool1(conv1_output) 
        conv2_output =F.relu(self.conv2(pool1_output))
        pool2_output= self.pool2(conv2_output)
        x = pool2_output.view(-1, 16 * 5 * 5)  
        x = F.relu(self.fc1(x))      
        x = F.relu(self.fc2(x))     
        x = self.fc3(x)            
        return x
    def visualize(self,img):
        conv1_output =F.relu(self.conv1(img))
        pool1_output= self.pool1(conv1_output) 
        conv2_output =F.relu(self.conv2(pool1_output))
        pool2_output= self.pool2(conv2_output)
        x = pool2_output.view(-1, 16 * 5 * 5)  
        x = F.relu(self.fc1(x))      
        fcl = F.relu(self.fc2(x))     
        x = self.fc3(fcl)
        return conv2_output,fcl,x
def train(model, train_loader, optimizer, criterion, epoch):
    model.train()  
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()#将所有参数的梯度清零
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total

    print('epoch {} Training loss: {}, accuracy: {}\n'.format(epoch,
        train_loss, accuracy))
    return train_loss, accuracy
def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print('Test loss: {}, accuracy: {}\n'.format(
        test_loss,  accuracy))
    return test_loss, accuracy