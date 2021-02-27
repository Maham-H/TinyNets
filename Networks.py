### Provides architecture of networks used in the training
import torch
import torch.nn as nn

### Multi Network architecture for sub networks
class section_NN(nn.Module):
    def __init__(self, config):
        super(section_NN, self).__init__()
        self.inputSize = config["input_dim"]
        self.outputSize = config["output_dim"]
        self.hiddenSize1 = config["hidden1"] 
        self.hiddenSize2 = config["hidden2"] 
        #self.d1 = config["drop1"]

        
        self.h1 = nn.Linear(self.inputSize, self.hiddenSize1)
        #self.d1 = nn.Dropout(p=0.5)
        self.h2 = nn.Linear(self.hiddenSize1,self.hiddenSize2)
        self.l3 = nn.LeakyReLU(0.1)
        self.h3 = nn.Linear(self.hiddenSize2,self.outputSize)
    def forward(self,x):
        x = self.l3(self.h1(x))
        x = self.l3(self.h2(x))
        x = self.h3(x)
        return x
    


### Single Network architecture

class Large_NN(nn.Module):
    def __init__(self, config):
        super(Large_NN, self).__init__()
        self.inputSize = config["input_dim"]
        self.outputSize = config["output_dim"]
        self.hiddenSize1 = config["hidden1"] 
        self.hiddenSize2 = config["hidden2"] 
        #self.d1 = config["drop1"]

        
        self.h1 = nn.Linear(self.inputSize, self.hiddenSize1)
        #self.d1 = nn.Dropout(p=0.5)
        self.h2 = nn.Linear(self.hiddenSize1,self.hiddenSize2)
        self.l3 = nn.LeakyReLU(0.1)
        self.h3 = nn.Linear(self.hiddenSize2,self.outputSize)

    def forward(self,x):
        
        #x = torch.relu(self.h1(x))
        #x = torch.relu(self.h2(x))
        #x = torch.relu(self.h3(x))
        x = self.l3(self.h1(x))
        x = self.l3(self.h2(x))
        x = self.h3(x)
        return x
    
    
    
### TinyNet architecture for picking the right section for data

class tiny_NN(nn.Module):
    def __init__(self, config):
        super(tiny_NN, self).__init__()
        self.inputSize = config["input_dim"]
        self.outputSize = config["output_dim"]
        self.hiddenSize1 = config["hidden1"] 
        self.hiddenSize2 = config["hidden2"] 
        #self.d1 = config["drop1"]

        
        self.h1 = nn.Linear(self.inputSize, self.hiddenSize1)
        #self.d1 = nn.Dropout(p=0.5)
        self.h2 = nn.Linear(self.hiddenSize1,self.hiddenSize2)
        self.l3 = nn.LeakyReLU(0.1)
        self.h3 = nn.Linear(self.hiddenSize2,self.outputSize)

    def forward(self,x):
        
        #x = torch.relu(self.h1(x))
        #x = torch.relu(self.h2(x))
        #x = torch.relu(self.h3(x))
        x = self.l3(self.h1(x))
        x = self.l3(self.h2(x))
        x = self.h3(x)
        return x
