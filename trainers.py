
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from Networks import Large_NN,tiny_NN, section_NN
import helperfunc as hf
### for each epoch returns loss of batch

def train_epoch(NN,optimizer,criterion,perm,batch_size,X_train,y_train,i):
    optimizer.zero_grad()
    indices = perm[i:i+batch_size]
    batch_x, batch_y = X_train[indices], y_train[indices]

    # in case you wanted a semi-full example
    outputs = NN.forward(batch_x)

    loss = criterion(outputs, batch_y)
    loss.backward()
    optimizer.step()
    return loss.item()



def trainsingle(config, X_train, y_train):
    N,D= np.shape(X_train)
    batch_size = config["batch_size"]
    iters = config["iters"]
    epochs = config["epochs"]
       
    NN = Large_NN(config)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(NN.parameters(), lr=config["lr"], betas=(0.9, 0.99), 
                           eps=1e-08,weight_decay=0.0, amsgrad=False)
        
    X_train = torch.tensor(X_train, dtype=torch.float) #  tensor
    Nb = N
        
    y_train = torch.tensor(y_train, dtype=torch.float)#  tensor
    try:
        Dy=y_train.size()[1]
    except:
        Dy=1

    y_train= torch.reshape(y_train,(N,Dy))

    print("starting training")
   
    for k in range(int(epochs)):
        running_loss =0.0
        permutation = torch.randperm(X_train.size()[0])

        for j in range(iters):
            running_loss += train_epoch(NN,optimizer,criterion,
                                        permutation,batch_size,X_train,y_train,k)
    
        if (k % config["display_interval"] == 0) and (config["print_it"]):
            print('iter : ',int(k),'\t train loss :'+"{:.3f}".format(running_loss/iters))#,

    outputs = NN(X_train)
    loss = criterion(outputs, y_train).item()
        
    return NN, criterion,loss


def traintiny(config, X_train, y_train):
    N,D= np.shape(X_train)
    batch_size = config["batch_size"]
    iters = config["iters"]
    epochs =config["epochs"]
        
    print("starting training")
        
    NN = tiny_NN(config)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(NN.parameters(), lr=config["lr"], betas=(0.9, 0.99), 
                           eps=1e-08,weight_decay=0.0, amsgrad=False)

    X_train = torch.tensor(X_train, dtype=torch.float) #  tensor
    Nb = N
        
    y_train = torch.tensor(y_train, dtype=torch.float)# 15k, tensor
    try:
        Dy=y_train.size()[1]
    except:
        Dy=1

    y_train= torch.reshape(y_train,(N,Dy))

        
    for k in range(int(epochs)):
        running_loss =0.0
        permutation = torch.randperm(X_train.size()[0])

        for j in range(iters):
            running_loss += hf.train_epoch(NN,optimizer,criterion,
                                        permutation,batch_size,X_train,y_train,k)
    
        if (k % config["display_interval"] == 0) and (config["print_it"]):
            print('iter : ',int(k),'\t train loss :'+"{:.3f}".format(running_loss/iters))#,

    outputs = NN(X_train)
    loss = criterion(outputs, y_train).item()
        
    return NN,criterion,loss

def trainsect(config, In, out,Sn,o_n,bn):
        
    N,D= np.shape(In)
    c=config["c"]
    dim=config["dim"]
    batch_size = config["batch_size"]
    iters = config["iters"]
    epochs = config["epochs"]
    criterion = nn.MSELoss()
    
    all_in = {}
    all_out = {}
    NN ={}
    loss=np.zeros(Sn)
    outputs = {}
    
    
    for i in range(Sn):

        all_in["batch_for_sec_"+str(i)]=[]
        all_out["batch_for_sec_"+str(i)]=[]
        for n in range(N):
            r = hf.filt(In[n,dim],o_n[i+1],bn[i],c)
            if r >= 0.5:
                all_in["batch_for_sec_"+str(i)].append(r*In[n])
                all_out["batch_for_sec_"+str(i)].append(out[n])
                
        print("starting training for section: ",i)
        
        NN[str(i)] = section_NN(config)
        optimizer = optim.Adam(NN[str(i)].parameters(), lr=config["lr"], betas=(0.9, 0.99), 
                           eps=1e-08,weight_decay=0.0, amsgrad=False)

        X_train = all_in["batch_for_sec_"+str(i)]
        y_train = all_out["batch_for_sec_"+str(i)]
        
        X_train = torch.tensor(X_train, dtype=torch.float) #  tensor
        Nb = X_train.size()[0]
        
        y_train = torch.tensor(y_train, dtype=torch.float)# 15k, tensor
        try:
            Dy=y_train.size()[1]
        except:
            Dy=1

        y_train= torch.reshape(y_train,(Nb,Dy))

        for k in range(int(epochs)):
            running_loss =0.0
            permutation = torch.randperm(X_train.size()[0])

            for j in range(iters):
                running_loss += hf.train_epoch(NN[str(i)],optimizer[str(i)],criterion[str(i)],
                                            permutation,batch_size,X_train,y_train,k)
    
            if (k % config["display_interval"] == 0) and (config["print_it"]):    # print every display_interval=100 mini-batches
                print('iter : ',int(k),'\t train loss :'+"{:.3f}".format(running_loss/iters))#,
        net = NN[str(i)]
        outputs[str(i)] = net(X_train)
        loss[i] = criterion(outputs[str(i)], y_train).item()

        
        print('data points for section:'+str(i)+' are: ',str(Nb))

    return NN, criterion,all_in,all_out,loss



