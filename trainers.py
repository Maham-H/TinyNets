
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from Networks import Large_NN,tiny_NN, section_NN
import helperfunc as hf
### for each epoch returns loss of batch

def train_epoch(NN,optimizer,criterion,perm,batch_size,X_train,y_train,i,device):
    optimizer.zero_grad()
    indices = perm[i:i+batch_size]
    batch_x, batch_y = X_train[indices].to(device), y_train[indices].to(device)

    # in case you wanted a semi-full example
    outputs = NN.forward(batch_x)

    loss = criterion(outputs, batch_y)
    loss.backward()
    optimizer.step()
    return loss.item()



def trainsingle(config, X_train, y_train,X_val,y_val):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    N,D= np.shape(X_train)
    batch_size = config["batch_size"]
    iters = config["iters"]
    epochs = config["epochs"]
       
    NN = Large_NN(config)
    NN.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(NN.parameters(), lr=config["lr"], betas=(0.9, 0.99), 
                           eps=1e-08,weight_decay=0.0, amsgrad=False)
        
    X_train = torch.tensor(X_train, dtype=torch.float) #  tensor
    y_train = torch.tensor(y_train, dtype=torch.float)#  tensor
    X_val = torch.tensor(X_val, dtype=torch.float).to(device) #  tensor
    y_val = torch.tensor(y_val, dtype=torch.float).to(device)#  tensor
    
    Nb = N
    train_loss =[]
    val_loss = []
    
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
                                        permutation,batch_size,X_train,y_train,k,device)
    
        if (k % config["display_interval"] == 0) and (config["print_it"]):
            print('iter : ',int(k),'\t train loss :'+"{:.3f}".format(running_loss/iters))#,
            
        train_loss.append(running_loss/iters)
        outputs = NN(X_val)
        val_loss.append(criterion(outputs, y_val).item())
    outputs = NN(X_train.to(device))
    loss = criterion(outputs, y_train.to(device)).item()
        
    return NN, criterion,loss,train_loss,val_loss


def traintiny(config, X_train, y_train, X_val, y_val):
    N,D= np.shape(X_train)
    batch_size = config["batch_size"]
    iters = config["iters"]
    epochs =config["epochs"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("starting training")
        
    NN = tiny_NN(config)
    NN.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(NN.parameters(), lr=config["lr"], betas=(0.9, 0.99), 
                           eps=1e-08,weight_decay=0.0, amsgrad=False)

    X_train = torch.tensor(X_train, dtype=torch.float) #  tensor
    y_train = torch.tensor(y_train, dtype=torch.float)#  tensor
    X_val = torch.tensor(X_val, dtype=torch.float).to(device) #  tensor
    y_val = torch.tensor(y_val, dtype=torch.float).to(device)#  tensor
    
    Nb = N
    train_loss =[]
    val_loss = []
    try:
        Dy=y_train.size()[1]
    except:
        Dy=1

    y_train= torch.reshape(y_train,(N,Dy))

        
    for k in range(int(epochs)):
        running_loss =0.0
        permutation = torch.randperm(X_train.size()[0])

        for j in range(iters):
            running_loss += train_epoch(NN,optimizer,criterion,
                                        permutation,batch_size,X_train,y_train,k,device)
    
        if (k % config["display_interval"] == 0) and (config["print_it"]):
            print('iter : ',int(k),'\t train loss :'+"{:.3f}".format(running_loss/iters))#,
        train_loss.append(running_loss/iters)
        outputs = NN(X_val)
        val_loss.append(criterion(outputs, y_val).item())
        
      
    return NN,criterion, train_loss,val_loss

def trainsect(config, In, out,Sn,o_n,bn, X_val, y_val):
        
    N,D= np.shape(In)
    c=config["c"]
    dim=config["dim"]
    batch_size = config["batch_size"]
    iters = config["iters"]
    epochs = config["epochs"]
    criterion = nn.MSELoss()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    all_in = {}
    all_out = {}
    val_in = {}
    val_out = {}
    train_loss ={}
    val_loss = {}
    NN ={}
    loss=np.zeros(Sn)

    
    for i in range(Sn):
        train_loss[str(i)] = []
        val_loss[str(i)] = []
        print("sorting data for section: ",i)
        r = np.array(hf.filt(In[:,dim],o_n[i+1],bn[i],c))
        all_in["batch_for_sec_"+str(i)]=In[r>=0.5]
        all_out["batch_for_sec_"+str(i)]=out[r>=0.5]
        
        r = np.array(hf.filt(X_val[:,dim],o_n[i+1],bn[i],c))
        val_in["batch_for_sec_"+str(i)]=X_val[r>=0.5]
        val_out["batch_for_sec_"+str(i)]=y_val[r>=0.5]
        #idx = np.where(a > 4)[0]

        print("starting training for section: ",i)
        
        NN[str(i)] = section_NN(config)
        NN[str(i)].to(device)
        optimizer = optim.Adam(NN[str(i)].parameters(), lr=config["lr"], betas=(0.9, 0.99), 
                           eps=1e-08,weight_decay=0.0, amsgrad=False)

        X_train = all_in["batch_for_sec_"+str(i)]
        y_train = all_out["batch_for_sec_"+str(i)]
        
        X_train = torch.tensor(X_train, dtype=torch.float) #  tensor
        y_train = torch.tensor(y_train, dtype=torch.float)# tensor
        
        X_val1 = val_in["batch_for_sec_"+str(i)]
        y_val1 = val_out["batch_for_sec_"+str(i)]
        
        X_val1 = torch.tensor(X_val1, dtype=torch.float).to(device) #  tensor
        y_val1 = torch.tensor(y_val1, dtype=torch.float).to(device) # tensor
        Nb = X_train.size()[0]
        
        
        try:
            Dy=y_train.size()[1]
        except:
            Dy=1

        y_train= torch.reshape(y_train,(Nb,Dy))

        for k in range(int(epochs)):
            running_loss =0.0
            permutation = torch.randperm(X_train.size()[0])

            for j in range(iters):
                running_loss += train_epoch(NN[str(i)],optimizer,criterion,
                                            permutation,batch_size,X_train,y_train,k,device)
    
            if (k % config["display_interval"] == 0) and (config["print_it"]): # print every display_interval=100 mini-batches
                print('iter : ',int(k),'\t train loss :'+"{:.3f}".format(running_loss/iters))#,
            train_loss[str(i)].append(running_loss/iters)
            outputs = NN[str(i)](X_val1)
            val_loss[str(i)].append(criterion(outputs, y_val1).item())
            
        net = NN[str(i)].to(device)
        outputs = net(X_train.to(device))
        loss[i] = criterion(outputs, y_train.to(device)).item()

        
        print('data points for section:'+str(i)+' are: ',str(Nb))

    return NN, criterion,all_in,all_out,loss,train_loss,val_loss



