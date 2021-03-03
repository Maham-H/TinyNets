import numpy as np

#filter and derivatives

def highpass(data,offset,band,slope=10):
    return np.nan_to_num(1/(1+ np.exp(slope*(data-(offset+band)))), copy=True, nan=0.0, posinf=500, neginf=-500)

def lowpass(data,offset,band,slope=10):
    return np.nan_to_num(1/(1+ np.exp(slope*(-data+offset))), copy=True, nan=0.0, posinf=500, neginf=-500)

def filt(x,o,b,c=10):
    hps = highpass(x,o,b,c)
    lps = lowpass(x,o,b,c)
    return (hps+lps)-1

def d_hps(x,o,b,c=10):
    hps =  highpass(x,o,b,c)
    return np.nan_to_num((hps)*(1-hps), copy=True, nan=0.0, posinf=500, neginf=-500)

def d_lps(x,o,b,c=10):
    lps =  lowpass(x,o,b,c)
    return np.nan_to_num(-(lps)*(1-lps), copy=True, nan=0.0, posinf=500, neginf=-500)    

##################################################################################

##################################################################################

def variance(x,b,o,c=10):
    return np.var(x*filt(x,o,b,c))

def der_var(In):
    N,D = np.shape(In)
    mu = np.mean(In,0)
    der = In-mu

    der = 2*der/(D*N) #variance across all the other dimensions
    return der
      
###################################################################################
def forwardMSE(net,In,out):
    weights ={}
    i=0
    j=0
    k=0
    lrelu=-0.1
    for param in net.parameters():
        if i%2==0:
            
            weights['W'+str(j)]=param.cpu().data.detach().numpy()
            j+=1
        else:
            weights['b'+str(k)]=param.cpu().data.detach().numpy()
            k+=1
        i+=1

    W1 = weights['W0'] # h1xin
    W2 = weights['W1'] # h2xh1
    W3 = weights['W2'] # outxh2
    b1 = weights['b0'] # h1
    b2 = weights['b1'] # h2
    b3 = weights['b2'] # out
    
    N,D = np.shape(In)
    
    y = np.dot(In,np.transpose(W1)) + b1 #Nxh1
    yr = np.maximum(lrelu,y) # Nxh1
    y2 = np.dot(yr,np.transpose(W2)) + b2 # Nxh2
    y2r = np.maximum(lrelu,y2)
    y3 = np.dot(y2r,np.transpose(W3)) + b3 # Nxout
    y3r = np.reshape(np.maximum(lrelu,y3),np.shape(out)) # Nxout
    MSE = np.sum((out-y3r)**2,0)/N

    y[y<=lrelu]=0
    y[y>lrelu]=1
    y2[y2<=lrelu] = 0
    y2[y2>lrelu] = 1
    y3[y3<=lrelu] = 0
    y3[y3>lrelu] = 1
    
    return MSE,weights, y,y2,y3,y3r

def MSE_der(net,In,out,config):
    N,D = np.shape(In)
    c = config["c"]
    
    MSE,weights,yd,y2d,y3d,ypred = forwardMSE(net,In,out)
    W1 = weights['W0'] # h1xin
    W2 = weights['W1'] # h2xh1
    W3 = weights['W2'] # outxh2
    b1 = weights['b0'] # h1
    b2 = weights['b1'] # h2
    b3 = weights['b2'] # out
    der = out-ypred
    der = np.reshape(der,np.shape(y3d))*y3d
    der = np.dot(der,W3)
    der = der*y2d
    der = np.dot(der,W2)
    der = der*yd
    der = np.dot(der,W1)
    der = der*In
    
    return der#[dim] #1xin
####################################################################################

def loss1_der(In,all_in,all_out,b,o,dim,sn,NN,config,lossi,mse):
    phi_1 = config["phi_1"]
    phi_2 = config["phi_2"]
    phi_3 = config["phi_3"]
    grad={}
    ref_s=config["ref_s"]
    vref = config["ref_v"]
    dim = config["dim"]
    lt = np.min(In[:,dim])
    ut = np.max(In[:,dim])
    
    ds = np.sign(np.log(sn/ref_s))*ref_s/sn#der_select(In,dim,b,sn)
    ds = -ds*(ut-lt)/(np.size(b))
    tot=0

    dv ={}
    dmse={}
    
    out = np.array(all_out["batch_for_sec_"+str(0)])
    c=config["c"]

    for i in range(sn):
        
        In = np.array(all_in["batch_for_sec_"+str(i)])
        out = np.array(all_out["batch_for_sec_"+str(i)])
        net = NN[str(i)]

        v = variance(In,b[i],o[i+1],c)
        v = np.sign(v-vref/2)

        dv[str(i)] = v*der_var(In) #shape of input

        ms=np.sign(lossi[i]-mse/sn)
        dmse[str(i)] = ms*MSE_der(net,In,out,config)
  
    for i in range(sn):
        In = np.array(all_in["batch_for_sec_"+str(i)])

        dbev = 0
        dbem = 0
        dbv = np.sum(d_hps(In,b[i],o[i+1],c)*dv[str(i)])
        dbm = np.sum(d_hps(In,b[i],o[i+1],c)*dmse[str(i)])
        #for j in range(i+1,sn):
        #    Inn = np.array(all_in["batch_for_sec_"+str(j)])
        #    dbev += np.sum((d_hps(Inn,b[j],o[j+1],c)+d_lps(Inn,b[j],o[j+1],c))*dv[str(j)])
        #    dbem += np.sum((d_hps(Inn,b[j],o[j+1],c)+d_lps(Inn,b[j],o[j+1],c))*dmse[str(j)])
        dbtv =  dbv + dbev
        dbtm = dbm + dbem
        grad['bn'+str(i)] =-( phi_3*dbtm + phi_1*dbtv + phi_2*ds)
    
    return grad,dbtm,dbtv,ds
    
def loss_1(all_in,all_out,sn,b,o,config,NN,lossi,mse):
    
    var = np.zeros(sn)
    out = np.array(all_out["batch_for_sec_"+str(0)])
    N,Dy=np.shape(out)
    
    c = config["c"]
    phi_1=config["phi_1"]
    phi_2=config["phi_2"]
    phi_3=config["phi_3"]
    
    bav=np.mean(b)
    
    sref = config["ref_s"]
    vref = config["ref_v"]
    
    sloss = np.abs(np.log(sn/sref)) # between -1.5 to 1
    
    
    for i in range(sn):
        net=NN[str(i)]
        In = np.array(all_in["batch_for_sec_"+str(i)])
        out = np.array(all_out["batch_for_sec_"+str(i)])
        var[i] = variance(In,b[i],o[i+1],c)
        
    mse= np.abs(np.mean(lossi)-(mse/sn))
    var_loss = np.sum(np.abs((var-vref/2))/sn)
    loss1 = phi_3*mse + phi_1*var_loss + phi_2*sloss
    return loss1,mse,var_loss,sloss
      
###################################################################################

####################################################################################
def update_b(In,all_in,all_out,b,o,sn,NN,config,lossi,mse,thresh=0.2):
    dim = config["dim"]
    beta = config["beta"]
    bnew = np.zeros(sn)
    grad,dbtm,dbtv,ds =loss1_der(In,all_in,all_out,b,o,dim,sn,NN,config,lossi,mse)
    for i in range(sn):
        bnew[i] = b[i] + beta*grad["bn"+str(i)]
        if bnew[i]<=thresh:
            bnew[i]=thresh
    return b,bnew,dbtm,dbtv,ds

def gen_s_b_o(bold,b,o,config,In,thresh=0.2):
    Sn = np.size(bold)
    dim=config["dim"]
    l_t = np.min(In[:,dim]) 
    u_t = np.max(In[:,dim])
    lim_d = u_t-l_t
    bav_new = (1/Sn)*np.sum(b)
    snew = int(np.ceil(lim_d/bav_new))
    bnew = np.zeros(snew)
    onew = np.zeros(snew+1) 
    
    boffs = l_t # offsets for o
    
    for i in range(snew):
        try:
            bnew[i]=b[i]
            onew[i+1] = onew[i]+boffs
            boffs=bnew[i]
        except:
            lim_new = lim_d-np.sum(b)
            bnew[i] = lim_new/(snew-Sn)
            if bnew[i]<bav_new:
                bnew[i]=bav_new
            elif bnew[i]>lim_d:
                bnew[i]=lim_d
            onew[i+1] = onew[i]+boffs
            boffs=bnew[i]
            
    if bnew[snew-1]+onew[snew]>u_t:
        bnew[snew-1] = u_t -onew[snew]
        if bnew[snew-1]<thresh:
            
            bnew = bnew[0:snew-1]
            onew = onew[0:snew]
            snew -=1
            
    return snew,bnew,onew
###################################################################################
def generate_rdata(SN,o,b,In,dim):
    limits = np.zeros((SN,2))
    N,D=np.shape(In)
    In_new=In[:,dim]
    labels=np.zeros(N)
    for i in range(SN):
        limits[i,0]= o[i+1]
        limits[i,1]= o[i+1]+b[i]

        for j in range(N):
            if (In_new[j] > limits[i,0]) and (In_new[j] < limits[i,1]):
                labels[j] = i

    return np.reshape(In_new,(N,1)),np.reshape(labels,(N,1))
            
####################################################################################

def init_s(Sn,dim,In):   
    cd = In[:,dim]
    lim_d = np.max(cd)- np.min(cd)
    bnew = np.zeros(Sn)
    onew = np.zeros(Sn+1)
    
    onew[0]=np.min(cd)
    b=0
    
    for i in range(Sn):
        bnew[i] = lim_d/Sn
        onew[i+1] = onew[i]+b
        b=bnew[i]
    return onew,bnew



##################################################################################
def corr(In,out,Dx,Dy):
    corr=np.zeros((Dx,Dy))
    for i in range(Dx):
        for j in range(Dy):
            a=In[:,i]
            v=out[:,j]

            a = (a - np.mean(a)) / (np.std(a) * len(a))

            v = (v - np.mean(v)) /  np.std(v)
            corr[i,j]=np.correlate(a,v)
    #from sklearn.preprocessing import normalize
    #print(normalize(out))
    cor=np.sum(np.abs(corr),1)
    return cor
##########################Calculating correlation#################################
