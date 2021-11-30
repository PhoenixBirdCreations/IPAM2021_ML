#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 13:13:59 2021

@author: mberbel
"""
import numpy as np
import csv
import matplotlib.pyplot as plt
#%%
def complicated_f(vector):
    m1=vector[0]; m2=vector[1]; s1x=vector[2]; s1y=vector[3]; s1z=vector[4]; 
    s2x=vector[5]; s2y=vector[6]; s2z=vector[7]
    angle=vector[8]; q=vector[9]; chirp=vector[10]
    
    c0=m1*np.abs(s1x)+s2y**2+m2/np.sqrt(chirp)+(1-q)/(s1z+3) #remains positive
    c1=m2**3*(chirp+np.pi-s2z)/(5-s2x+s1y) #remains positive
    c2=np.sqrt((s2x-s1x)**2+(s1z-s1y)**2) #below 1
    c3=-np.sqrt((s2y-s1y)**2+(s1x-s1z)**2)
    c4=np.sqrt((s2z-s1z)**2+(s1y-s1x)**2)
    c5=-np.sqrt((s2x-s1x)**2+(s2z-s2y)**2)
    c6=np.sqrt((s2y-s1y)**2+(s2x-s2z)**2)
    c7=-np.sqrt((s2z-s1z)**2+(s2y-s2x)**2)
    c8=np.abs(angle-s1x*s2y*m2)/np.pi;      # remains positive
    c9=np.abs(np.sin(chirp*s2y-q+0.4*m2)) #below 1
    c10=chirp
    return [c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10]
    
def complicated_fv1(n,vector):
    m1=vector[0]; m2=vector[1]; s1x=vector[2]; s1y=vector[3]; s1z=vector[4]; 
    s2x=vector[5]; s2y=vector[6]; s2z=vector[7]
    angle=vector[8]; q=vector[9]; chirp=vector[10]
    
    if (n==1):
        c0=m1*np.abs(s1x)+s2y**2+m2/np.sqrt(chirp)+(1-q)/(s1z+3) #remains positive
        c1=m2**2*(chirp+np.pi-s2z)/(4-s2x+s1y) #remains positive
        c2=np.sqrt(np.abs((s2x-s1x)**2+(s1z-s1y)**2)) #below 1
        c3=-np.sqrt(np.abs((s2y-s1y)**2+(s1x-s1z)**2))
        c4=np.sqrt(np.abs((s2z-s1z)**2+(s1y-s1x)**2))
        c5=-np.sqrt(np.abs((s2x-s1x)**2+(s2z-s2y)**2))
        c6=np.sqrt(np.abs((s2y-s1y)**2+(s2x-s2z)**2))
        c7=-np.sqrt(np.abs((s2z-s1z)**2+(s2y-s2x)**2))
        c8=np.abs(angle-s1x*s2y*m2)/np.pi;      # remains positive
        c9=np.abs(np.sin(chirp*s2y-q+0.4*m2)) #below 1
        c10=chirp
    elif (n==2):
        c0=m1*np.abs(s1z)+s2y**2+m2/np.sqrt(chirp)+(1.5-q)/(s1z+2) #remains positive
        c1=m2**4*(m2+np.pi*0.5-s2z)/(5-s2x+s1y) #remains positive
        c2=np.sqrt(np.abs((s2y-s1y)**2+(s1z-s2y)**2)) #below 1
        c3=-np.sqrt(np.abs((s2z-s1z)**2+(s1x-s1z)**2))
        c4=np.sqrt(np.abs((s2x-s1x)**2+(s2y-s1x)**2))
        c5=-np.sqrt(np.abs((s2x-s1x)**3+(s2z-s2y)**2))
        c6=np.sqrt(np.abs((s2y-s1y)**2+(s2x-s2z)**3))
        c7=-np.sqrt(np.abs((s2z-s1z)**4+(s2y-s2x)))
        c8=np.abs(angle*0.5-s1x*s2y*m1)/np.pi;      # remains positive
        c9=np.abs(np.cos(chirp*s2y-q+0.4*m2)) #below 1
        c10=chirp+0.1
    elif (n==3):
        c0=m2*np.abs(s1x)+s2y**2+m2/np.sqrt(chirp)+(angle)/(s1z+3) #remains positive
        c1=m2**3*(m1+np.pi-s2z)/(6-s2z-s1y) #remains positive
        c2=np.sqrt(np.abs((s1z-s1y)**2+(s1z-s1y)**2)) #below 1
        c3=np.sqrt(np.abs((s2x-s1x)**2+(s2x-s1z)**2))
        c4=-np.sqrt(np.abs((s2y-s1x)**2+(s1y-s1x)**2))
        c5=-np.sqrt(np.abs((s2x-s1z)**2+(s2z-s2x)**2))
        c6=np.sqrt(np.abs((s2z-s1y)**2+(s2x-s2z)**2))
        c7=np.sqrt(np.abs((s2x-s1z)**2+(s2y-s1y)**2))
        c8=-np.abs(chirp-s1x*s2y*m1)/np.pi;      # remains positive
        c9=np.abs(np.sin(chirp*s1z-q+0.2*m1)) #below 1
        c10=chirp-0.05
    elif (n==4):
        c0=m1*m2*np.abs(s1x)+s2y**3+m2/np.sqrt(chirp)+(1-q)/(s1z+1.5) #remains positive
        c1=chirp**3*(m1-m2+np.pi-s2z)/(2.3-s2x+s1y) #remains positive
        c2=np.sqrt(np.abs((s2x-s1x)**2+(s1x-s1y)**5)) #below 1
        c3=np.sqrt(np.abs((s2y-s1x)**2+(s1y-s1x)))
        c4=np.sqrt(np.abs((s2z-s1y)**3+(s1y-s1x)**2))
        c5=np.sqrt(np.abs((s2z-s1x)**2+(s2x-s2y)**2))
        c6=np.sqrt(np.abs((s2y-s1y)**2+(s2x-s2y)**4))
        c7=np.sqrt(np.abs((s2x-s1z)**2+(s2y-s2x)**2))
        c8=np.abs(angle-s1x*s2y*m2);      # remains positive
        c9=np.abs(np.sin(angle*s2x+q+0.4*m2)) #below 1
        c10=chirp+s1x*0.5
    else:
        c0=chirp*np.abs(s1z)+s2y**3+m1/np.sqrt(m1)+(1-q)/(s1z+1.2) #remains positive
        c1=m2*(chirp+np.pi-s1x)/(5-s2z+s1x) #remains positive
        c2=-np.sqrt(np.abs((s2x-s1x)+(s1z-s1y)**2)) #below 1
        c3=-np.sqrt(np.abs((s2y-s1y)**2+(s1x-s1y)**2))
        c4=-np.sqrt(np.abs((s2x-s1z)**3+(s1y-s1x)**2))
        c5=-np.sqrt(np.abs((s2x-s1y)**2+(s2x-s2y)))
        c6=-np.sqrt(np.abs((s1y-s1y)**4+(s2x-s1y)**2))
        c7=-np.sqrt(np.abs((s2z-s1x)**2+(s2y-s2x)**3))
        c8=np.abs(q*chirp*angle-s1x*s2y*m1)/np.pi;      # remains positive
        c9=np.abs(np.sin(m1*s1z-q+0.2*m1)) #below 1
        c10=chirp-s2z*0.3
    return [c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10]

def complicated_fv2(n,vector):
    m1=vector[0]; m2=vector[1]; s1x=vector[2]; s1y=vector[3]; s1z=vector[4]; 
    s2x=vector[5]; s2y=vector[6]; s2z=vector[7]
    angle=vector[8]; q=vector[9]; chirp=vector[10]
    
    if (n==1):
        c0=m1*np.abs(s1x)+s2y**2+m2/np.sqrt(chirp)+(1-q)/(s1z+3) #remains positive
        c1=m1*chirp+np.cos(q) #remains positive
        c2=np.sqrt((s2x-s1x)**2+(s1z-s1y)**2) #below 1
        c3=-np.sqrt((s2y-s1y)**2+(s1x-s1z)**2)
        c4=np.sqrt((s2z-s1z)**2+(s1y-s1x)**2)
        c5=-np.sqrt((s2x-s1x)**2+(s2z-s2y)**2)
        c6=np.sqrt((s2y-s1y)**2+(s2x-s2z)**2)
        c7=-np.sqrt((s2z-s1z)**2+(s2y-s2x)**2)
        c8=np.abs(angle-s1x*s2y*m2)/np.pi;      # remains positive
        c9=np.abs(np.sin(chirp*s2y-q+0.4*m2)) #below 1
        c10=chirp
    elif (n==2):
        c0=chirp*np.abs(s1x)+q*angle #remains positive
        c1=m2**4*(m2+np.pi*0.5-s2z)/(5-s2x+s1y) #remains positive
        c2=(s1x+s2z-s1y**2)/chirp #below 1
        c3=-(s1z+s2y-s2x**2)/(chirp+m1)
        c4=(s1x+s2y-s1y**2)/(chirp**2)
        c5=-(s2x+s2z-s1y**2)/(chirp-1*m1)
        c6=(s1x+s2z-s2y**2)/(m1)
        c7=-(s1y+s2x-s1z**2)/(m1*m2**2)
        c8=np.abs(m1*m2-chirp);      # remains positive
        c9=np.exp(-q*s1z*s2y) #below 1
        c10=chirp+s1z*0.5
    elif (n==3):
        c0=s1x*s2z*s1y+chirp +m1*q #remains positive
        c1=m2+chirp+m1-angle #remains positive
        c2=angle*s1x**3 #below 1
        c3=(angle-q)*s2x**2
        c4=-q*angle*s1y**3
        c5=m2**2/m1**3*angle*s1z**4
        c6=m2*angle*s1x*s2y**3/chirp
        c7=-angle*(s1x+s2z)**3
        c8= np.pi-q**2+s1x     # remains positive
        c9= m2/m1**3#below 1
        c10=chirp+angle*0.3
    elif (n==4):
        c0=m1**2-q+angle/chirp +s1z #remains positive
        c1=np.sqrt(chirp)+m1**2/angle**3 #remains positive
        c2=-np.cos(q*s1z**2-s2x) #below 1
        c3=np.cos(q*s1z**2-s2x)/m2
        c4=-np.cos(q*s1z**4-s2z)
        c5=np.cos(q*s1z**2-s2y)/m1
        c6=-np.sin(q*s1y**3-s2x)
        c7=np.sin(q*s2x**2-s2x)*q
        c8= np.log(np.sqrt(m1)+np.abs(m2*s2z+s1z*q)+3)   # remains positive
        c9= np.exp(-s1x)#below 1
        c10=chirp+0.05
    else:
        c0=s1x*m1+s2z*m2+chirp+angle+np.abs(s2y+s1y) #remains positive
        c1=3*m2**2-np.log(chirp)-s1x*q #remains positive
        c2=np.exp(-q) #below 1
        c3=np.exp(-np.abs(np.sin(q*s2x**2-s2x)*q))
        c4=np.exp(-q/m2)
        c5=np.exp(-np.abs(angle))
        c6=np.exp(-m2/chirp)
        c7=np.exp(-q*s1x)
        c8= np.tan(-chirp)/np.pi     # remains positive
        c9= np.abs(np.sin(angle+chirp))#below 1
        c10=chirp-s1x*s2y*s1z*s1x*angle
    return [c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10]


def pipelinev0(y,errorpct=0.05):
    Nsample=len(y)
    dsample=len(y[0])
    
    x=np.zeros((Nsample,dsample));
    for i in range (0,Nsample):
        x[i]=complicated_f(y[i])
        for j in range(0, dsample): #plus some random error up to specified % of the value
            x[i][j]=x[i][j]+x[i][j]*(-1)**np.random.randint(1,3)*errorpct*np.random.rand()
    return x


def pipelinev1(y,errorpct=0.05):
    Nsample=len(y)
    dsample=len(y[0])
    
    x=np.zeros((Nsample,dsample));
    for i in range (0,Nsample):
        n=np.random.randint(1,6)
        x[i]=complicated_fv1(n,y[i])
        for j in range(0, dsample): #plus some random error up to specified % of the value
            x[i][j]=x[i][j]+x[i][j]*(-1)**np.random.randint(1,3)*errorpct*np.random.rand()
    return x

def pipelinev2(y,errorpct=0.05):
    Nsample=len(y)
    dsample=len(y[0])
    
    x=np.zeros((Nsample,dsample));
    for i in range (0,Nsample):
        n=np.random.randint(1,6)
        x[i]=complicated_fv2(n,y[i])
        for j in range(0, dsample): #plus some random error up to specified % of the value
            x[i][j]=x[i][j]+x[i][j]*(-1)**np.random.randint(1,3)*errorpct*np.random.rand()
    return x
    
def pm(x):
    return 1.0/(np.log(99.5/0.6))*(1.0/(x-0.5))

def massDistribution(x0=1.1,x1=100):
    count=0
    x=np.zeros(2)
    while(count<2):
        s=np.random.uniform(x0,x1)
        prob=pm(s)
        r=np.random.uniform(0,1)
        if (r<prob):
            x[count]=s        
            count=count+1
    return x


def generateEvents(Nsample, dsample=11):
    y=np.zeros((Nsample,dsample))
    for i in range (0,Nsample):
        x=massDistribution()
        m1=x[0]; m2=x[1]
        aux=np.maximum(m1,m2)
        m2=np.minimum(m1,m2)
        y[i][0]=aux
        y[i][1]=m2
        norm=2
        while(norm>1):
            y[i][2:5]=-1+2*np.random.random_sample((1,3))
            norm=np.sqrt(y[i][2]**2+y[i][3]**2+y[i][4]**2)
        norm=2
        while(norm>1):
            y[i][5:8]=-1+2*np.random.random_sample((1,3))
            norm=np.sqrt(y[i][5]**2+y[i][6]**2+y[i][7]**2)        
        y[i][8]=np.dot([y[i][2],y[i][3],y[i][4]],[y[i][5],y[i][6],y[i][7]])/(np.dot([y[i][2],y[i][3],y[i][4]],[y[i][2],y[i][3],y[i][4]])*np.dot([y[i][5],y[i][6],y[i][7]],[y[i][5],y[i][6],y[i][7]]))  #angle between spins
        y[i][9]=y[i][0]/y[i][1]  #added mass ratio!
        y[i][10]=(y[i][0]*y[i][1])**(3.0/5)/(y[i][0]+y[i][1])**(1.0/5)    #added chirp mass!
    return y


def SetSizes(Nfeatures, Nsample, Ntest, test_ratio, max_Nfeatures, max_Nsample, default_Nsample=-1):
        if default_Nsample<0:
            default_Nsample = max_Nsample

        if Nfeatures<0:
            Nfeatures = max_Nfeatures
        if Nsample<0:
            Nsample  = default_Nsample
        if Nfeatures>max_Nfeatures:
            Nfeatures = max_Nfeatures
            print('Too many features! Setting Nfeatures=', max_Nfeatures, sep='')
        if Nsample>max_Nsample:
            Nsample = max_Nsample
            print('Too many samples! Setting Nsample=', max_Nsample, sep='')
        
        Nsample = int(Nsample)
        if(test_ratio<0):
            Ntrain=Nsample-Ntest
        else:
            Ntrain  = int((1-test_ratio)*Nsample) 
        Ntest   = Nsample-Ntrain
        return Nfeatures, Nsample, Ntrain, Ntest

def GenerateData(dataset_name, Ntest,Nsample=-1,  test_ratio=0.3, Nfeatures=-1, seed=-1):
    if seed>=0:
        np.random.seed(seed)
        
    if dataset_name=='v0':
        max_Nfeatures   = 15
        max_Nsample     = 10000000
        default_Nsample = 1000
        Nfeatures, Nsample, Ntrain, Ntest = SetSizes(Nfeatures, Nsample, Ntest,test_ratio, \
                max_Nfeatures, max_Nsample, default_Nsample)
        
        y = generateEvents(Nsample)
        x = pipelinev0(y) 
        # randomness in generateEvents, no need to shuffle
        xtrain = x[:Ntrain , :Nfeatures]
        ytrain = y[:Ntrain , :Nfeatures]
        xtest  = x[Ntrain:-1, :Nfeatures]
        ytest  = y[Ntrain:-1, :Nfeatures]
        
    elif dataset_name=='v1':
        max_Nfeatures   = 15
        max_Nsample     = 10000000
        default_Nsample = 1000
        Nfeatures, Nsample, Ntrain, Ntest = SetSizes(Nfeatures, Nsample, Ntest,test_ratio, \
                max_Nfeatures, max_Nsample, default_Nsample)
        
        y = generateEvents(Nsample)
        x = pipelinev1(y) 
        # randomness in generateEvents, no need to shuffle
        xtrain = x[:Ntrain , :Nfeatures]
        ytrain = y[:Ntrain , :Nfeatures]
        xtest  = x[Ntrain:-1, :Nfeatures]
        ytest  = y[Ntrain:-1, :Nfeatures]
        
    elif dataset_name=='v2':
        max_Nfeatures   = 15
        max_Nsample     = 10000000
        default_Nsample = 1000
        Nfeatures, Nsample, Ntrain, Ntest = SetSizes(Nfeatures, Nsample, Ntest,test_ratio, \
                max_Nfeatures, max_Nsample, default_Nsample)
        
        y = generateEvents(Nsample)
        x = pipelinev2(y) 
        # randomness in generateEvents, no need to shuffle
        xtrain = x[:Ntrain , :Nfeatures]
        ytrain = y[:Ntrain , :Nfeatures]
        xtest  = x[Ntrain:-1, :Nfeatures]
        ytest  = y[Ntrain:-1, :Nfeatures]

    else:
        print("'",dataset_name,"' not found! Returning empty arrays",sep='')
        xtrain   = []
        ytrain   = []
        xtest    = []
        ytest    = []
        Ntrain   = 0
        Ntest    = 0

    # wrap data into a dictionary
    dataset  = {}
    dataset['name']     = dataset_name
    dataset['xtrain']   = xtrain
    dataset['ytrain']   = ytrain
    dataset['xtest']    = xtest
    dataset['ytest']    = ytest
    dataset['Ntrain']   = Ntrain
    dataset['Ntest']    = Ntest
    dataset['Nsample']  = Ntrain+Ntest 
    
    return dataset


def exportDictCSV(data,name,column):
    #fieldnames = ['name', 'xtrain', 'ytrain','xtest','ytest','Ntrain','Ntest','Nsample']
    with open(name, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for element in data[column]:
            writer.writerow(element)
        
        
        
def exportArrayCSV(data, name):
    with open(name, 'w') as f:
        writer = csv.DictWriter(f)
        for element in data:
            writer.writerow(element)

#--------------------------------------------------------------------------------------------------------------
#Generate tags: somewhat conditional exclusive tags
# hasNS   hasRMN    category    criteria
#   N        N        0          m2>3     
#   Y        N        1          m2<3 && f<0
#   Y        Y        2          m2<3 && f>0
def f_conditional(x):
    m1=x[0]; m2=x[1]; s1x=x[2]; s1y=x[3]; s1z=x[4]; s2x=x[5]; s2y=x[6]; s2z=x[7];
    s1=s1x+s1y+s1z; s2=s2x+s2y+s2z;
    f=-m2/2+m1*s1-m2*s2
    
    if (m2<3 and f>=0):
        return int(2)
    elif (m2<3 and f<0):
        return int(1)
    elif (m2>=3):
        return int(0)
    
def f_new(x):
    m1=x[0]; m2=x[1]; s1x=x[2]; s1y=x[3]; s1z=x[4]; s2x=x[5]; s2y=x[6]; s2z=x[7];
    s1=np.sqrt(s1x*s1x+s1y*s1y+s1z*s1z); 
    s2=np.sqrt(s2x*s2x+s2y*s2y+s2z*s2z); 
    
    if (m2>=3): #binary black hole
        return int(0)
    else:
        if (m1>3): #cases from the talk
            if (s1>0.8):
                return int(2)
            elif (m1>30 and s1<(0.000065934*m1**2+0.14066)):
                return int(1)
        alfa=0.0;b=0.4;h=40;k=0.3 ; a=h-2.5 
        ellipse=((m1-h)*np.cos(alfa)+(s1-k)*np.sin(alfa))**2/a**2+((m1-h)*np.sin(alfa)-(s1-k)*np.cos(alfa))**2/b**2
        if (ellipse>1):
            return int(2)
        else:
            return int(1)
        '''
            value=m1/10-s1-s2
            if(value<0):
                print(value,m1,s1,s2)
                return int(1)
            else:
                return int(2)
    '''

def categorize(x,talk=False):
    N=len(x)
    tags=np.zeros(N,dtype=int)
    for i in range (0,N):
        #tags[i]=f_conditional(x[i])
        tags[i]=f_new(x[i])
    if(talk):
        count_arr=np.bincount(tags)
        print('No NS ', count_arr[0]/N*100,'%')
        print('Has NS, no remnant: ', count_arr[1]/N*100,'%')
        print('Has NS, yes remnant: ', count_arr[2]/N*100,'%')
    return tags


#%%
dic1=GenerateData('v0',0,15000,0.0)
#%%
categ=categorize(dic1['ytrain'],True)
#%%
bins=[1.1,10,20,30,40,50,60,70,80,90,100]
plt.title('Mass1 distribution')
plt.hist(dic1['ytrain'][:,0],bins=bins); plt.show();plt.clf()
#plt.hist(dic1['ytrain'][:,0],bins=bins,density=True);plt.plot(bins,pm(np.asarray(bins))); plt.show();plt.clf()
plt.title('Mass2 distribution')
plt.hist(dic1['ytrain'][:,1],bins=bins); plt.show();plt.clf()
#%%
plt.title('Binaries generated')
plt.xlabel('m1')
plt.ylabel('m2')
plt.scatter(dic1['ytrain'][:,0],dic1['ytrain'][:,1]);plt.show()
#%%
plt.plot(dic1['ytrain'][:,9],'o'); plt.show();plt.clf()
#%%
categnew=categorize(dic1['ytrain'],True)
#%%
norm=np.sqrt(dic1['ytrain'][:,2]*dic1['ytrain'][:,2]+dic1['ytrain'][:,3]*dic1['ytrain'][:,3]+dic1['ytrain'][:,4]*dic1['ytrain'][:,4])
col=[]
for e in categnew:
    if (e==0):
        col.append('green')
    elif (e==1):
        col.append('blue')
    else:
        col.append('red')
#%%
plt.title('Categories: 0-green 1-blue 2-red')
plt.xlabel('m1')
plt.ylabel('s1')
plt.scatter(dic1['ytrain'][:,0],norm,c=col,s=0.5)
#%%

s2=np.sqrt(dic1['ytrain'][:,5]*dic1['ytrain'][:,5]+dic1['ytrain'][:,6]*dic1['ytrain'][:,6]+dic1['ytrain'][:,7]*dic1['ytrain'][:,7])
s1=np.sqrt(dic1['ytrain'][:,2]*dic1['ytrain'][:,2]+dic1['ytrain'][:,3]*dic1['ytrain'][:,3]+dic1['ytrain'][:,4]*dic1['ytrain'][:,4])

plt.scatter(s1,s2,c=col,s=0.5)

#%%
#exportDictCSV(dic1,'/home/IPAMNET/mberbel/Documents/ML/predict_v0_x.csv', 'xtrain')    
#%%
plt.xlabel('m')
plt.ylabel('probability')
plt.plot(np.arange(1.1,100,0.1),pm(np.arange(1.1,100,0.1)))

#%%
plt.plot(dic1['ytrain'][:,3],dic1['xtrain'][:,3],'o');plt.show()





