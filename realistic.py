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
    
    c0=m1*np.abs(s1x)+s2y**2+m2/np.sqrt(chirp)+(1.2-q)/(s1z+3) #remains positive
    c1=m2**3*(chirp+np.pi-s2z)/(5-s2x+s1y) #remains positive
    c2=(s2x-s1x)+(s1z-s1y) 
    c3=-(s2y-s1y)+(s1x-s1z) 
    c4=(s2z-s1z)+(s1y-s1x) 
    c5=-(s2x-s1x)+(s2z-s2y) 
    c6=(s2y-s1y)+(s2x-s2z) 
    c7=-(s2z-s1z)+(s2y-s2x) 
    c8=(chirp*m1+angle-s1x*s2y*m2)/np.pi;      
    c9=chirp*s2y-q+0.4*m1*np.exp(-m2) 
    c10=chirp
    return [c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10]
    
def complicated_fv1(n,vector):
    m1=vector[0]; m2=vector[1]; s1x=vector[2]; s1y=vector[3]; s1z=vector[4]; 
    s2x=vector[5]; s2y=vector[6]; s2z=vector[7]
    angle=vector[8]; q=vector[9]; chirp=vector[10]
    
    if (n==1):
        c0=m1*np.abs(s1x)+s2y**2+m2/np.sqrt(chirp)+(1.2-q)/(s1z+3) #remains positive
        c1=m2**2*(chirp+np.pi-s2z)/(4-s2x+s1y) #remains positive
        c2=(s2x-s1x)**3+(s1z-s1y)
        c3=(s2y-s1y)**3+(s1x-s1z)
        c4=(s2z-s1z)**3+(s1y-s1x)
        c5=(s2x-s1x)+(s2z-s2y)**3
        c6=(s2y-s1y)+(s2x-s2z)**3
        c7=(s2z-s1z)+(s2y-s2x)**3
        c8=(angle-s1z*s2y*m2)/np.pi;      # remains positive
        c9=chirp*s2y*s1z-q+0.4*m2 #below 1?
        c10=chirp
    elif (n==2):
        c0=m1*np.abs(s1z)+s2y**2+m2/np.sqrt(chirp)+(1.5-q)/(s1z+2) #remains positive
        c1=m2**4*(m2+np.pi*0.5-s2z)/(5-s2x+s1y) #remains positive
        c2=(s2y-s1y)+(s1z-s2y)**3 #below 1
        c3=(s2z-s1z)+(s1x-s1z)**3
        c4=(s2x-s1x)+(s2y-s1x)**3
        c5=(s2x-s1x)**3+(s2z-s2y)
        c6=(s2y-s1y)**3+(s2x-s2z)
        c7=(s2z-s1z)**3+(s2y-s2x)
        c8=(angle*0.5-s1x*s2y*chirp)/np.pi;      # remains positive
        c9=chirp*s2y-q+0.4*m2*np.exp(-m1) #below 1
        c10=chirp+0.1
    elif (n==3):
        c0=m2*np.abs(s1x)+s2y**2+m2/np.sqrt(chirp)+(angle)/(s1z+3) #remains positive
        c1=m2**3*(m1+np.pi-s2z)/(6-s2z-s1y) #remains positive
        c2=-(s1z-s1y)**3+(s1z-s1y) #below 1
        c3=(s2x-s1x)**3+(s2x-s1z)
        c4=-(s2y-s1x)**3+(s1y-s1x)
        c5=(s2x-s1z)+(s2z-s2x)**3
        c6=-(s2z-s1y)+(s2x-s2z)**3
        c7=(s2x-s1z)+(s2y-s1y)**3
        c8=(chirp-s1y*s2y*m2)/np.pi;      # remains positive
        c9=np.exp(-np.abs(angle))*chirp*s1z-q+0.2*m1 #below 1
        c10=chirp-0.05
    elif (n==4):
        c0=m1*m2*np.abs(s1x)+s2y**3+m2/np.sqrt(chirp)+(1.6-q)/(s1z+1.5) #remains positive
        c1=chirp**3*(m1-m2+np.pi-s2z)/(2.3-s2x+s1y) #remains positive
        c2=(s2x-s1x)+(s1x-s1y)**5 #below 1
        c3=-(s2y-s1x)+(s1y-s1x)**3
        c4=(s2z-s1y)**3+(s1y-s1x)
        c5=(s2z-s1x)-(s2x-s2y)**5
        c6=-(s2y-s1y)+(s2x-s2y)
        c7=(s2x-s1z)**3+(s2y-s2x)
        c8=angle-s1y*s2z*m2;      # remains positive
        c9=angle*s2x+q+0.4*m2 #below 1
        c10=chirp+s1x*0.5
    else:
        c0=chirp*np.abs(s1z)+s2y**3+m1/np.sqrt(m1)+(1.1-q)/(s1z+1.2) #remains positive
        c1=m2*(chirp+np.pi-s1x)/(5-s2z+s1x) #remains positive
        c2=-(s2x-s1x)+(s1z-s1y)**3 #below 1
        c3=-(s2y-s1y)**3+(s1x-s1y)
        c4=-(s2x-s1z)**3+(s1y-s1x)
        c5=-(s2x-s1y)+(s2x-s2y)**3
        c6=-(s1y-s1y)**3+(s2x-s1y)
        c7=-(s2z-s1x)+(s2y-s2x)**3
        c8=(q*chirp*angle-s1z*s2x*m2)/np.pi;      # remains positive
        c9=m1*s1z-q+0.2*m1 #below 1
        c10=chirp-s2z*0.3
    return [c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10]

def complicated_fv2(n,vector):
    m1=vector[0]; m2=vector[1]; s1x=vector[2]; s1y=vector[3]; s1z=vector[4]; 
    s2x=vector[5]; s2y=vector[6]; s2z=vector[7]
    angle=vector[8]; q=vector[9]; chirp=vector[10]
    
    if (n==1):
        c0=m1*np.abs(s1x)+s2y**2+m2/np.sqrt(chirp)+(1.2-q)/(s1z+3) #remains positive
        c1=m1*chirp+np.cos(q) #remains positive
        c2=(s2x-s1x)+(s1z-s1y)**3 #below 1
        c3=-(s2y-s1y)+(s1x-s1z)**5
        c4=(s2z-s1z)+(s1y-s1x)**3
        c5=-(s2x-s1x)**5+(s2z-s2y)**3
        c6=(s2y-s1y)+(s2x-s2z)**5
        c7=-(s2z-s1z)+(s2y-s2x)**3
        c8=chirp*m1*angle-s1x*s2y*m2/np.pi;      # remains positive
        c9=chirp*s2y-q+0.4*m2*np.exp(-m1*m2) #below 1
        c10=chirp
    elif (n==2):
        c0=chirp*np.abs(s1x)+q*angle #
        c1=np.sqrt(m2)*(m2+np.pi*0.5-s2z)/(5-s2x+s1y) #remains positive
        c2=(s1x+s2z-s1y**2)/(3+chirp) #below 1
        c3=-m2*(s1z+s2y-s2x**2)/(chirp+m1)
        c4=(s1x+s2y-s1y**2)/(chirp**2)
        c5=-(s2x+s2z-s1y**2)/(chirp-1*m1)
        c6=(s1x+s2z-s2y**2)/(m1)
        c7=-(s1y+s2x-s1z**2)/(m1*m2**2)
        c8=m1*m2-chirp*s1z*s1x*s2y;      # remains positive
        c9=np.exp(-q*s1z*s2y) #below 1
        c10=chirp+s1z*0.5
    elif (n==3):
        c0=s1x*s2z*s1y+chirp +m1*q #remains positive
        c1=m2+chirp+m1-angle #remains positive
        c2=angle*s1x**3 #below 1
        c3=(angle-q)*s2x**5
        c4=-q*angle*s1y**3
        c5=m2**2/m1**3*angle*s1z**3
        c6=m2*angle*s1x*s2y**3/chirp
        c7=-angle*(s1x+s2z)**3
        c8= np.pi-q**2+s1x     # remains positive
        c9= m2/m1**3#below 1
        c10=chirp+angle*0.3
    elif (n==4):
        c0=m1**2-q+angle/chirp +s1z #remains positive
        c1=np.sqrt(chirp)+m1**2/(2+angle**3) #remains positive
        c2=-(q**2*s1z-s2x) #below 1
        c3=(q**2*s1z-s2x)/m2
        c4=-(q*s1z**5-s2z)
        c5=(q**2*s1z-s2y)/m1
        c6=-(q*s1y**3-s2x)
        c7=(q**2*s2x-s2x)*q
        c8= np.log(np.sqrt(m1)+m2*q**3)   # remains positive
        c9= np.exp(-s1x)#below 1
        c10=chirp+0.05
    else:
        c0=s1x*m1+s2z*m2+chirp+angle+np.abs(s2y+s1y) #remains positive
        c1=3*m2**2-np.log(chirp+1)-s1x*q #remains positive
        c2=np.exp(-q) #below 1
        c3=np.exp(-q*s2x**3-s2x*q) 
        c4=np.exp(-q/m2)
        c5=np.exp(-angle)
        c6=np.exp(-m2/chirp)
        c7=np.exp(-q*s1x)
        c8= np.tan(-chirp)/np.pi     # remains positive
        c9= angle+chirp**3.2
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


def generateEvents(Nsample, cv,dsample=11):
    y=np.zeros((Nsample,dsample))
    tag=np.zeros(Nsample,dtype=int)
    third=int(Nsample/3)
    total0=third+(Nsample-third*3)
    total1=third
    total2=third
    print(total0,total1,total2)
    count0=0; count1=0; count2=0
    i=0
    while(i<Nsample):
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
        if(cv==0):
            tag[i]=f_conditional(y[i])
        else:
            tag[i]=f_new(y[i])
        
        if(tag[i]==0):
            if (count0<total0):
                count0=count0+1
                y[i][8]=np.dot([y[i][2],y[i][3],y[i][4]],[y[i][5],y[i][6],y[i][7]])/np.sqrt(np.dot([y[i][2],y[i][3],y[i][4]],[y[i][2],y[i][3],y[i][4]])*np.dot([y[i][5],y[i][6],y[i][7]],[y[i][5],y[i][6],y[i][7]]))  #angle between spins
                y[i][9]=y[i][0]/y[i][1]  #added mass ratio!
                y[i][10]=(y[i][0]*y[i][1])**(3.0/5)/(y[i][0]+y[i][1])**(1.0/5)
                i=i+1

        elif(tag[i]==1):
            if (count1<total1):
                count1=count1+1
                y[i][8]=np.dot([y[i][2],y[i][3],y[i][4]],[y[i][5],y[i][6],y[i][7]])/np.sqrt(np.dot([y[i][2],y[i][3],y[i][4]],[y[i][2],y[i][3],y[i][4]])*np.dot([y[i][5],y[i][6],y[i][7]],[y[i][5],y[i][6],y[i][7]]))  #angle between spins
                y[i][9]=y[i][0]/y[i][1]  #added mass ratio!
                y[i][10]=(y[i][0]*y[i][1])**(3.0/5)/(y[i][0]+y[i][1])**(1.0/5)
                i=i+1

        else:
            if (count2<total2):
                count2=count2+1
                y[i][8]=np.dot([y[i][2],y[i][3],y[i][4]],[y[i][5],y[i][6],y[i][7]])/np.sqrt(np.dot([y[i][2],y[i][3],y[i][4]],[y[i][2],y[i][3],y[i][4]])*np.dot([y[i][5],y[i][6],y[i][7]],[y[i][5],y[i][6],y[i][7]]))  #angle between spins
                y[i][9]=y[i][0]/y[i][1]  #added mass ratio!
                y[i][10]=(y[i][0]*y[i][1])**(3.0/5)/(y[i][0]+y[i][1])**(1.0/5)
                i=i+1

    return y, tag


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

def GenerateData(dataset_name, cv, Ntest,Nsample=-1,  test_ratio=0.3, Nfeatures=-1, seed=-1):
    if seed>=0:
        np.random.seed(seed)
        
    if dataset_name=='v0':
        max_Nfeatures   = 15
        max_Nsample     = 10000000
        default_Nsample = 1000
        Nfeatures, Nsample, Ntrain, Ntest = SetSizes(Nfeatures, Nsample, Ntest,test_ratio, \
                max_Nfeatures, max_Nsample, default_Nsample)
        
        y,tag = generateEvents(Nsample,cv)
        x = pipelinev0(y) 
        # randomness in generateEvents, no need to shuffle
        xtrain = x[:Ntrain , :Nfeatures]
        ytrain = y[:Ntrain , :Nfeatures]
        traintag = tag[:Ntrain]
        xtest  = x[Ntrain:-1, :Nfeatures]
        ytest  = y[Ntrain:-1, :Nfeatures]
        testtag = tag[Ntrain:-1]
        
    elif dataset_name=='v1':
        max_Nfeatures   = 15
        max_Nsample     = 10000000
        default_Nsample = 1000
        Nfeatures, Nsample, Ntrain, Ntest = SetSizes(Nfeatures, Nsample, Ntest,test_ratio, \
                max_Nfeatures, max_Nsample, default_Nsample)
        
        y,tag = generateEvents(Nsample,cv)
        x = pipelinev1(y) 
        # randomness in generateEvents, no need to shuffle
        xtrain = x[:Ntrain , :Nfeatures]
        ytrain = y[:Ntrain , :Nfeatures]
        traintag = tag[:Ntrain]
        xtest  = x[Ntrain:-1, :Nfeatures]
        ytest  = y[Ntrain:-1, :Nfeatures]
        testtag = tag[Ntrain:-1]
        
    elif dataset_name=='v2':
        max_Nfeatures   = 15
        max_Nsample     = 10000000
        default_Nsample = 1000
        Nfeatures, Nsample, Ntrain, Ntest = SetSizes(Nfeatures, Nsample, Ntest,test_ratio, \
                max_Nfeatures, max_Nsample, default_Nsample)
        
        y,tag = generateEvents(Nsample,cv)
        x = pipelinev2(y) 
        # randomness in generateEvents, no need to shuffle
        xtrain = x[:Ntrain , :Nfeatures]
        ytrain = y[:Ntrain , :Nfeatures]
        traintag = tag[:Ntrain]
        xtest  = x[Ntrain:-1, :Nfeatures]
        ytest  = y[Ntrain:-1, :Nfeatures]
        testtag = tag[Ntrain:-1]

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
    dataset['traintag'] = traintag
    dataset['xtest']    = xtest
    dataset['ytest']    = ytest
    dataset['testtag'] = testtag
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
    data.reshape(len(data),1)
    with open(name, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for el in data:
            writer.writerow(el)

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


def categorize(x,talk=False):
    N=len(x)
    tags=np.zeros(N,dtype=int)
    for i in range (0,N):
        tags[i]=f_conditional(x[i])
    if(talk):
        count_arr=np.bincount(tags)
        print('No NS ', count_arr[0]/N*100,'%')
        print('Has NS, no remnant: ', count_arr[1]/N*100,'%')
        print('Has NS, yes remnant: ', count_arr[2]/N*100,'%')
    return tags

def categorize_new(x,talk=False):
    N=len(x)
    tags=np.zeros(N,dtype=int)
    for i in range (0,N):
        tags[i]=f_new(x[i])
    if(talk):
        count_arr=np.bincount(tags)
        print('No NS ', count_arr[0]/N*100,'%')
        print('Has NS, no remnant: ', count_arr[1]/N*100,'%')
        print('Has NS, yes remnant: ', count_arr[2]/N*100,'%')
    return tags


#%%
version='v2'
extra='c0'
purpose='test'
dic1=GenerateData(version,1,0,1000,0.0)
N=len(dic1['traintag'])
count_arr=np.bincount(dic1['traintag'])
print('No NS ', count_arr[0]/N*100,'%')
print('Has NS, no remnant: ', count_arr[1]/N*100,'%')
print('Has NS, yes remnant: ', count_arr[2]/N*100,'%')

#%%
save=False
bins=[1.1,10,20,30,40,50,60,70,80,90,100]
plt.title('Mass1 distribution')
plt.hist(dic1['ytrain'][:,0],bins=bins);
if (save):
    plt.savefig('/home/IPAMNET/mberbel/Documents/ML/NewRealistic/m1_hist'+version+extra+purpose+'.png',dpi=200,bbox_inches='tight')
plt.clf()

plt.title('Mass2 distribution')
plt.hist(dic1['ytrain'][:,1],bins=bins);
if (save):    
    plt.savefig('/home/IPAMNET/mberbel/Documents/ML/NewRealistic/m2_hist'+version+extra+purpose+'.png',dpi=200,bbox_inches='tight')
plt.clf()

plt.title('Binaries generated')
plt.xlabel('m1'); plt.ylabel('m2')
plt.scatter(dic1['ytrain'][:,0],dic1['ytrain'][:,1]);
if (save):
    plt.savefig('/home/IPAMNET/mberbel/Documents/ML/NewRealistic/binaries'+version+extra+purpose+'.png',dpi=200,bbox_inches='tight')
plt.clf()

norm=np.sqrt(dic1['ytrain'][:,2]*dic1['ytrain'][:,2]+dic1['ytrain'][:,3]*dic1['ytrain'][:,3]+dic1['ytrain'][:,4]*dic1['ytrain'][:,4])
col=[]
for e in dic1['traintag']:
    if (e==0):
        col.append('green')
    elif (e==1):
        col.append('blue')
    else:
        col.append('red')
plt.title('Categories: 0-green 1-blue 2-red')
plt.xlabel('m1')
plt.ylabel('s1')
plt.scatter(dic1['ytrain'][:,0],norm,c=col,s=0.5)
if (save):
    plt.savefig('/home/IPAMNET/mberbel/Documents/ML/NewRealistic/categories'+version+extra+'.png',dpi=200,bbox_inches='tight')
plt.clf()

plt.title('0');plt.plot(dic1['xtrain'][:,0]); plt.show(); plt.clf()
plt.title('1');plt.plot(dic1['xtrain'][:,1]); plt.show(); plt.clf()
plt.title('2');plt.plot(dic1['xtrain'][:,2]); plt.show(); plt.clf()
plt.title('3');plt.plot(dic1['xtrain'][:,3]); plt.show(); plt.clf()
plt.title('4');plt.plot(dic1['xtrain'][:,4]); plt.show(); plt.clf()
plt.title('5');plt.plot(dic1['xtrain'][:,5]); plt.show(); plt.clf()
plt.title('6');plt.plot(dic1['xtrain'][:,6]); plt.show(); plt.clf()
plt.title('7');plt.plot(dic1['xtrain'][:,7]); plt.show(); plt.clf()
plt.title('8');plt.plot(dic1['xtrain'][:,8]); plt.show(); plt.clf()
plt.title('9');plt.plot(dic1['xtrain'][:,9]); plt.show(); plt.clf()
plt.title('10');plt.plot(dic1['xtrain'][:,10]); plt.show(); plt.clf()

#%%
exportDictCSV(dic1,'/home/IPAMNET/mberbel/Documents/ML/NewRealistic/'+version+extra+purpose+'_x.csv', 'xtrain') 
exportDictCSV(dic1,'/home/IPAMNET/mberbel/Documents/ML/NewRealistic/'+ version+extra+purpose+'_y.csv', 'ytrain') 
np.savetxt('/home/IPAMNET/mberbel/Documents/ML/NewRealistic/'+ version+extra+purpose+'_tag.csv', dic1['traintag'], delimiter=',')
print('all exported')
#%%
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

#%%

#%%

s2=np.sqrt(dic1['ytrain'][:,5]*dic1['ytrain'][:,5]+dic1['ytrain'][:,6]*dic1['ytrain'][:,6]+dic1['ytrain'][:,7]*dic1['ytrain'][:,7])
s1=np.sqrt(dic1['ytrain'][:,2]*dic1['ytrain'][:,2]+dic1['ytrain'][:,3]*dic1['ytrain'][:,3]+dic1['ytrain'][:,4]*dic1['ytrain'][:,4])

plt.scatter(s1,s2,c=col,s=0.5)

#%%
 
#%%
plt.xlabel('m')
plt.ylabel('probability')
plt.plot(np.arange(1.1,100,0.1),pm(np.arange(1.1,100,0.1)))

#%%
plt.plot(dic1['ytrain'][:,3],dic1['xtrain'][:,3],'o');plt.show()


#%%
N=len(dic1['traintag'])
count_arr=np.bincount(dic1['traintag'])
print('No NS ', count_arr[0]/N*100,'%')
print('Has NS, no remnant: ', count_arr[1]/N*100,'%')
print('Has NS, yes remnant: ', count_arr[2]/N*100,'%')


