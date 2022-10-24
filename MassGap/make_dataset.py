import numpy as np
import matplotlib.pyplot as plt
import random
import csv

pathBNS = "C:\\Users\\marin\\Desktop\\ML_workingGroup\\NS_dataset_NOPUBLIC\\"
pathHere = "C:\\Users\\marin\\Desktop\\ML_workingGroup\\classy_RF\\HasMassGap\\AllEvents\\"
pathClassy = "C:\\Users\\marin\\Desktop\\ML_workingGroup\\classy_RF\\"


#Mass Gap File Header
#inj_mass1_source_frame,inj_mass2_source_frame,id,inj_m1,inj_m2,inj_spin1z,inj_spin2z,inj_redshift,rec_m1,rec_m2,rec_spin1z,rec_spin2z,Gamma1,cfar,snr,gpstime

#BNS FIle Header
# ["ID", "m1_inj" , "m2_inj", "chi1_inj", "chi2_inj", "mc_inj", "q_inj", "R_isco_inj", "Compactness_inj", "m1_rec", "m2_rec", "chi1_rec", "chi2_rec", "mc_rec", "frac_mc_err", "q_rec", "R_isco_rec", "Compactness_rec", "snr", "label"]


def readfile(filename, header=True):
    lst=[]
    
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        if(header):
            next(csv_reader, None)
        for row in csv_reader:
            lst.append(row)
    data=np.array(lst, dtype=float)
    
    return data

def writefile(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(["m1inj, m2inj, m1rec, m2rec, chi1rec, chi2rec, snr, label"])
        for row in data:
            spamwriter.writerow(row)
            
def categorize(m1i, m2i):
    #2 categories
    vector1=np.where(np.logical_and(m1i>= 3, m1i<= 5), 1, 0)
    vector2=np.where(np.logical_and(m2i>= 3, m2i<= 5), 1, 0)
    label_vector_test=vector1+vector2
    label_2cat=np.where(label_vector_test==2,1,label_vector_test)
    print("Two categories:  0 - No Mass Gap, 1 - Mass Gap")
    unique, counts = np.unique(label_2cat, return_counts=True)
    print(dict(zip(unique, counts)))
          
    #3 categories
    vector=np.where(np.logical_or(m1i <= 3, m2i <=3), 0, 5)
    vector=np.where(np.logical_or(m1i >= 5, m2i >=5), 2, vector)
    label_3cat=np.where(np.logical_or(np.logical_and(m1i > 3, m1i <5),np.logical_and(m2i > 3, m2i < 5)), 1, vector)
    print("Three categories: 0 - a mass<3,  1 - Mass Gap,  2 - a mass>5")
    unique, counts = np.unique(label_3cat, return_counts=True)
    print(dict(zip(unique, counts)))
    
    #4 categories
    vector=np.where(np.logical_and(m1i <= 3, m2i <=3), 0, 5)
    vector=np.where(np.logical_and(m1i >= 5, m2i >=5), 3, vector)
    vector=np.where(np.logical_and(m1i >= 5, m2i <=5), 2, vector)
    label_4cat=np.where(np.logical_or(np.logical_and(m1i > 3, m1i <5),np.logical_and(m2i > 3, m2i < 5)), 1, vector)
    print("Four categories: 0 - m1&m2<3 (BNS),  1 - Mass Gap,  2 - m1>5 & m2<3 (NSBH) ,  3 - m1&m2>5 (BBH)")
    unique, counts = np.unique(label_4cat, return_counts=True)
    print(dict(zip(unique, counts)))
          
    return label_2cat, label_3cat, label_4cat

            
            
def createJointDataset():
    train_BNS = readfile(pathBNS + 'train_NS.csv', False)
    test_BNS = readfile(pathBNS + 'test_NS.csv', False)
    massgap = readfile(pathHere + 'massgap.csv', True)
    
    N = len(massgap);
    test_N = int(np.floor(0.3*N));
    train_N = N - test_N
    print("Adding ", train_N, " samples in the mass gap to training")
    print("Adding ", test_N, " samples in the mass gap to testing")
    random.seed(4815162342)
    
    indexes_test=random.sample(list(np.arange(0,N)), test_N);
    all_indexes=(np.arange(0,N)).tolist()
    indexes_train=list(set(all_indexes) - set(indexes_test))
    
    BNS_m1_inj = 1
    BNS_m2_inj = 2 
    BNS_m1_rec = 9
    BNS_m2_rec = 10
    BNS_chi1_rec = 11
    BNS_chi2_rec = 12
    BNS_snr = 18
    
    MG_m1_inj = 0
    MG_m2_inj = 1
    MG_m1_rec = 8
    MG_m2_rec = 9 
    MG_chi1_rec = 10
    MG_chi2_rec = 11
    MG_snr = 14
    
    train_BNS = train_BNS[:,[BNS_m1_inj, BNS_m2_inj, BNS_m1_rec, BNS_m2_rec, BNS_chi1_rec, BNS_chi2_rec, BNS_snr]]
    Train = list(train_BNS)
    
    for i in range (0, train_N):
        Train.append( massgap[ indexes_train[i], [MG_m1_inj, MG_m2_inj, MG_m1_rec, MG_m2_rec, MG_chi1_rec, MG_chi2_rec, MG_snr]])
    
    test_BNS = test_BNS[:,[BNS_m1_inj, BNS_m2_inj, BNS_m1_rec, BNS_m2_rec, BNS_chi1_rec, BNS_chi2_rec, BNS_snr]]
    Test = list(test_BNS)
    
    for i in range (0, test_N):
        Test.append( massgap[ indexes_test[i], [MG_m1_inj, MG_m2_inj, MG_m1_rec, MG_m2_rec, MG_chi1_rec, MG_chi2_rec, MG_snr]])
        
    Train = np.asarray(Train)
    Test = np.asarray(Test)
    
    m1_inj = 0
    m2_inj = 1
    m1_rec = 2
    m2_rec = 3
    chi1 = 4
    chi2 = 5
    snr = 6
    
    m1i = Train[:,m1_inj]
    m2i = Train[:,m2_inj]
    label_train_2, label_train_3, label_train_4 = categorize(m1i, m2i)
          
    m1i = Test[:,m1_inj]
    m2i = Test[:,m2_inj]
    label_test_2, label_test_3, label_test_4 = categorize(m1i, m2i)
          
    Train_dataset_2cat = np.c_[Train, label_train_2]
    Train_dataset_3cat = np.c_[Train, label_train_3]
    Train_dataset_4cat = np.c_[Train, label_train_4]
          
    Test_dataset_2cat = np.c_[Test, label_test_2]
    Test_dataset_3cat = np.c_[Test, label_test_3]
    Test_dataset_4cat = np.c_[Test, label_test_4]
    
    writefile(pathHere + 'train_2cat.csv', Train_dataset_2cat)
    writefile(pathHere + 'train_3cat.csv', Train_dataset_3cat)
    writefile(pathHere + 'train_4cat.csv', Train_dataset_4cat)
    writefile(pathHere + 'test_2cat.csv', Test_dataset_2cat)
    writefile(pathHere + 'test_3cat.csv', Test_dataset_3cat)
    writefile(pathHere + 'test_4cat.csv', Test_dataset_4cat)
          


#-----------------------------------------------------------------
# SCORES
#-----------------------------------------------------------------
def otherscores(RF):
    pred=RF.test_prediction
    true=RF.labels_test
    N=len(true);
    tp=0; fp=0; fn=0; tn=0;
    for i in range(0,N):
        if true[i]==1:
            if pred[i]==1:
                tp=tp+1
            else:
                fn=fn+1
        else:
            if pred[i]==1:
                fp=fp+1
            else:
                tn=tn+1
    tpr=tp/N; fpr=fp/N; fnr=fn/N; tnr=tn/N;  
    sensitivity=tpr/(tpr+fnr)
    precision=tpr/(tpr+fpr)
    f1score=2*(precision*sensitivity)/(precision+sensitivity)
    print("Sensitivity ", sensitivity)
    print("Precision ",precision)
    print("F1 score ",f1score)
    
    
def hist_ROC_MassGap(algo, figname):
    probs=algo.model.predict_proba(algo.data_test)
    pred=probs[:,1]
    truelabel=algo.labels_test
    index_events_has=(np.where(truelabel == 1))
    p_events_has=pred[index_events_has]
    index_events_nohas=np.where(truelabel != 1)
    p_events_nohas=pred[index_events_nohas]
    
    
    allprob = algo.model.predict_proba(algo.data_test)
    v_prob_NS = allprob[:,1]
    events_have_NS = np.where(algo.labels_test==1)[0]
    N = len(events_have_NS)
    M = len(algo.labels_test) - N
    threshold = np.linspace(0,1,101)[1:-1]
    TP = np.zeros(99)
    FP = np.zeros(99)
    i=0
    for thr in threshold:
        index_say_yes = np.where(v_prob_NS>=thr)[0]
        count_yes = 0.0; count_no = 0.0
        for index in index_say_yes:
            if index in events_have_NS:
                count_yes=count_yes + 1.0
            else:
                count_no=count_no + 1.0
        TP[i] = count_yes/N
        FP[i] = count_no/M
        i = i + 1
    
    
    
    f, (ax) = plt.subplots(1,2,figsize=(10, 5),constrained_layout=True)
    
    ax[0].hist(p_events_nohas,bins=np.linspace(0,1,20),color='green',alpha=0.5, label='No MassGap')
    ax[0].hist(p_events_has,bins=np.linspace(0,1,20),color=(0.1, 0.2, 0.5, 0.),edgecolor='black', hatch="/",label='MassGap')
    ax[0].set_yscale('log')
    ax[0].set_yticks([1e2,1e3,1e4,1e5])
    ax[0].set_xticks([0,0.2,0.4,0.6,0.8,1])
    ax[0].set_ylim([4.5e1,1.9e5])
    ax[0].set_xlabel('P(MassGap)')
    ax[0].axvline(x=0.5,color='black',ls='--')
    ax[0].grid(ls='--')
    handles, labels = ax[0].get_legend_handles_labels()
    order = [1,0]
    #add legend to plot
    ax[0].legend([handles[idx] for idx in order],[labels[idx] for idx in order],fontsize=12,loc=1)
    
    
    sc=ax[1].scatter(FP, TP, c=threshold, cmap='viridis')
    f.colorbar(sc,ax=ax[1], label="Threshold", aspect=50)
    ax[1].set_xlabel("False Positive")
    ax[1].set_ylabel("True Positive")
    ax[1].grid(ls='--')
    ax[1].set_ylim([0.82,1.02])
    ax[1].set_xlim([0,0.2])
    ax[1].set_yticks(np.linspace(0.825,1,8))
    
    
    if algo.save_plots:
        plt.savefig(figname+".png",dpi=200,bbox_inches='tight')
    if algo.show_plots:
        plt.show()
    return