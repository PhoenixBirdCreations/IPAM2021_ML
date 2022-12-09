import KNNclassy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def hist_NS(algo, figname="histNS"):
    plt.rcParams["font.size"]=14

    probs=algo.metrics["probab"]
    pred=1-probs[:,0]
    truelabel=algo.label_test
    index_events_has=(np.where((truelabel == 1) | (truelabel == 2)))
    p_events_has=pred[index_events_has]
    index_events_nohas=np.where(truelabel==0)
    p_events_nohas=pred[index_events_nohas]

    plt.hist(p_events_nohas,bins=np.linspace(0,1,20),color='green',alpha=0.5, label='No NS')
    plt.hist(p_events_has,bins=np.linspace(0,1,20),color=(0.1, 0.2, 0.5, 0.),edgecolor='black', hatch="/",label='Has NS')
    plt.yscale('log')
    plt.yticks([1e2,1e3,1e4,1e5])
    plt.xticks([0,0.2,0.4,0.6,0.8,1])
    plt.ylim([4.5e1,1.9e5])
    plt.xlabel('P(HasNS)')
    plt.axvline(x=0.5,color='black',ls='--')
    plt.grid(ls='--')
    plt.title(eos)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,0]
    #add legend to plot
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],fontsize=12,loc=1)

    plt.savefig(figname+".png",dpi=200,bbox_inches='tight')
    plt.close()

    return

def hist_REM(algo, figname="histREM"):
    probs=algo.metrics["probab"]
    pred=probs[:,2]
    truelabel=algo.label_test
    index_events_has=np.where(truelabel == 2)
    p_events_has=pred[index_events_has]
    index_events_nohas=np.where((truelabel==0) | (truelabel==1))
    p_events_nohas=pred[index_events_nohas]

    plt.hist(p_events_nohas,bins=np.linspace(0,1,20),color='green',alpha=0.5, label='No Remnant')
    plt.hist(p_events_has,bins=np.linspace(0,1,20),color=(0.1, 0.2, 0.5, 0.),edgecolor='black', hatch="/",label='Has Remnant')
    plt.yscale('log')
    plt.yticks([1e2,1e3,1e4,1e5])
    plt.xticks([0,0.2,0.4,0.6,0.8,1])
    plt.ylim([4.5e1,1.9e5])
    plt.xlabel('P(HasRemnant)')
    plt.axvline(x=0.5,color='black',ls='--')
    plt.title(eos)
    plt.grid(ls='--')
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,0]
    #add legend to plot
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],fontsize=12,loc=1)

    plt.savefig(figname+".png",dpi=200,bbox_inches='tight')
    plt.close()
    return

def ROC_NS(algo, thr_wanted = [], figname="ROC_NS"):
    allprob = algo.metrics["probab"]
    v_prob_NS = 1-allprob[:,0]
    events_have_NS = np.where((algo.label_test==1) | (algo.label_test==2))[0]
    N = len(events_have_NS)
    M = len(algo.label_test) - N
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
        if (len(thr_wanted)>0 and (thr in thr_wanted)):
            print("Threshold ",thr, "TP: {:.3f}, FP {:.3f}".format(TP[i], FP[i]))
        i = i + 1

    plt.figure()
    sc=plt.scatter(FP, TP, c=threshold, cmap='viridis')
    plt.colorbar(sc, label="Threshold")
    plt.xlabel("False Positive")
    plt.ylabel("True Positive")
    plt.title(eos)
    plt.grid(ls='--')
    plt.ylim([0.82,1.02])
    plt.xlim([0,0.2])
    plt.yticks(np.linspace(0.825,1,8))

    plt.savefig(figname+'.png',dpi=200,bbox_inches='tight')

    return np.asarray(FP), np.asarray(TP), threshold

def ROC_REM(algo, thr_wanted = [], figname="ROC_REM"):
    allprob = algo.metrics["probab"]
    v_prob_REM = allprob[:,2]
    events_have_REM = np.where(algo.label_test==2)[0]
    N = len(events_have_REM)
    M = len(algo.label_test) - N
    threshold = np.linspace(0,1,101)[1:-1]
    TP = np.zeros(99)
    FP = np.zeros(99)
    i=0
    for thr in threshold:
        index_say_yes = np.where(v_prob_REM>=thr)[0]
        count_yes = 0.0; count_no = 0.0
        for index in index_say_yes:
            if index in events_have_REM:
                count_yes=count_yes + 1.0
            else:
                count_no=count_no + 1.0
        TP[i] = count_yes/N
        FP[i] = count_no/M
        if (len(thr_wanted)>0 and (thr in thr_wanted)):
            print("Threshold ",thr, "TP: {:.3f}, FP {:.3f}".format(TP[i], FP[i]))
        i = i + 1

    plt.figure()
    sc=plt.scatter(FP, TP, c=threshold, cmap='viridis')
    plt.colorbar(sc, label="Threshold")
    plt.xlabel("False Positive")
    plt.ylabel("True Positive")
    plt.title(eos)
    plt.grid(ls='--')
    plt.ylim([0.82,1.02])
    plt.xlim([0,0.2])
    plt.yticks(np.linspace(0.825,1,8))

    plt.savefig(figname+'.png',dpi=200,bbox_inches='tight')
    plt.close()
    return np.asarray(FP), np.asarray(TP), threshold

def ROC_NS_all_EOS(EOS,figname = "ROC_NS_all_EOS"):

    plt.figure()

    for eos in EOS:
        KNN = KNNclassy.ClassificationKNN()
        KNN.load_original_dataset(pathData,eos+"/EMB/original_data_"+eos+"_s300_f0d7.csv")
        KNN.loadModel("models_eos/", "knn_3cat_eos_"+eos)
        KNN.compute_metrics()

        allprob = KNN.metrics["probab"]
        v_prob_NS = 1-allprob[:,0]
        events_have_NS = np.where((KNN.label_test==1) | (KNN.label_test==2))[0]
        N = len(events_have_NS)
        M = len(KNN.label_test) - N
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


        plt.plot(FP,TP,linewidth= 0.5,linestyle = 'solid', label = eos)

    plt.xlabel("False Positive")
    plt.ylabel("True Positive")
    plt.grid(ls='--')
    plt.ylim([0.82,1.02])
    plt.xlim([0,0.2])
    plt.yticks(np.linspace(0.825,1,8))
    plt.legend(bbox_to_anchor =(1.45,1.5))

    plt.savefig(figname+'.png',dpi=200,bbox_inches='tight')
    plt.close()
    return

def ROC_REM_all_EOS(EOS,figname = "ROC_REM_all_EOS"):

    plt.figure()

    for eos in EOS:
        KNN = KNNclassy.ClassificationKNN()
        KNN.load_original_dataset(pathData,eos+"/EMB/original_data_"+eos+"_s300_f0d7.csv")
        KNN.loadModel("models_eos/", "knn_3cat_eos_"+eos)
        KNN.compute_metrics()

        allprob = KNN.metrics["probab"]
        v_prob_REM = allprob[:,2]
        events_have_REM = np.where(KNN.label_test==2)[0]
        N = len(events_have_REM)
        M = len(KNN.label_test) - N
        threshold = np.linspace(0,1,101)[1:-1]
        TP = np.zeros(99)
        FP = np.zeros(99)
        i=0
        for thr in threshold:
            index_say_yes = np.where(v_prob_REM>=thr)[0]
            count_yes = 0.0; count_no = 0.0
            for index in index_say_yes:
                if index in events_have_REM:
                    count_yes=count_yes + 1.0
                else:
                    count_no=count_no + 1.0
            TP[i] = count_yes/N
            FP[i] = count_no/M

            i = i + 1

        plt.plot(FP,TP,linewidth= 0.5,linestyle = 'solid', label = eos)

    plt.xlabel("False Positive")
    plt.ylabel("True Positive")
    plt.grid(ls='--')
    plt.ylim([0.82,1.02])
    plt.xlim([0,0.2])
    plt.yticks(np.linspace(0.825,1,8))
    plt.legend(bbox_to_anchor =(1.3,1))

    plt.savefig(figname+'.png',dpi=200,bbox_inches='tight')
    plt.close()
    return

import importlib
importlib.reload(KNNclassy)


pathClassy = "/home/miquelmiravet/University/Doctorat/Projects/IPAM_ML/KNN_miq/"
pathData = "/home/miquelmiravet/University/Doctorat/Projects/IPAM_ML/KNN_miq/input/"

EOS = ["APR4_EPP", "BHF_BBB2", "H4", "HQC18", "KDE0V", "KDE0V1", "MPA1", "MS1_PP", "MS1B_PP", "RS", "SK255", "SK272", "SKI2", "SKI3", "SKI4", "SKI5", "SKI6", "SKMP", "SKOP", "SLy",
"SLY2", "SLY9", "SLY230A"]
BayesFactor = [1.526, 1.555, 0.056, 1.422, 1.177, 1.283, 0.276, 0.001, 0.009, 0.176, 0.179, 0.159, 0.108, 0.107, 0.33, 0.025, 0.288, 0.29, 0.618, 1.0, 1.028, 0.37, 0.932]

EOSdic = {}
count = 0

for eos in EOS:
    print("Doing", eos)
    KNN = KNNclassy.ClassificationKNN()
    KNN.load_original_dataset(pathData,eos+"/EMB/original_data_"+eos+"_s300_f0d7.csv")
    KNN.loadModel("models_eos/", "knn_3cat_eos_"+eos)
    new_dic = {}
    new_dic['knn'] = KNN    
    new_dic['weight'] = BayesFactor[count]
    EOSdic[eos] = new_dic
    count = count + 1 

    KNN.compute_metrics()

    hist_NS(KNN,'figsKNN_eos/histNS/hist_NS_eos_'+eos)
    hist_REM(KNN,'figsKNN_eos/histREM/hist_REM_eos_'+eos)
    EOSdic[eos]['NS_FP'], EOSdic[eos]['NS_TP'], EOSdic[eos]['NS_thr'] = ROC_NS(KNN, thr_wanted = [],figname='figsKNN_eos/ROC_NS/ROC_NS_eos_'+eos)
    EOSdic[eos]['REM_FP'], EOSdic[eos]['REM_TP'], EOSdic[eos]['REM_thr'] = ROC_REM(KNN, thr_wanted = [],figname='figsKNN_eos/ROC_REM/ROC_REM_eos_'+eos)
    
    with open('ROC_CURVES_NS/NS_ROC_'+eos+'.txt','w') as f:
        f.write('#FP \t TP \t Thr\n')
        FP_NS =  EOSdic[eos]['NS_FP']
        TP_NS =  EOSdic[eos]['NS_TP']
        thr_NS =  EOSdic[eos]['NS_thr']
        for i in range(0,len(FP_NS)):
            f.write('%f \t %f \t %f \n'%(FP_NS[i],TP_NS[i],thr_NS[i]))
        
    with open('ROC_CURVES_REM/REM_ROC_'+eos+'.txt','w') as f:
        f.write('#FP  TP  Thr\n')
        FP_REM =  EOSdic[eos]['REM_FP']
        TP_REM =  EOSdic[eos]['REM_TP']
        thr_REM =  EOSdic[eos]['REM_thr']
        for i in range(0,len(FP_REM)):
            f.write('%f \t %f \t %f \n'%(FP_REM[i],TP_REM[i],thr_REM[i]))


ROC_NS_all_EOS(EOS,"figsKNN_eos/ROC_NS/ROC_NS_all_EOS")
ROC_REM_all_EOS(EOS,"figsKNN_eos/ROC_REM/ROC_REM_all_EOS")
