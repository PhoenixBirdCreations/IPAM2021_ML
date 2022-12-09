import KNNclassy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
#import importlib
#importlib.reload(KNNclassy)

pathREM = "/home/miquelmiravet/University/Doctorat/Projects/IPAM_ML/KNN_miq/ipam_REM_set"
pathBNS = "/home/miquelmiravet/University/Doctorat/Projects/IPAM_ML/KNN_miq/ipam_NS_set"
pathClassy = "/home/miquelmiravet/University/Doctorat/Projects/IPAM_ML/KNN_miq/"
pathData = "/home/miquelmiravet/University/Doctorat/Projects/IPAM_ML/KNN_miq/input/"
#eos = "APR4_EPP"

#KNN = KNNclassy.ClassificationKNN()
#KNN.load_original_dataset(pathData,eos+"/EMB/original_data_"+eos+"_s300_f0d7.csv")

#KNN.CrossVal()

#KNN.build_train_model(KNN.optimal["k"], KNN.optimal["metric"], KNN.optimal["algo"],KNN.optimal["weight"])
#KNN.saveModel(pathClassy,'knn_3cat_eos_APR4_EPP')

EOS = ["APR4_EPP", "BHF_BBB2", "H4", "HQC18", "KDE0V", "KDE0V1", "MPA1", "MS1_PP", "MS1B_PP", "RS", "SK255", "SK272", "SKI2", "SKI3", "SKI4", "SKI5", "SKI6", "SKMP", "SKOP", "SLy", "SLY2", "SLY9", "SLY230A"]

random.seed(42)
plt.rcParams["font.size"]=14

score_list=[]

for eos in EOS:

    print("Doing", eos)
    KNN = KNNclassy.ClassificationKNN()
    KNN.load_original_dataset(pathData,eos+"/EMB/original_data_"+eos+"_s300_f0d7.csv")

    KNN.CrossVal()

    KNN.build_train_model(KNN.optimal["k"], KNN.optimal["metric"], KNN.optimal["algo"],KNN.optimal["weight"])
    KNN.saveModel(pathClassy,'knn_3cat_eos_'+eos)
    KNN.compute_metrics()
    
    score_list.append(KNN.metrics["score"])
    print("#"*60)

print(score_list)
