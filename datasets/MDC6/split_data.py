import csv
import numpy as np

# Columns of MDC6/output.txt:

#time_H          0
#time_V          1
#time_L          2
#mass1_inj       3
#mass2_inj       4
#spin1z_inj      5
#spin2z_inj      6
#Mc_inj          7
#z_inj           8
#distance_inj    9 
#event          10
#superevent     11
#gpstime        12
#mass1          13
#mass2          14 
#spin1z         15
#spin2z         16
#snr_H1         17
#snr_L1         18
#snr_V1         19
#combined_snr   20
#combined_far   21
#IFOS           22
#mchirp         23
#pipeline       24

def extractData(filename, verbose=False):
    lst=[]
    header = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        header.append(next(csv_reader))
        for row in csv_reader:
            lst.append(row)
        data=np.array(lst)
    if verbose:
        print(filename, 'loaded')
    return header, data

# Read header information and data
header, test_data = extractData('output.csv')

# Read the column with pipeline information and make list of indices for each pipeline
pipelines      = test_data[:,-1]
inds = {}
inds['pycbc']  = [ind for ind,_ in enumerate(pipelines) if pipelines[ind]=='pycbc']
inds['gstlal'] = [ind for ind,_ in enumerate(pipelines) if pipelines[ind]=='gstlal']
inds['mbta']   = [ind for ind,_ in enumerate(pipelines) if pipelines[ind]=='MBTAOnline']

pips = list(inds.keys())

#Using the indices separate data for each pipeline m1,m2,chi1,chi2
# y_ is the injected values and x_ the recovered
inj_idx  = [3,4,5,6,7]
rec_idx  = [13,14,15,16,23]
x = {}
y = {}
header = '# 0-m1_inj, 1-m2_inj, 2-chi1_inj, 3-chi2_inj, 4-Mc_inj, 5-m1_rec, 6-m2_rec, 7-chi1_rec, 8-chi2_rec, 9-Mc_rec'
for k in pips:
    tmp  = test_data[inds[k],:]    
    y[k] = tmp[:, inj_idx].astype(float)
    x[k] = tmp[:, rec_idx].astype(float)
    X = np.concatenate( (y[k],x[k]), axis=1)
    np.savetxt(k+'.txt', X, fmt='%25.18e', header=header)

