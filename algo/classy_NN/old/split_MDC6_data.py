import numpy as np

# m1_inj    0
# m2_inj    1 
# chi1_inj  2
# chi2_inj  3
# Mc_inj    4   
# m1_rec    5
# m2_rec    6 
# chi1_rec  7
# chi2_rec  8
# Mc_rec    9   

def split_MDC6_data(X, features='mass&spin', m1_cutoff=None, Mc_min=None):
    if features=='all':
        inj = X[:,0:5]
        rec = X[:,5:10]
        names = [r'$m_1$', r'$m_2$', r'$\chi_1$', r'$\chi_2$', r'${\cal M}_c$']
        
    elif features=='mass&spin':
        m1_inj          = X[:, 0]
        chi1_inj        = X[:, 2]
        chi2_inj        = X[:, 3]
        mc_inj          = X[:, 4]
        m1_rec          = X[:, 5]
        chi1_rec        = X[:, 7]
        chi2_rec        = X[:, 8]
        mc_rec          = X[:, 9]
        inj = np.column_stack((m1_inj,chi1_inj,chi2_inj,mc_inj))
        rec = np.column_stack((m1_rec,chi1_rec,chi2_rec,mc_rec))
        names = [r'$m_1$', r'$\chi_1$', r'$\chi_2$', r'${\cal M}_c$']
    
    elif features=='m1m2chi1chi2':
        m1_inj          = X[:, 0]
        m2_inj          = X[:, 1]
        chi1_inj        = X[:, 2]
        chi2_inj        = X[:, 3]
        m1_rec          = X[:, 5]
        m2_rec          = X[:, 6]
        chi1_rec        = X[:, 7]
        chi2_rec        = X[:, 8]
        inj = np.column_stack((m1_inj,m2_inj,chi1_inj,chi2_inj))
        rec = np.column_stack((m1_rec,m2_rec,chi1_rec,chi2_rec))
        names = [r'$m_1$', r'$m_2$', r'$\chi_1$', r'$\chi_2$']
    
    
    if m1_cutoff is not None:
        mask = np.argwhere(m1_inj>m1_cutoff)
        #ID  = np.delete(ID , mask)
        #snr = np.delete(snr, mask)
        inj = np.delete(inj, mask, axis=0)
        rec = np.delete(rec, mask, axis=0)
    
    if Mc_min is not None:
        mask = np.argwhere(mc_inj<Mc_min)
        #ID  = np.delete(ID , mask)
        #snr = np.delete(snr, mask)
        inj = np.delete(inj, mask, axis=0)
        rec = np.delete(rec, mask, axis=0)
    
    out          = {}
    out['inj']   = inj
    out['rec']   = rec
    out['SNR']   = []
    out['names'] = names
    out['ID']    = []
    return out
