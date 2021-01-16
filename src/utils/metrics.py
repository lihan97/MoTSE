import numpy as np
from math import sqrt
from sklearn.metrics import pairwise, auc, precision_recall_curve
from scipy.special import betainc
from scipy import stats
def cosine_similarity(a,b):
    a = np.array(a)
    b = np.array(b)
    return pairwise.cosine_similarity([a,b])[0,1]

def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs

def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp

def auprc(y, f):
    precision, recall, thresholds = precision_recall_curve(y, f)
    auprc_ = auc(recall, precision)
    return auprc_

def pearson_matrix(matrix):
    r = np.corrcoef(matrix)
    rf = r[np.triu_indices(r.shape[0], 1)]
    df = matrix.shape[1] - 2
    ts = rf * rf * (df / (1 - rf * rf))
    pf = betainc(0.5 * df, 0.5, df / (df + ts))
    p = np.zeros(shape=r.shape)
    p[np.triu_indices(p.shape[0], 1)] = pf
    p[np.tril_indices(p.shape[0], -1)] = pf
    p[np.diag_indices(p.shape[0])] = np.ones(p.shape[0])
    return (r, p)