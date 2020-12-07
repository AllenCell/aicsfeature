import numpy as np
import pandas as pd
from igraph import *
from sklearn.decomposition import PCA
import scipy.spatial.distance as scispatial

def get_laplacian_score(df_input, params):

    k = 4 if 'k' not in params else params['k']

    scale = None if 'scale' not in params else params['scale']

    '''
         Implementation of the learning-independent feature selection
         algorithm described in [1]. The importance of a feature is
         evaluated by its power of locality preserving.

         [1] - Laplacian score for feature selection. Xiaofei He et all.
    '''
    X = df_input.values.copy()
    # Distance matrix
    D = scispatial.cdist(X,X)
    '''
        Scale is given by the mean of distance matrix in case no other
        value is provided.
    '''
    if not scale:
        scale = D.mean()
    '''
        Create the k-neighbors graph
    '''
    n = X.shape[0]
    g = Graph(directed=False)
    g.add_vertices(n)
    '''
        Weight matrix based on observations pairwise distance.
    '''
    S = np.zeros((n,n), dtype=np.float32)
    for i in range(n):
        d = D[i]
        neighs = d.argsort()
        edges = [(i,j) for j in neighs[1:k+1]]
        for edge in edges:
            j = edge[1]
            dij = d[j]
            wij = np.exp(-(dij*dij)/scale)
            S[i,j] = wij
            S[j,i] = wij
        g.add_edges(edges)
    '''
        Either do a per connected component analysis if graph is not conencted
        or increase the value of k.
    '''
    assert g.is_connected() == True
    '''
        Diagonal, Ones and Laplacian matrix
    '''
    D = np.diag(S.sum(axis=1))
    O = np.ones((n,1))
    L = D - S
    '''
        Compute the score for each feature. The efficiency can be increased by
        using matrix equation here.
    '''
    score = []
    for fid in range(X.shape[1]):
        f = X[:,fid].copy()
        beta_nom = np.matmul(np.matmul(f.reshape(1,-1),D),O).flatten()
        beta_den = np.matmul(np.matmul(O.reshape(1,-1),D),O).flatten()
        f -= beta_nom/beta_den
        beta_nom = np.matmul(np.matmul(f.reshape(1,-1),L),f).flatten()
        beta_den = np.matmul(np.matmul(f.reshape(1,-1),D),f).flatten()
        score.append(beta_nom/beta_den)
    
    '''
        We invert the score so that important features have higher scores. Also
        normalize the score in the range [0,1].
    '''
    
    score = np.max(score)-score
    
    score = (score-np.min(score))/(np.max(score)-np.min(score))
    
    return score.reshape(-1), (scale,g,L)

def get_pca_score(df_input):
    X = df_input.values.copy()
    '''
        PCA from sklearn
    '''
    pca = PCA().fit(X)
    loading = pca.components_.T * np.sqrt(pca.explained_variance_)
    '''
        Decaying weight to give more importance to features that
        contribute to the first components.
    '''
    weight = np.linspace(0,1,loading.shape[1],endpoint=True).reshape(1,-1)
    weight = np.repeat(weight,loading.shape[0],axis=0)
    weight = np.exp(-5.0*weight)
    
    score = (loading*weight).max(axis=1)
    '''
        We normalize the score in the range [0,1].
    '''
    score = (score-np.min(score))/(np.max(score)-np.min(score))
    
    return score

def rank_features(df_input, info):
    
    score = np.zeros(df_input.shape[1], dtype=np.float32)
       
    '''
        Laplacian score
    '''
    if 'laplacian' in info:
        score_lap, _ = get_laplacian_score(df_input, info['laplacian'])
        score += score_lap
    '''
        PCA score
    '''
    if 'pca' in info:
        score_pca = get_pca_score(df_input)
        score += score_pca
       
    return [(df_input.columns[i],score[i]) for i in score.argsort().flatten()[::-1]]