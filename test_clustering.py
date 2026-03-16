# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:15:57 2026

@author: granged
"""

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score, homogeneity_completeness_v_measure
import pandas as pd
import numpy as np
from pathlib import Path
import anndata as ad
import snf

def Fused_dist(D1,D2, K=20, mu=0.5):
    
    W1 = snf.make_affinity(D1, K=20, mu=0.5)  # K=neighbors; mu=local scaling
    W2 = snf.make_affinity(D2, K=20, mu=0.5)
    
    # Fuse the two affinities:
    Wf = snf.snf([W1, W2], K=20, t=20)  # t = diffusion steps
    
    #Turn fused similarity into a distance:
    Df = 1.0 - Wf  
    
    np.fill_diagonal(Df, 0.0)
    
    return Df

#%%
ROOT = Path("E:/granged/data/HTAN/atlas/")
output_folder = ROOT / 'output'

dataset = 'lung'
dataset='breast'
#cell_type = 'natural killer cell'
#cell_type = 'neutrophil'
#cell_type = 'endothelial cell'
#cell_type = 'epithelial cell'
#cell_type = 'malignant cell'
#cell_type = 'fibroblast'
#cell_type = 'endothelial'
#cell_type = 'macrophage'
cell_type = 'malignant cell'
cell_type = 'vein endothelial cell'
cell_type = 'exhausted T cell'
cell_type = 'naive T cell'

# cell_df = ad.read_h5ad(output_folder / dataset / 'cells' / (cell_type + '.h5ad'))
# X = cell_df.X.toarray()
# np.save(output_folder / dataset / 'feats' / (cell_type + '.csv'), X)


#cell_filename = output_folder / dataset / 'cells'
dist_filename = output_folder /  dataset / 'dists' / (cell_type + '.npz')
clin_filename = output_folder / dataset / 'clin' / (dataset + '_clin.csv')
#fuse_filename = output_folder /  'breast' / 'dists' / 'similarity_fused_similarity_max12nn_E_BW_dist.csv'


clin_df = pd.read_csv(clin_filename, index_col=0)
dist_dict = dict(np.load(dist_filename, allow_pickle=True))

dist_name_list = ['Bures_Wasserstein', 'Wasserstein', 'Frobenius', 'Euclidean', 'Fused']
patient_list = [str(patient) for patient in dist_dict['patient_list']]
patient_disease_list = clin_df.loc[patient_list, 'disease']


fused = Fused_dist(dist_dict['Bures_Wasserstein'], dist_dict['Euclidean'])

dist_dict['Fused'] = fused

#%%
# subtypes = ['squamous cell lung carcinoma', 'lung adenocarcinoma']

# clin_df = clin_df.loc[[disease in subtypes for disease in clin_df['disease']]]
# subtype_patients = clin_df.index

# keep_patients = [patient in subtype_patients for patient in patient_list]
# patient_disease_list = patient_disease_list[keep_patients]
# n_diseases = len(patient_disease_list.unique())

# for dist_name in dist_name_list:
#     dist = dist_dict[dist_name]
#     dist_dict[dist_name] = dist[keep_patients][:,keep_patients]
#%%

mapping = {
    "HER2 positive breast carcinoma": "HER2-positive",
    "estrogen-receptor positive breast cancer": "HR-positive (ER+)",
    "triple-negative breast carcinoma": "Triple-negative",
    "invasive ductal breast carcinoma": "Histologic subtype (other)",
    "invasive lobular breast carcinoma": "Histologic subtype (other)",
    "invasive tubular breast carcinoma || invasive lobular breast carcinoma": "Histologic subtype (other)",
    "breast mucinous carcinoma": "Histologic subtype (other)",
    "metaplastic breast carcinoma": "Histologic subtype (other)",
    "breast apocrine carcinoma": "Histologic subtype (other)",
    "breast cancer": "Unspecified/other",
    "breast carcinoma": "Unspecified/other"
}

int_mapping = {name: i for i, name in enumerate(list(set(mapping.values())))}


#patient_disease_list = [int_mapping[mapping[disease]] for disease in patient_disease_list]


#%%

from typing import Union, Sequence, Literal
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

def cluster_from_distance(
    dist: Union[np.ndarray, Sequence[float]],
    n_classes: int,
    linkage_method: Literal["single", "complete", "average", "weighted", "ward", "centroid", "median"] = "average"
) -> np.ndarray:
    """
    Cluster points into a specified number of classes using a precomputed distance matrix.

    Parameters
    ----------
    dist : array-like
        Either:
          - A square (n x n) distance matrix (zeros on diagonal, symmetric), or
          - A condensed distance vector of length n*(n-1)/2 as used by scipy.spatial.distance.squareform.
    n_classes : int
        The desired number of clusters (k >= 1).
    linkage_method : {'single','complete','average','weighted','ward','centroid','median'}, optional
        Linkage strategy for hierarchical clustering. Default is 'average'.
        Note: 'ward' expects Euclidean distances derived from points; it can still be used with distances
        but is only strictly correct for Euclidean pairwise distances.

    Returns
    -------
    labels : np.ndarray, shape (n_samples,)
        Cluster labels in 1..n_classes, aligned with the original order of items.

    Raises
    ------
    ValueError
        If input shapes are invalid or n_classes < 1.
    """
    if n_classes < 1:
        raise ValueError("n_classes must be >= 1")

    dist = np.asarray(dist)

    # Normalize to condensed form required by scipy.cluster.hierarchy.linkage
    if dist.ndim == 2:
        if dist.shape[0] != dist.shape[1]:
            raise ValueError("Square distance matrix must be n x n.")
        # Basic sanity checks
        if not np.allclose(np.diag(dist), 0, atol=1e-12):
            raise ValueError("Square distance matrix must have zeros on the diagonal.")
        if not np.allclose(dist, dist.T, atol=1e-12):
            raise ValueError("Square distance matrix must be symmetric.")
        condensed = squareform(dist, checks=False)
        n = dist.shape[0]
    elif dist.ndim == 1:
        condensed = dist
        # Infer n from length of condensed vector: m = n*(n-1)/2
        m = len(condensed)
        # Solve n^2 - n - 2m = 0
        n = int((1 + np.sqrt(1 + 8*m)) / 2)
        if n * (n - 1) // 2 != m:
            raise ValueError("Invalid condensed distance vector length.")
    else:
        raise ValueError("dist must be either a square matrix (2D) or a condensed vector (1D).")

    # Compute linkage from distances
    Z = linkage(condensed, method=linkage_method)

    # Cut the dendrogram to obtain exactly n_classes clusters
    labels = fcluster(Z, t=n_classes, criterion="maxclust")

    # labels are 1..n_classes; length n
    if labels.shape[0] != n:
        raise RuntimeError("Unexpected label length mismatch.")

    return labels
#%%        

for dist_name in dist_name_list:
    dist_matrix = dist_dict[dist_name]
    
    #print(dist_matrix.shape)
    
    labels = cluster_from_distance(dist_matrix, n_diseases, linkage_method='average')
    
    # Test results
    
    nmi_score = normalized_mutual_info_score(labels, patient_disease_list)
    ar_score = adjusted_rand_score(labels, patient_disease_list)
    fm_score = fowlkes_mallows_score(labels, patient_disease_list)
    _,_,jc_score = homogeneity_completeness_v_measure(labels, patient_disease_list)
    
    print(f'{dist_name}\n Normalized_mutual information: {nmi_score}\n Adjusted Rand Score: {ar_score}\n Fowlkes Mallows Score: {fm_score}\n V Score: {jc_score}')
    
    
            